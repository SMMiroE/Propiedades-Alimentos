import streamlit as st
import numpy as np
from scipy.special import jv
import pandas as pd

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Calculadora de Procesos Térmicos en Alimentos",
    layout="wide"
)

# --- Constantes globales para la ecuación de fracción de hielo y PMs ---
L_molar_fusion_agua = 6010.0 # J/mol (aproximación de 333.6 J/g * 18.015 g/mol)
R_gas = 8.314 # J/(mol·K) - Constante universal de los gases
T0_ref = 273.15 # K (0°C) - Temperatura de fusión del hielo puro
PM_agua = 18.015 # g/mol (o kg/kmol) - Peso molecular del agua

# --- Funciones de Choi y Okos (Propiedades termofísicas) ---

# Propiedades del Agua (valores para agua líquida y hielo)
def densidad_agua(t):
    """Calcula la densidad del agua/hielo en kg/m³ para una temperatura t en °C."""
    if t >= 0: # Agua líquida
        return 997.18 + 3.1439e-3 * t - 3.7574e-3 * t**2
    else: # Hielo
        return 916.89 - 0.13071 * t

def cp_agua(t):
    """Calcula el calor específico del agua/hielo en J/(kg·K) para una temperatura t en °C."""
    if t >= 0: # Agua líquida
        return 4176.2 - 9.0864e-2 * t + 5.4731e-3 * t**2
    else: # Hielo
        return 2062.3 + 6.0769 * t

def k_agua(t):
    """Calcula la conductividad térmica del agua/hielo en W/(m·K) para una temperatura t en °C."""
    if t >= 0: # Agua líquida
        return 0.57109 + 1.7625e-3 * t - 6.7036e-6 * t**2
    else: # Hielo
        return 2.2196 - 6.2489e-3 * t + 1.0154e-4 * t**2

# Propiedades de otros componentes (válidas para el rango de temperatura de Choi y Okos)
def densidad_proteina(t): return 1329.9 - 0.5184 * t
def cp_proteina(t): return 2008.2 + 1.2089 * t - 1.3129e-3 * t**2
def k_proteina(t): return 0.17881 + 1.1958e-3 * t - 2.7178e-6 * t**2

def densidad_grasa(t): return 925.59 - 0.41757 * t
def cp_grasa(t): return 1984.2 + 1.4733 * t - 4.8008e-3 * t**2
def k_grasa(t): return 0.18071 - 2.7604e-4 * t - 1.7749e-7 * t**2

def densidad_carbohidrato(t): return 1599.1 - 0.31046 * t
def cp_carbohidrato(t): return 1548.8 + 1.9625 * t - 5.9399e-3 * t**2
def k_carbohidrato(t): return 0.20141 + 1.3874e-3 * t - 4.3312e-6 * t**2

def densidad_fibra(t): return 1311.5 - 0.36589 * t
def cp_fibra(t): return 1845.9 + 1.8306 * t - 4.6509e-3 * t**2
def k_fibra(t): return 0.18331 + 1.2497e-3 * t - 3.1683e-6 * t**2

def densidad_cenizas(t): return 2423.8 - 0.28063 * t
def cp_cenizas(t): return 1092.6 + 1.8896 * t - 3.6817e-3 * t**2
def k_cenizas(t): return 0.32962 + 1.4011e-3 * t - 2.9069e-6 * t**2

# Función para calcular la fracción de hielo (Xi) y agua no congelada (Xu)
def calcular_fraccion_hielo(t_celsius, agua_porcentaje_inicial, Tf_input_celsius):
    """
    Calcula la fracción de hielo (Xi) y agua no congelada (Xu)
    usando la ecuación de la depresión del punto de congelación para XA.

    Args:
        t_celsius (float): Temperatura actual del alimento en °C.
        agua_porcentaje_inicial (float): Porcentaje inicial de agua en el alimento (0-100).
        Tf_input_celsius (float): Temperatura inicial de congelación del alimento en °C.

    Returns:
        tuple: (Xi, Xu) Fracciones de masa de hielo y agua no congelada (0-1) respecto a la masa total del alimento.
    """
    T_kelvin = t_celsius + 273.15
    Tf_input_kelvin = Tf_input_celsius + 273.15

    agua_total_fraccion_masa = agua_porcentaje_inicial / 100.0

    if t_celsius >= Tf_input_celsius:
        # No hay hielo si la temperatura es igual o superior al punto de congelación inicial
        Xi = 0.0
        Xu = agua_total_fraccion_masa
    else:
        # Calcular la fracción molar de agua no congelada (XA)
        if T_kelvin <= 0: # Para evitar log(0) o divisiones por 0 o valores muy pequeños que puedan dar inf
            XA_fraccion_molar = 0.0 # Completamente congelado si T es 0K o menos (teórico)
        else:
            try:
                # Ecuación: ln(XA) = (lambda/R) * (1/T0 - 1/T)
                ln_XA = (L_molar_fusion_agua / R_gas) * ((1 / T0_ref) - (1 / T_kelvin))
                XA_fraccion_molar = np.exp(ln_XA)
            except OverflowError:
                XA_fraccion_molar = 0.0 # Si el valor es muy pequeño, exp(ln_XA) puede ser 0
            except Exception as e:
                st.warning(f"Advertencia en el cálculo de fracción molar de agua no congelada (XA): {e}. Asumiendo 0% agua no congelada.")
                XA_fraccion_molar = 0.0 # En caso de error, asumir completamente congelado

        # Asegurarse de que XA_fraccion_molar esté en el rango [0, 1]
        XA_fraccion_molar = max(0.0, min(1.0, XA_fraccion_molar))

        # La XA_fraccion_molar obtenida es la actividad del agua. Asumimos que esta actividad
        # representa la fracción de agua no congelada respecto a la cantidad total de agua inicial.
        Xu = XA_fraccion_molar * agua_total_fraccion_masa

        # Fracción de hielo (Xi) sobre masa total de alimento
        Xi = agua_total_fraccion_masa - Xu

        # Asegurarse de que Xi sea no negativo y Xu no exceda el agua total
        Xi = max(0.0, Xi)
        Xu = min(agua_total_fraccion_masa, Xu)
        # Ajuste para consistencia: si Xi > 0, Xu debe ser el complemento
        if Xi > 0:
            Xu = agua_total_fraccion_masa - Xi
        else: # Si no hay hielo, toda el agua está no congelada
            Xu = agua_total_fraccion_masa


    return Xi, Xu

# Funciones principales para calcular propiedades del alimento
def calcular_densidad_alimento(t, composicion, Tf_input):
    """Calcula la densidad del alimento en kg/m³ a una temperatura t en °C."""
    agua_porcentaje = composicion['agua']
    Xi, Xu = calcular_fraccion_hielo(t, agua_porcentaje, Tf_input)

    # Convertir a fracciones de masa
    f_p = composicion['proteina'] / 100
    f_g = composicion['grasa'] / 100
    f_c = composicion['carbohidrato'] / 100
    f_f = composicion['fibra'] / 100
    f_z = composicion['cenizas'] / 100

    rho_inv = (Xu / densidad_agua(t)) + \
              (Xi / densidad_agua(t - 0.0001)) + \
              (f_p / densidad_proteina(t)) + \
              (f_g / densidad_grasa(t)) + \
              (f_c / densidad_carbohidrato(t)) + \
              (f_f / densidad_fibra(t)) + \
              (f_z / densidad_cenizas(t))
    
    if rho_inv == 0: return 0
    return 1 / rho_inv

def calcular_cp_alimento(t, composicion, Tf_input):
    """Calcula el calor específico del alimento en J/(kg·K) a una temperatura t en °C."""
    agua_porcentaje = composicion['agua']
    Xi, Xu = calcular_fraccion_hielo(t, agua_porcentaje, Tf_input)

    f_p = composicion['proteina'] / 100
    f_g = composicion['grasa'] / 100
    f_c = composicion['carbohidrato'] / 100
    f_f = composicion['fibra'] / 100
    f_z = composicion['cenizas'] / 100

    cp_val = (Xu * cp_agua(t)) + \
             (Xi * cp_agua(t - 0.0001)) + \
             (f_p * cp_proteina(t)) + \
             (f_g * cp_grasa(t)) + \
             (f_c * cp_carbohidrato(t)) + \
             (f_f * cp_fibra(t)) + \
             (f_z * cp_cenizas(t))
    return cp_val

def calcular_k_alimento(t, composicion, Tf_input):
    """Calcula la conductividad térmica del alimento en W/(m·K) a una temperatura t en °C."""
    agua_porcentaje = composicion['agua']
    Xi, Xu = calcular_fraccion_hielo(t, agua_porcentaje, Tf_input)

    f_p = composicion['proteina'] / 100
    f_g = composicion['grasa'] / 100
    f_c = composicion['carbohidrato'] / 100
    f_f = composicion['fibra'] / 100
    f_z = composicion['cenizas'] / 100

    k_val = (Xu * k_agua(t)) + \
            (Xi * k_agua(t - 0.0001)) + \
            (f_p * k_proteina(t)) + \
            (f_g * k_grasa(t)) + \
            (f_c * k_carbohidrato(t)) + \
            (f_f * k_fibra(t)) + \
            (f_z * k_cenizas(t))
    return k_val

def calcular_alpha_alimento(t, composicion, Tf_input):
    """Calcula la difusividad térmica del alimento en m²/s a una temperatura t en °C."""
    densidad = calcular_densidad_alimento(t, composicion, Tf_input)
    cp = calcular_cp_alimento(t, composicion, Tf_input)
    k = calcular_k_alimento(t, composicion, Tf_input)
    if densidad * cp == 0:
        return 0
    return k / (densidad * cp)

# --- Funciones de Cálculo de Procesos ---

# Coeficientes A1 y lambda1 para Heisler (Primer término)
def get_heisler_coeffs(geometry, bi):
    """
    Obtiene los coeficientes A1 y lambda1 para el primer término de la serie de Heisler
    basados en la geometría y el número de Biot.
    Estos valores provienen de tablas de soluciones para la conducción transitoria.
    """
    if geometry == 'Placa Plana':
        if bi <= 0.01: return 1.0000, 0.0998
        if bi <= 0.02: return 1.0000, 0.1412
        if bi <= 0.03: return 1.0001, 0.1730
        if bi <= 0.04: return 1.0002, 0.1994
        if bi <= 0.05: return 1.0002, 0.2231
        if bi <= 0.06: return 1.0003, 0.2446
        if bi <= 0.07: return 1.0004, 0.2647
        if bi <= 0.08: return 1.0005, 0.2836
        if bi <= 0.09: return 1.0006, 0.3015
        if bi <= 0.1: return 1.0007, 0.3185
        if bi <= 0.2: return 1.0025, 0.4417
        if bi <= 0.3: return 1.0050, 0.5423
        if bi <= 0.4: return 1.0079, 0.6277
        if bi <= 0.5: return 1.0109, 0.7017
        if bi <= 0.6: return 1.0139, 0.7674
        if bi <= 0.7: return 1.0169, 0.8267
        if bi <= 0.8: return 1.0197, 0.8809
        if bi <= 0.9: return 1.0224, 0.9308
        if bi <= 1.0: return 1.0249, 0.9774
        if bi <= 1.5: return 1.0347, 1.1656
        if bi <= 2.0: return 1.0416, 1.3149
        if bi <= 3.0: return 1.0505, 1.5369
        if bi <= 4.0: return 1.0567, 1.6961
        if bi <= 5.0: return 1.0612, 1.8174
        if bi <= 10.0: return 1.0700, 2.0729
        return 1.2732, 1.5708 # Bi -> Inf aproximación para placa

    elif geometry == 'Cilindro':
        if bi <= 0.01: return 1.0000, 0.1412
        if bi <= 0.02: return 1.0001, 0.1995
        if bi <= 0.03: return 1.0002, 0.2449
        if bi <= 0.04: return 1.0003, 0.2839
        if bi <= 0.05: return 1.0004, 0.3187
        if bi <= 0.06: return 1.0005, 0.3503
        if bi <= 0.07: return 1.0006, 0.3795
        if bi <= 0.08: return 1.0007, 0.4067
        if bi <= 0.09: return 1.0008, 0.4323
        if bi <= 0.1: return 1.0009, 0.4565
        if bi <= 0.2: return 1.0040, 0.6698
        if bi <= 0.3: return 1.0078, 0.8251
        if bi <= 0.4: return 1.0116, 0.9408
        if bi <= 0.5: return 1.0151, 1.0322
        if bi <= 0.6: return 1.0183, 1.1077
        if bi <= 0.7: return 1.0211, 1.1723
        if bi <= 0.8: return 1.0236, 1.2289
        if bi <= 0.9: return 1.0259, 1.2792
        if bi <= 1.0: return 1.0279, 1.3250
        if bi <= 1.5: return 1.0360, 1.5317
        if bi <= 2.0: return 1.0415, 1.6880
        if bi <= 3.0: return 1.0477, 1.9081
        if bi <= 4.0: return 1.0514, 2.0620
        if bi <= 5.0: return 1.0538, 2.1793
        if bi <= 10.0: return 1.0594, 2.4048
        return 1.6018, 2.4048 # Bi -> Inf aproximación para cilindro

    elif geometry == 'Esfera':
        if bi <= 0.01: return 1.0000, 0.1730
        if bi <= 0.02: return 1.0001, 0.2449
        if bi <= 0.03: return 1.0002, 0.2996
        if bi <= 0.04: return 1.0003, 0.3465
        if bi <= 0.05: return 1.0004, 0.3881
        if bi <= 0.06: return 1.0005, 0.4256
        if bi <= 0.07: return 1.0006, 0.4599
        if bi <= 0.08: return 1.0007, 0.4916
        if bi <= 0.09: return 1.0008, 0.5212
        if bi <= 0.1: return 1.0009, 0.5490
        if bi <= 0.2: return 1.0040, 0.8159
        if bi <= 0.3: return 1.0080, 1.0000
        if bi <= 0.4: return 1.0120, 1.1448
        if bi <= 0.5: return 1.0159, 1.2647
        if bi <= 0.6: return 1.0195, 1.3683
        if bi <= 0.7: return 1.0229, 1.4601
        if bi <= 0.8: return 1.0261, 1.5427
        if bi <= 0.9: return 1.0290, 1.6178
        if bi <= 1.0: return 1.0318, 1.6862
        if bi <= 1.5: return 1.0423, 2.0288
        if bi <= 2.0: return 1.0494, 2.2889
        if bi <= 3.0: return 1.0577, 2.5704
        if bi <= 4.0: return 1.0628, 2.7566
        if bi <= 5.0: return 1.0660, 2.8982
        if bi <= 10.0: return 1.0761, 3.2044
        return 1.5708, 3.1416 # Bi -> Inf aproximación para esfera
    return 1.0, 0.0

# Factor de posición X(x/Lc, lambda1) para Heisler
def get_heisler_position_factor(geometry, x_over_Lc, lambda1):
    """
    Calcula el factor de posición X(x/Lc, lambda1) para las ecuaciones de Heisler,
    que representa la relación de temperatura en una posición específica respecto al centro.
    """
    if geometry == 'Placa Plana':
        return np.cos(lambda1 * x_over_Lc)
    elif geometry == 'Cilindro':
        if lambda1 * x_over_Lc == 0:
            return 1.0
        return jv(0, lambda1 * x_over_Lc)
    elif geometry == 'Esfera':
        if lambda1 * x_over_Lc == 0:
            return 1.0
        # Evitar división por cero si lambda1 * x_over_Lc es extremadamente pequeño pero no cero.
        # En la práctica, esto solo ocurre si x_over_Lc es 0 y lambda1 no lo es.
        # Ya se maneja el caso x_over_Lc == 0 con el if anterior.
        try:
            return np.sin(lambda1 * x_over_Lc) / (lambda1 * x_over_Lc)
        except ZeroDivisionError:
            return 1.0 # Límite cuando el denominador tiende a cero es 1.0
    return 1.0

# Calculo de propiedades del alimento (para mostrar al usuario)
def calcular_propiedades_alimento(composicion, T_referencia, Tf_input):
    """
    Calcula la densidad, calor específico, conductividad térmica y difusividad térmica
    del alimento a una temperatura de referencia dada.
    """
    densidad = calcular_densidad_alimento(T_referencia, composicion, Tf_input)
    cp = calcular_cp_alimento(T_referencia, composicion, Tf_input)
    k = calcular_k_alimento(T_referencia, composicion, Tf_input)
    alpha = calcular_alpha_alimento(T_referencia, composicion, Tf_input)
    return densidad, cp, k, alpha

# Calculo de temperatura final en el punto frío (Heisler)
def calcular_temperatura_final_punto_frio(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a):
    """
    Calcula la temperatura en el centro de un alimento después de un tiempo 't'
    usando la aproximación del primer término de Heisler.
    """
    if k_alimento_medio == 0 or h == 0:
        st.error("Error: La conductividad térmica o el coeficiente de convección no pueden ser cero para calcular el Bi. Por favor, revise las propiedades o los datos de entrada.")
        return None

    Lc = dimension_a # Longitud característica (a)
    Bi = (h * Lc) / k_alimento_medio
    Fo = (alpha_alimento_medio * t_segundos) / (Lc**2)

    A1, lambda1 = get_heisler_coeffs(geometria, Bi)

    if Fo < 0.2:
        st.warning(f"Advertencia: El número de Fourier (Fo = {Fo:.3f}) es menor que 0.2. La solución del primer término de la serie de Heisler puede no ser precisa. Considere tiempos de proceso más largos.")

    theta_0 = A1 * np.exp(-(lambda1**2) * Fo)

    T_final_centro = T_medio + theta_0 * (T_inicial_alimento - T_medio)
    return T_final_centro, Fo, Bi, A1, lambda1

# Cálculo del tiempo para alcanzar una temperatura final (Heisler)
def calcular_tiempo_para_temperatura(T_final_alimento, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a):
    """
    Calcula el tiempo necesario para que el centro de un alimento alcance
    una temperatura final deseada, usando la aproximación de Heisler.
    """
    if T_medio == T_inicial_alimento:
        st.error("Error: La temperatura del medio no puede ser igual a la temperatura inicial del alimento para este cálculo.")
        return None, None, None, None, None
    if T_medio == T_final_alimento:
        return 0, 0, 0, 0, 0
    if (T_inicial_alimento - T_medio) == 0:
         st.error("Error: La diferencia entre la temperatura inicial del alimento y la temperatura del medio es cero, lo que impide el cálculo.")
         return None, None, None, None, None
    
    Lc = dimension_a
    Bi = (h * Lc) / k_alimento_medio
    A1, lambda1 = get_heisler_coeffs(geometria, Bi)

    theta_0_target = (T_final_alimento - T_medio) / (T_inicial_alimento - T_medio)

    if theta_0_target <= 0 or theta_0_target >= A1:
         st.error(f"Error: La temperatura final objetivo ({T_final_alimento:.2f}°C) es inalcanzable o ya superada para las condiciones dadas.")
         st.info(f"La relación (Tf-Tinf)/(Ti-Tinf) debe ser mayor a 0 y menor que A1 ({A1:.4f}).")
         return None, None, None, None, None

    try:
        Fo = -np.log(theta_0_target / A1) / (lambda1**2)
    except Exception as e:
        st.error(f"Error al calcular Fo: {e}. Puede que la temperatura objetivo sea inalcanzable con estos parámetros, o los valores de A1/lambda1 no son válidos para Bi extremo.")
        return None, None, None, None, None

    if Fo < 0.2:
        st.warning(f"Advertencia: El número de Fourier calculado (Fo = {Fo:.3f}) es menor que 0.2. La solución del primer término de la serie de Heisler puede no ser precisa.")
    elif Fo < 0: # Esto no debería ocurrir si theta_0_target es positivo y menor que A1
        st.error("Error: El número de Fourier calculado es negativo, lo que indica un problema con las temperaturas de entrada (por ejemplo, el alimento ya está más frío/caliente que el objetivo).")
        return None, None, None, None, None

    t_segundos = (Fo * (Lc**2)) / alpha_alimento_medio
    t_minutos = t_segundos / 60
    return t_minutos, Fo, Bi, A1, lambda1

# Calculo de temperatura en posición específica (Heisler)
def calcular_temperatura_posicion(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a, posicion_x):
    """
    Calcula la temperatura en una posición específica 'x' dentro de un alimento
    después de un tiempo 't', usando la aproximación de Heisler.
    """
    if posicion_x > dimension_a:
        st.error("Error: La posición 'x' no puede ser mayor que la dimensión característica 'a' (radio/semiespesor).")
        return None, None, None, None, None, None
    if dimension_a == 0:
        st.error("Error: La dimensión característica 'a' no puede ser cero.")
        return None, None, None, None, None, None

    Lc = dimension_a
    Bi = (h * Lc) / k_alimento_medio
    Fo = (alpha_alimento_medio * t_segundos) / (Lc**2)

    A1, lambda1 = get_heisler_coeffs(geometria, Bi)

    if Fo < 0.2:
        st.warning(f"Advertencia: El número de Fourier (Fo = {Fo:.3f}) es menor que 0.2. La solución del primer término de la serie de Heisler puede no ser precisa.")

    theta_0 = A1 * np.exp(-(lambda1**2) * Fo)

    x_over_Lc = posicion_x / Lc
    position_factor = get_heisler_position_factor(geometria, x_over_Lc, lambda1)
    theta_x = theta_0 * position_factor

    T_final_x = T_medio + theta_x * (T_inicial_alimento - T_medio)
    return T_final_x, Fo, Bi, A1, lambda1, position_factor

# Cálculo del tiempo de congelación (Plank)
def calcular_tiempo_congelacion_plank(Tf_input, T_ambiente_congelacion, h, k_congelado, L_efectivo, geometria, dimension_a):
    """
    Calcula el tiempo de congelación de un alimento usando la ecuación de Plank.
    """
    if Tf_input <= T_ambiente_congelacion: # Corregido para manejar caso donde Tf es menor o igual a Ta
        st.error("Error: La temperatura inicial de congelación del alimento (Tf) debe ser mayor que la temperatura del medio de congelación (Ta) para que la congelación sea físicamente posible según este modelo.")
        return None, None, None, None

    if geometria == 'Placa Plana':
        P = 0.5
        R = 0.125
    elif geometria == 'Cilindro':
        P = 0.25
        R = 0.0625
    elif geometria == 'Esfera':
        P = 0.1667
        R = 0.0417
    else:
        st.error("Geometría no válida para la ecuación de Plank.")
        return None, None, None, None

    # T_f - T_a es el denominador, ya se chequeó que no sea <= 0
    
    t_segundos = (L_efectivo / (Tf_input - T_ambiente_congelacion)) * \
                 ((P * dimension_a / h) + (R * dimension_a**2 / k_congelado))

    if t_segundos < 0: # Esto no debería ocurrir si Tf > Ta y L_efectivo > 0
        st.error("Error: El tiempo de congelación calculado es negativo. Revise las temperaturas o propiedades.")
        return None, None, None, None

    t_minutos = t_segundos / 60
    return t_minutos, P, R, L_efectivo

# Nueva función para calcular PMs
def calcular_pm_solido_aparente(Tf_input_celsius, agua_porcentaje_inicial):
    """
    Calcula el peso molecular aparente del sólido a partir de la temperatura inicial
    de congelación del alimento (Tf) y su composición de agua, utilizando la relación
    entre fracción molar y fracción másica de agua.
    """
    if agua_porcentaje_inicial >= 100:
        st.error("El alimento no contiene sólidos para calcular su peso molecular. El porcentaje de agua debe ser menor a 100%.")
        return None
    if agua_porcentaje_inicial < 0:
        st.error("El porcentaje de agua no puede ser negativo.")
        return None

    Tf_kelvin = Tf_input_celsius + 273.15

    # Paso 1: Calcular XA (fracción molar de agua no congelada o actividad de agua) a Tf
    # (Usando la ecuación de Clausius-Clapeyron/Raoult)
    try:
        # Si Tf_input_celsius es 0 o muy cerca de 0, Tf_kelvin será 273.15,
        # haciendo (1/T0_ref - 1/Tf_kelvin) cercano a cero, y ln_XA cercano a cero, XA_at_Tf cercano a 1.
        # Si Tf_input_celsius es muy por encima de 0, (1/T0 - 1/T) será negativo, y XA_at_Tf < 1
        # Si Tf_input_celsius es -infinito, (1/T) tiende a 0, XA_at_Tf tiende a exp(L/RT0) que es un valor muy grande.
        # La ecuación de ln XA es válida para T <= T0. Si Tf_input_celsius > 0, esta ecuación no aplica directamente
        # para *definir* la actividad del agua, ya que el punto de congelación ya ha pasado.
        # Sin embargo, si el usuario introduce una Tf > 0 para este cálculo de PMs, debemos considerar.
        
        # Ajuste para evitar divisiones o logaritmos problemáticos
        if Tf_kelvin <= 1e-6: # Evitar división por cero si Tf es muy cercano a 0 K
            st.error("Temperatura de congelación inicial (Tf) demasiado baja (cercana a 0 K). No es un valor físico para alimentos.")
            return None

        # Si Tf_input_celsius > 0, significa que el alimento no congelará a 0C, sino a una temp mayor, lo cual es incorrecto.
        # Por definición, Tf debe ser <= 0°C. Si el usuario ingresa Tf > 0, esto causaría ln_XA a ser negativo, y XA_at_Tf > 1.
        # XA (actividad de agua/fracción molar de agua) no puede ser mayor que 1.
        if Tf_input_celsius > 0:
            st.error(f"La temperatura inicial de congelación (Tf = {Tf_input_celsius:.1f}°C) debe ser menor o igual a 0°C para este cálculo.")
            return None
            
        ln_XA = (L_molar_fusion_agua / R_gas) * ((1 / T0_ref) - (1 / Tf_kelvin))
        XA_at_Tf = np.exp(ln_XA)
    except Exception as e:
        st.error(f"Error al calcular la fracción molar de agua (XA) para PMs: {e}. Verifique la temperatura de congelación inicial (Tf).")
        return None

    # Asegurarse de que XA_at_Tf esté entre 0 y 1. Si Tf_input_celsius > 0, XA_at_Tf podría ser > 1, pero ya se maneja el error.
    XA_at_Tf = max(0.0, min(1.0, XA_at_Tf))

    # Fracciones másicas iniciales
    m_u0 = agua_porcentaje_inicial / 100.0 # Fracción másica de agua inicial
    m_s = 1.0 - m_u0 # Fracción másica de sólidos totales

    # Paso 2: Despejar PM_s de la ecuación XA = (mu/PM_agua) / ((mu/PM_agua) + (ms/PMs))
    # Aquí, mu es la fracción másica de agua no congelada en el punto de congelación inicial (m_u0)

    # Evitar división por cero o problemas si XA_at_Tf es muy cercano a 1 (alimento casi agua pura)
    if (1 - XA_at_Tf) < 1e-9: # Si XA_at_Tf es muy cercano a 1 (alimento es casi agua pura o Tf muy cerca de 0°C)
        st.warning("Advertencia: El PM del sólido tiende a infinito (alimento es casi agua pura o Tf es muy cercana a 0°C).")
        return float('inf') # Retorna infinito si es casi agua pura
    
    if m_u0 == 0:
        st.error("Error: No hay agua en el alimento para calcular la fracción molar de agua no congelada a partir de la depresión crioscópica.")
        return None
    if m_s == 0:
        st.error("Error: No hay sólidos en el alimento para calcular un peso molecular de sólido. El porcentaje de sólidos debe ser mayor a 0%.")
        return None

    try:
        # PM_s = (XA_at_Tf * m_s * PM_agua) / (m_u0 * (1 - XA_at_Tf))
        # Reorganizando:
        # XA = (nu_agua) / (nu_agua + nu_solidos)  donde nu es moles
        # nu_agua = m_u0 / PM_agua
        # nu_solidos = m_s / PM_s
        # XA = (m_u0/PM_agua) / ( (m_u0/PM_agua) + (m_s/PM_s) )
        # Despejando m_s/PM_s:
        # XA * ( (m_u0/PM_agua) + (m_s/PM_s) ) = m_u0/PM_agua
        # XA * (m_s/PM_s) = (m_u0/PM_agua) * (1 - XA)
        # PM_s = (XA * m_s) / ( (m_u0/PM_agua) * (1 - XA) )
        # PM_s = (XA * m_s * PM_agua) / (m_u0 * (1 - XA))
        PM_s = (XA_at_Tf * m_s * PM_agua) / (m_u0 * (1 - XA_at_Tf))

    except Exception as e:
        st.error(f"Error al despejar PM_s: {e}. Puede haber un problema con los valores intermedios (XA o fracciones).")
        return None

    return PM_s


# --- Interfaz de Usuario Streamlit ---

st.title("Calculadora de Procesos Térmicos en Alimentos")
st.markdown("Desarrollada por **Dra. Silvia M. Miro Erdmann**") 

st.markdown("""
Esta aplicación permite calcular propiedades termofísicas de alimentos y simular procesos de calentamiento, enfriamiento y congelación utilizando modelos de la ingeniería de alimentos.
""")

st.sidebar.header("1. Composición del Alimento (%)")
st.sidebar.markdown("Introduce los porcentajes en peso de cada componente. La suma debe ser 100%.")

col1, col2 = st.sidebar.columns(2)
with col1:
    agua = st.number_input("Agua (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
    proteina = st.number_input("Proteína (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
    grasa = st.number_input("Grasa (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
with col2:
    carbohidrato = st.number_input("Carbohidratos (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
    fibra = st.number_input("Fibra (%)", min_value=0.0, max_value=100.0, value=0.5, step=0.1)
    cenizas = st.number_input("Cenizas (%)", min_value=0.0, max_value=100.0, value=0.5, step=0.1)

composicion = {
    'agua': agua,
    'proteina': proteina,
    'grasa': grasa,
    'carbohidrato': carbohidrato,
    'fibra': fibra,
    'cenizas': cenizas
}

total_composicion = sum(composicion.values())

if total_composicion != 100.0:
    st.sidebar.error(f"La suma de los porcentajes es {total_composicion:.1f}%. Debe ser 100%.")
else:
    st.sidebar.success("Suma de composición: 100%. ¡Perfecto!")

st.sidebar.markdown("---")
st.sidebar.header("2. Temperatura de Congelación (Tf)")
Tf_input = st.sidebar.number_input("Temperatura inicial de congelación del alimento (Tf) [ºC]", value=-2.0, step=0.1, help="Punto donde el agua en el alimento comienza a congelarse. Típicamente entre -0.5 y -3 °C.")

# Selección del tipo de cálculo
st.header("3. Elige el cálculo que quieres realizar")
calculation_type = st.radio(
    "Selecciona una opción:",
    (
        "Propiedades a T > 0°C",
        "Propiedades a T < 0°C",
        "Temperatura final en el punto frío (ºC)",
        "Tiempo de proceso para alcanzar una temperatura final (ºC)",
        "Temperatura en una posición específica (X) en el alimento (ºC)",
        "Tiempo de congelación (min)",
        "Peso Molecular Aparente del Sólido (PMs) [g/mol]" # Nueva opción
    )
)

# --- Inputs dinámicos según la selección ---
st.markdown("---")
st.header("4. Parámetros del Cálculo")

if calculation_type == "Propiedades a T > 0°C":
    T_prop = st.number_input("Temperatura de referencia para propiedades [ºC]", value=20.0, step=1.0)
    if T_prop < Tf_input:
        st.warning(f"La temperatura de referencia ({T_prop}ºC) es menor que la temperatura de congelación inicial ({Tf_input}ºC). Considera usar 'Propiedades a T < 0ºC'.")
    calculated_properties = calcular_propiedades_alimento(composicion, T_prop, Tf_input)

elif calculation_type == "Propiedades a T < 0°C":
    T_prop = st.number_input("Temperatura de referencia para propiedades [ºC]", value=-10.0, step=1.0)
    if T_prop >= Tf_input:
        st.warning(f"La temperatura de referencia ({T_prop}ºC) es mayor o igual que la temperatura de congelación inicial ({Tf_input}ºC). Considera usar 'Propiedades a T > 0ºC'.")
    calculated_properties = calcular_propiedades_alimento(composicion, T_prop, Tf_input)

elif calculation_type in ["Temperatura final en el punto frío (ºC)", "Tiempo de proceso para alcanzar una temperatura final (ºC)", "Temperatura en una posición específica (X) en el alimento (ºC)"]:
    T_inicial_alimento = st.number_input("Temperatura Inicial del Alimento [ºC]", value=20.0, step=1.0)
    T_medio = st.number_input("Temperatura del Medio Calefactor/Enfriador [ºC]", value=80.0, step=1.0)
    h = st.number_input("Coeficiente de Convección (h) [W/(m²·K)]", value=100.0, step=5.0)

    geometria = st.selectbox(
        "Geometría del Alimento:",
        ("Placa Plana", "Cilindro", "Esfera")
    )
    if geometria == 'Placa Plana':
        st.info("Para placa plana, 'Dimensión Característica a' es el semi-espesor.")
    elif geometria == 'Cilindro':
        st.info("Para cilindro, 'Dimensión Característica a' es el radio.")
    elif geometria == 'Esfera':
        st.info("Para esfera, 'Dimensión Característica a' es el radio.")
    dimension_a = st.number_input("Dimensión Característica 'a' [m]", value=0.02, format="%.4f", help="Radio (cilindro, esfera) o semi-espesor (placa).")

    # Calculamos propiedades medias para Heisler. Estas propiedades son del alimento en su conjunto,
    # y deben evaluarse a una temperatura representativa del proceso.
    # Si la temperatura media del proceso está en la zona de congelación, el modelo de Choi y Okos
    # seguirá usando la fracción de hielo calculada.
    T_heisler_props_avg = (T_inicial_alimento + T_medio) / 2
    if T_heisler_props_avg < Tf_input:
        st.warning(f"La temperatura media para las propiedades ({T_heisler_props_avg:.1f}ºC) es menor que la de congelación ({Tf_input:.1f}ºC). Los modelos de Choi y Okos usados aquí asumen un comportamiento simple de congelación. Para procesos de congelación profundos, las propiedades pueden variar significativamente, afectando la precisión de Heisler en esa fase.")
    
    alpha_alimento_medio = calcular_alpha_alimento(T_heisler_props_avg, composicion, Tf_input)
    k_alimento_medio = calcular_k_alimento(T_heisler_props_avg, composicion, Tf_input)

    if calculation_type == "Temperatura final en el punto frío (ºC)":
        t_minutos = st.number_input("Tiempo de Proceso [min]", value=30.0, min_value=0.0, step=1.0)
        t_segundos = t_minutos * 60

    elif calculation_type == "Tiempo de proceso para alcanzar una temperatura final (ºC)":
        T_final_alimento = st.number_input("Temperatura Final deseada en el centro [ºC]", value=60.0, step=1.0)

    elif calculation_type == "Temperatura en una posición específica (X) en el alimento (ºC)":
        t_minutos = st.number_input("Tiempo de Proceso [min]", value=30.0, min_value=0.0, step=1.0)
        t_segundos = t_minutos * 60
        posicion_x = st.number_input("Posición 'x' desde el centro [m]", value=0.01, format="%.4f", help="Distancia desde el centro (0) hasta el borde (a). Debe ser <= 'a'.")

elif calculation_type == "Tiempo de congelación (min)":
    T_ambiente_congelacion = st.number_input("Temperatura del Medio de Congelación (Ta) [ºC]", value=-20.0, step=1.0)
    h_congelacion = st.number_input("Coeficiente de Convección (h) [W/(m²·K)]", value=20.0, step=1.0, help="Coeficiente de convección para el proceso de congelación.")

    # Para k_f de Plank, se suele evaluar a una temperatura media entre Tf y Ta
    # o a una temperatura representativa de la fase congelada (ej. -5°C o Tf - X°C).
    # Aquí la evaluamos a una temperatura un poco por debajo de Tf.
    T_kf_plank = min(-5.0, (Tf_input + T_ambiente_congelacion) / 2)
    # Ajuste para asegurar que la temperatura de evaluación de k_f no esté por encima de Tf
    if T_kf_plank > Tf_input:
         T_kf_plank = Tf_input - 2 # Asegurarse de que esté en la zona congelada

    k_alimento_congelado = calcular_k_alimento(T_kf_plank, composicion, Tf_input)

    # El calor latente efectivo (Le) para Plank considera el calor latente de congelación
    # y el calor sensible hasta la temperatura final de congelación. Aquí, una simplificación común
    # es usar solo el calor latente del agua inicial.
    L_e = (composicion['agua'] / 100) * 333.6e3 # J/kg (Calor latente de fusión del hielo a 0°C)
    st.info(f"Calor latente efectivo (Le) utilizado para Plank: {L_e/1000:.2f} kJ/kg (Basado solo en calor latente de fusión del agua inicial).")

    geometria_plank = st.selectbox(
        "Geometría del Alimento:",
        ("Placa Plana", "Cilindro", "Esfera")
    )
    if geometria_plank == 'Placa Plana':
        st.info("Para placa plana, 'Dimensión Característica a' es el semi-espesor.")
    elif geometria_plank == 'Cilindro':
        st.info("Para cilindro, 'Dimensión Característica a' es el radio.")
    elif geometria_plank == 'Esfera':
        st.info("Para esfera, 'Dimensión Característica a' es el radio.")
    dimension_a_plank = st.number_input("Dimensión Característica 'a' [m]", value=0.02, format="%.4f")

elif calculation_type == "Peso Molecular Aparente del Sólido (PMs) [g/mol]":
    st.info("Este cálculo estima el peso molecular promedio del sólido basándose en la temperatura inicial de congelación del alimento y su contenido de agua.")


# --- Botón de cálculo y resultados ---
st.markdown("---")
if st.button("Realizar Cálculo"):
    st.header("5. Resultados del Cálculo")
    if total_composicion != 100.0:
        st.error("Por favor, ajusta los porcentajes de composición para que sumen 100% antes de calcular.")
    else:
        if calculation_type == "Propiedades a T > 0°C" or calculation_type == "Propiedades a T < 0°C":
            densidad_val, cp_val, k_val, alpha_val = calculated_properties
            st.success(f"Propiedades Termofísicas del Alimento a {T_prop:.1f} °C:")
            st.write(f"**Densidad (ρ):** {densidad_val:.2f} kg/m³")
            st.write(f"**Calor Específico (Cp):** {cp_val:.2f} J/(kg·K)")
            st.write(f"**Conductividad Térmica (k):** {k_val:.4f} W/(m·K)")
            st.write(f"**Difusividad Térmica (α):** {alpha_val:.2e} m²/s")

        elif calculation_type == "Temperatura final en el punto frío (ºC)":
            result = calcular_temperatura_final_punto_frio(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a)
            if result:
                T_final_centro, Fo, Bi, A1, lambda1 = result
                st.success(f"Temperatura en el centro al finalizar el proceso: **{T_final_centro:.2f} °C**")
                st.markdown("---")
                st.subheader("Parámetros Adicionales del Proceso:")
                st.write(f"**Número de Biot (Bi):** {Bi:.2f}")
                st.write(f"**Número de Fourier (Fo):** {Fo:.3f}")
                st.write(f"**Coeficiente A1:** {A1:.4f}")
                st.write(f"**Valor propio Lambda1 (λ1):** {lambda1:.4f}")
                st.write(f"*(Propiedades evaluadas a la temperatura media del proceso: {T_heisler_props_avg:.1f} °C)*")

        elif calculation_type == "Tiempo de proceso para alcanzar una temperatura final (ºC)":
            result = calcular_tiempo_para_temperatura(T_final_alimento, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a)
            if result and result[0] is not None:
                t_minutos, Fo, Bi, A1, lambda1 = result
                st.success(f"Tiempo necesario para que el centro alcance {T_final_alimento:.1f} °C: **{t_minutos:.2f} minutos**")
                st.markdown("---")
                st.subheader("Parámetros Adicionales del Proceso:")
                st.write(f"**Número de Biot (Bi):** {Bi:.2f}")
                st.write(f"**Número de Fourier (Fo):** {Fo:.3f}")
                st.write(f"**Coeficiente A1:** {A1:.4f}")
                st.write(f"**Valor propio Lambda1 (λ1):** {lambda1:.4f}")
                st.write(f"*(Propiedades evaluadas a la temperatura media del proceso: {T_heisler_props_avg:.1f} °C)*")

        elif calculation_type == "Temperatura en una posición específica (X) en el alimento (ºC)":
            result = calcular_temperatura_posicion(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a, posicion_x)
            if result:
                T_final_x, Fo, Bi, A1, lambda1, position_factor = result
                st.success(f"Temperatura en la posición x={posicion_x:.4f} m al finalizar el proceso: **{T_final_x:.2f} °C**")
                st.markdown("---")
                st.subheader("Parámetros Adicionales del Proceso:")
                st.write(f"**Número de Biot (Bi):** {Bi:.2f}")
                st.write(f"**Número de Fourier (Fo):** {Fo:.3f}")
                st.write(f"**Coeficiente A1:** {A1:.4f}")
                st.write(f"**Valor propio Lambda1 (λ1):** {lambda1:.4f}")
                st.write(f"**Factor de Posición X(x/Lc, λ1):** {position_factor:.4f}")
                st.write(f"*(Propiedades evaluadas a la temperatura media del proceso: {T_heisler_props_avg:.1f} °C)*")

        elif calculation_type == "Tiempo de congelación (min)":
            result = calcular_tiempo_congelacion_plank(Tf_input, T_ambiente_congelacion, h_congelacion, k_alimento_congelado, L_e, geometria_plank, dimension_a_plank)
            if result:
                t_minutos_plank, P_plank, R_plank, Le_plank = result
                st.success(f"Tiempo de congelación estimado (Plank): **{t_minutos_plank:.2f} minutos**")
                st.markdown("---")
                st.subheader("Parámetros Adicionales del Proceso:")
                st.write(f"**Temperatura del medio (Ta):** {T_ambiente_congelacion:.1f} °C")
                st.write(f"**Coeficiente de convección (h):** {h_congelacion:.1f} W/(m²·K)")
                st.write(f"**Conductividad del alimento congelado (kf):** {k_alimento_congelado:.4f} W/(m·K) *(evaluada a {T_kf_plank:.1f}°C)*")
                st.write(f"**Calor latente efectivo (Le):** {Le_plank/1000:.2f} kJ/kg")
                st.write(f"**Factor Geométrico P:** {P_plank}")
                st.write(f"**Factor Geométrico R:** {R_plank}")

        elif calculation_type == "Peso Molecular Aparente del Sólido (PMs) [g/mol]":
            pm_s_result = calcular_pm_solido_aparente(Tf_input, composicion['agua'])
            if pm_s_result is not None:
                if pm_s_result == float('inf'):
                    st.success(f"Peso Molecular Aparente del Sólido (PMs): **Infinito** (Alimento es casi agua pura o Tf muy cercana a 0°C).")
                else:
                    st.success(f"Peso Molecular Aparente del Sólido (PMs): **{pm_s_result:.2f} g/mol**")
                st.info(f"*(Este valor es una estimación basada en la temperatura inicial de congelación del alimento ({Tf_input:.1f}°C) y la fracción de agua inicial ({composicion['agua']}%) a través de la ecuación de depresión crioscópica. Asume un comportamiento ideal de la solución y que los sólidos son el único soluto no congelable.)*")


# --- Sección de Información Adicional ---
st.markdown("---")
st.markdown("<h4 style='font-size: 1.4em;'>Información Adicional</h4>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Guía Rápida de Uso", "Referencias Bibliográficas", "Bases de Datos de Composición de Alimentos", "Ecuaciones Utilizadas"])

with tab1:
    st.markdown("<h5 style='font-size: 1.2em;'>Guía Rápida de Uso</h5>", unsafe_allow_html=True)
    st.markdown("""
    Para utilizar esta herramienta de simulación de procesos térmicos, sigue estos sencillos pasos:

    1.  **Define la Composición Proximal:**
        * En la sección "Introduce la composición del alimento" de la barra lateral izquierda, ingresa los porcentajes de **Agua, Proteína, Grasa, Carbohidratos, Fibra** y **Cenizas** de tu alimento.
        * Asegúrate de que la suma total sea **100%**. La aplicación te indicará si necesitas ajustar los valores.

    2.  **Define la Temperatura de Congelación (Tf):**
        * En la barra lateral izquierda, introduce la temperatura a la cual el alimento comienza a congelarse.

    3.  **Selecciona el Tipo de Cálculo:**
        * En la sección "Elige el cálculo que quieras realizar" en la parte central, usa las opciones de radio button para seleccionar la simulación que deseas.

    4.  **Ingresa los Parámetros Específicos:**
        * Debajo de la selección de cálculo, aparecerán los campos de entrada relevantes para tu simulación (temperaturas, coeficientes, geometría, etc.). Completa todos los datos necesarios.

    5.  **Realiza el Cálculo:**
        * Haz clic en el botón **"Realizar Cálculo"** en la parte inferior de la pantalla principal.
        * Los resultados se mostrarán en la sección central, junto con parámetros adicionales.
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h5 style='font-size: 1.2em;'>Referencias Bibliográficas</h5>", unsafe_allow_html=True)
    st.markdown("""
    * **Choi, Y., & Okos, M. R. (1986).** *Thermal Properties of Foods*. In M. R. Okos (Ed.), Physical Properties of Food Materials (pp. 93-112). Purdue University.
    * **Singh, R. P., & Heldman, D. D. (2009).** *Introduction to Food Engineering* (4th ed.). Academic Press.
    * **Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. (2007).** *Fundamentals of Heat and Mass Transfer* (6th ed.). John Wiley & Sons.
    * **Geankoplis, C. J. (2003).** *Transport Processes and Separation Process Principles* (4th ed.). Prentice Hall. (Para Ecuación de Plank)
    * **Fennema, O. R. (Ed.). (1996).** *Food Chemistry* (3rd ed.). Marcel Dekker. (Para Termodinámica de la Congelación)
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h5 style='font-size: 1.2em;'>Bases de Datos de Composición de Alimentos</h5>", unsafe_allow_html=True)
    st.markdown("""
    Aquí puedes encontrar enlaces a bases de datos confiables para consultar la composición proximal de diversos alimentos:

    * **USDA FoodData Central (Estados Unidos):**
        [https://fdc.nal.usda.gov/](https://fdc.nal.usda.gov/)
    * **BEDCA - Base de Datos Española de Composición de Alimentos (España):**
        [http://www.bedca.net/](http://www.bedca.net/)
    * **Tabla de Composición de Alimentos del INTA (Argentina):**
        [https://inta.gob.ar/documentos/tablas-de-composicion-de-alimentos](https://inta.gob.ar/documentos/tablas-de-composicion-de-alimentos)
    * **FAO/INFOODS (Internacional):**
        [https://www.fao.org/infoods/infoods/es/](https://www.fao.org/infoods/infoods/es/)
    * **Food Composition Databases (EUFIC - Europa):**
        [https://www.eufic.org/en/food-composition/article/food-composition-databases](https://www.eufic.org/en/food-composition/article/food-composition-databases)
    """, unsafe_allow_html=True)

with tab4:
    st.markdown("<h5 style='font-size: 1.2em;'>Ecuaciones Utilizadas</h5>", unsafe_allow_html=True)
    st.markdown("""
    Esta sección detalla las principales ecuaciones utilizadas en los cálculos de la aplicación.
    """)

    st.markdown("---")
    st.markdown("##### 1. Ecuaciones de Choi y Okos (1986) para Propiedades Termofísicas")
    st.markdown("""
    Las propiedades termofísicas del alimento ($\rho$, $C_p$, $k$) se estiman a partir de la suma ponderada de las propiedades de sus componentes (agua, proteína, grasa, carbohidratos, fibra, cenizas), evaluadas a la temperatura del proceso.

    *Para temperaturas **por encima de la temperatura de congelación inicial ($T_f$)** (fase no congelada), se utilizan las ecuaciones polinómicas de Choi y Okos para cada componente individual.*
    *Para temperaturas **por debajo de la temperatura de congelación inicial ($T_f$)** (fase congelada), se considera la formación de hielo. La fracción de agua no congelada ($X_u$) y la fracción de hielo ($X_i$) se calculan primero, y luego las propiedades se determinan ponderando las propiedades del hielo, del agua no congelada y de los sólidos. Las ecuaciones polinómicas para los componentes se ajustan automáticamente para considerar la fase (agua líquida vs. hielo).

    La difusividad térmica ($\alpha$) se calcula a partir de estas propiedades combinadas:
    """)
    st.latex(r"""
    \frac{1}{\rho_{\text{alimento}}} = \sum_{i=1}^{n} \frac{X_i}{\rho_i}
    """)
    st.latex(r"""
    C_{p, \text{alimento}} = \sum_{i=1}^{n} X_i \cdot C_{p,i}
    """)
    st.latex(r"""
    k_{\text{alimento}} = \sum_{i=1}^{n} X_i \cdot k_i
    """)
    st.latex(r"""
    \alpha_{\text{alimento}} = \frac{k_{\text{alimento}}}{\rho_{\text{alimento}} \cdot C_{p, \text{alimento}}}
    """)
    st.markdown(r"""
    Donde $X_i$ es la fracción de masa del componente $i$, y $\rho_i$, $C_{p,i}$, $k_i$ son la densidad, el calor específico y la conductividad térmica del componente $i$, respectivamente.
    """)

    st.markdown("---")
    st.markdown("##### 2. Fracción de Hielo y Fracción Molar de Agua No Congelada ($X_A$)")
    st.markdown("""
    Para temperaturas por debajo del punto de congelación inicial ($T_f$), la **fracción de agua no congelada ($X_u$)** se estima a partir de la **fracción molar de agua no congelada ($X_A$)**, que se calcula mediante la siguiente relación termodinámica (ecuación de depresión crioscópica, similar a Clausius-Clapeyron para soluciones ideales):
    """)
    st.latex(r"""
    \ln X_A = \frac{\lambda}{R} \left( \frac{1}{T_0} - \frac{1}{T} \right)
    """)
    st.markdown(r"""
    Donde:
    * $X_A$: Fracción molar de agua no congelada. Representa la actividad del agua ($a_w$) a la temperatura $T$.
    * $\lambda$: Calor latente **molar** de fusión del agua (aprox. 6010 J/mol).
    * $R$: Constante universal de los gases (8.314 J/(mol·K)).
    * $T_0$: Temperatura de fusión del hielo puro (273.15 K o 0°C).
    * $T$: Temperatura actual del alimento (en Kelvin).

    A partir de $X_A$, la **fracción de masa de agua no congelada ($m_u$)** se obtiene asumiendo que $X_A$ es la fracción de agua líquida sobre el total de agua inicial ($m_{u0}$).
    La fracción de hielo ($X_i$) se calcula como la fracción de agua inicial menos la fracción de agua no congelada ($X_i = m_{u0} - X_u$).
    """)

    st.markdown("---")
    st.markdown("##### 3. Peso Molecular Aparente del Sólido ($PM_s$)")
    st.markdown("""
    El peso molecular aparente del sólido ($PM_s$) puede ser estimado a partir de la fracción molar de agua no congelada ($X_A$) en el punto de congelación inicial ($T_f$) y la composición inicial del alimento. La relación utilizada es:
    """)
    st.latex(r"""
    X_A = \frac{m_u / PM_{\text{agua}}}{m_u / PM_{\text{agua}} + m_s / PM_s}
    """)
    st.markdown(r"""
    Donde:
    * $X_A$: Fracción molar de agua no congelada a la temperatura de congelación inicial ($T_f$). Se calcula a partir de la ecuación anterior.
    * $m_u$: Fracción de masa de agua **inicial** del alimento (agua no congelada a $T_f$).
    * $m_s$: Fracción de masa de sólidos totales del alimento ($1 - m_u$).
    * $PM_{\text{agua}}$: Peso molecular del agua (18.015 g/mol).
    * $PM_s$: Peso molecular aparente del sólido (g/mol).

    Despejando $PM_s$ de esta ecuación obtenemos:
    """)
    st.latex(r"""
    PM_s = \frac{X_A \cdot m_s \cdot PM_{\text{agua}}}{m_u (1 - X_A)}
    """)
    st.markdown("""
    Este cálculo proporciona una estimación del peso molecular promedio de los sólidos no acuosos presentes en el alimento, asumiendo un comportamiento ideal de la solución.
    """)


    st.markdown("---")
    st.markdown("##### 4. Ecuación de Plank (Tiempo de Congelación)")
    st.markdown("""
    El tiempo de congelación se calcula utilizando la **ecuación de Plank**, que es un modelo semi-empírico para el tiempo necesario para congelar un alimento de forma aproximada:
    """)
    st.latex(r"""
    t = \frac{L_e}{T_f - T_a} \left( \frac{P \cdot a}{h} + \frac{R \cdot a^2}{k_f} \right)
    """)
    st.markdown(r"""
    Donde:
    * $t$: Tiempo de congelación (s)
    * $L_e$: Calor latente efectivo (J/kg), considerando el calor latente de congelación del agua y el calor sensible involucrado.
    * $T_f$: Temperatura inicial de congelación del alimento (°C)
    * $T_a$: Temperatura del medio ambiente de congelación (°C)
    * $P, R$: Factores geométricos específicos para cada forma (ver tabla)
    * $a$: Dimensión característica (radio para cilindro/esfera, semiespesor para placa) (m)
    * $h$: Coeficiente de transferencia de calor por convección (W/(m²·K))
    * $k_f$: Conductividad térmica del alimento congelado (W/(m·K)), evaluada típicamente a la temperatura media del proceso de congelación.
    """)
    st.markdown("""
    | Geometría | P | R |
    | :-------- | :- | :- |
    | Placa Plana | 0.5 | 0.125 |
    | Cilindro | 0.25 | 0.0625 |
    | Esfera | 0.1667 | 0.0417 |
    """)

    st.markdown("---")
    st.markdown("##### 5. Ecuaciones de Heisler (Calentamiento/Enfriamiento Transitorio)")
    st.markdown("""
    Para el calentamiento o enfriamiento de un cuerpo, se utiliza el **primer término de la serie de Fourier**, que es una simplificación de las cartas o tablas de Heisler. Esta aproximación es válida cuando el **Número de Fourier ($Fo$) es mayor a 0.2**.
    """)
    st.latex(r"""
    Fo = \frac{\alpha \cdot t}{L_c^2}
    """)
    st.latex(r"""
    Bi = \frac{h \cdot L_c}{k}
    """)
    st.markdown(r"""
    Donde:
    * $Fo$: Número de Fourier
    * $Bi$: Número de Biot
    * $\alpha$: Difusividad térmica del alimento (m²/s)
    * $t$: Tiempo (s)
    * $L_c$: Longitud característica (m)
    * $h$: Coeficiente de transferencia de calor por convección (W/(m²·K))
    * $k$: Conductividad térmica del alimento (W/(m·K))
    """)

    st.markdown("""
    **a) Temperatura Final en el Punto Frío (Centro, $x=0$):**
    """)
    st.markdown("""
    Esta ecuación se usa para encontrar la temperatura en el centro del alimento a un tiempo dado.
    """)
    st.latex(r"""
    \frac{T_{\text{centro}}(t) - T_{\infty}}{T_i - T_{\infty}} = A_1 \cdot \exp(-\lambda_1^2 \cdot Fo)
    """)
    st.markdown(r"""
    Donde:
    * $T_{\text{centro}}(t)$: Temperatura en el centro al tiempo $t$ (°C)
    * $T_i$: Temperatura inicial uniforme del alimento (°C)
    * $T_{\infty}$: Temperatura del medio ambiente (°C)
    * $A_1, \lambda_1$: Coeficientes y valores propios del primer término, dependientes de la geometría y $Bi$. Se obtienen de tablas o soluciones numéricas.
    """)

    st.markdown("""
    **b) Tiempo de Proceso para Alcanzar una Temperatura Final:**
    """)
    st.markdown("""
    Para determinar el tiempo ($t$) necesario para que el **centro** del alimento alcance una temperatura específica ($T_{final}$), se despeja $t$ de la ecuación anterior. Esto aplica tanto para **calentamiento** como para **enfriamiento hasta una temperatura superior a la de congelación ($T > T_f$)**:
    """)
    st.latex(r"""
    t = -\frac{L_c^2}{\alpha \cdot \lambda_1^2} \cdot \ln \left( \frac{1}{A_1} \cdot \frac{T_{final} - T_{\infty}}{T_i - T_{\infty}} \right)
    """)

    st.markdown("""
    **c) Temperatura en una Posición Específica (X) en el Alimento:**
    """)
    st.markdown("""
    La temperatura en una posición $x$ (distancia desde el centro) se calcula multiplicando la relación de temperatura del centro por un factor de posición $X(x/L_c, \lambda_1)$:
    """)
    st.latex(r"""
    \frac{T(x,t) - T_{\infty}}{T_i - T_{\infty}} = \left( \frac{T_{\text{centro}}(t) - T_{\infty}}{T_i - T_{\infty}} \right) \cdot X(x/L_c, \lambda_1)
    """)
    st.markdown(r"""
    Donde $X(x/L_c, \lambda_1)$ es la función de posición del primer término, que depende de la geometría y de la relación $x/L_c$.
    """)
    st.markdown("""
    * **Placa Plana:**
        $X(x/L_c, \lambda_1) = \cos(\lambda_1 \cdot x/L_c)$
   * **Cilindro Infinito:**
        $X(x/L_c, \lambda_1) = J_0(\lambda_1 \cdot x/L_c)$ (donde $J_0$ es la función de Bessel de primera clase, orden cero)
    * **Esfera:**
    """)
    st.latex(r"""
    X(x/L_c, \lambda_1) = \frac{\sin(\lambda_1 \cdot x/L_c)}{\lambda_1 \cdot x/L_c}
    """)
    st.markdown("""
    *Nota: Para el cálculo en cilindros, se requiere la función de Bessel de primera clase de orden cero ($J_0$), que se obtiene de librerías matemáticas como `scipy.special`.*
    """)
