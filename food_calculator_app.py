import streamlit as st
import numpy as np
from scipy.special import jv # Para funciones de Bessel
import pandas as pd # Para la tabla de datos de Fo vs Bi

# --- Configuración de la página de Streamlit ---
st.set_page_config(
    page_title="Calculadora de Procesos Térmicos en Alimentos",
    page_icon="🍎",
    layout="wide"
)

# --- Funciones de Choi y Okos (Propiedades termofísicas) ---

# Propiedades del Agua (valores para agua líquida y hielo)
def densidad_agua(t):
    if t >= 0: # Agua líquida
        return 997.18 + 3.1439e-3 * t - 3.7574e-3 * t**2
    else: # Hielo
        return 916.89 - 0.13071 * t

def cp_agua(t):
    if t >= 0: # Agua líquida
        return 4176.2 - 9.0864e-2 * t + 5.4731e-3 * t**2
    else: # Hielo
        return 2062.3 + 6.0769 * t

def k_agua(t):
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
def calcular_fraccion_hielo(t, agua_porcentaje, Tf_input):
    L0 = 333.6e3 # Calor latente de fusión del hielo a 0°C en J/kg
    cp_agua_liquida_ref = 4186 # J/(kg·K) - Cp del agua líquida de referencia
    if t < Tf_input:
        # Fracción de hielo (Xi)
        Xi = (L0 / (cp_agua_liquida_ref * (Tf_input - t))) * (agua_porcentaje / 100)
        # Asegurarse de que Xi esté entre 0 y 1
        Xi = max(0, min(1, Xi))
        # Fracción de agua no congelada (Xu)
        Xu = (agua_porcentaje / 100) - Xi
    else:
        Xi = 0
        Xu = (agua_porcentaje / 100)
    return Xi, Xu

# Funciones principales para calcular propiedades del alimento
def calcular_densidad_alimento(t, composicion, Tf_input):
    agua_porcentaje = composicion['agua']
    Xi, Xu = calcular_fraccion_hielo(t, agua_porcentaje, Tf_input)

    rho_inv = (Xu / densidad_agua(t)) + \
              (Xi / densidad_agua(t - 0.0001)) + \
              (composicion['proteina'] / 100 / densidad_proteina(t)) + \
              (composicion['grasa'] / 100 / densidad_grasa(t)) + \
              (composicion['carbohidrato'] / 100 / densidad_carbohidrato(t)) + \
              (composicion['fibra'] / 100 / densidad_fibra(t)) + \
              (composicion['cenizas'] / 100 / densidad_cenizas(t))
    return 1 / rho_inv

def calcular_cp_alimento(t, composicion, Tf_input):
    agua_porcentaje = composicion['agua']
    Xi, Xu = calcular_fraccion_hielo(t, agua_porcentaje, Tf_input)

    cp_val = (Xu * cp_agua(t)) + \
             (Xi * cp_agua(t - 0.0001)) + \
             (composicion['proteina'] / 100 * cp_proteina(t)) + \
             (composicion['grasa'] / 100 * cp_grasa(t)) + \
             (composicion['carbohidrato'] / 100 * cp_carbohidrato(t)) + \
             (composicion['fibra'] / 100 * cp_fibra(t)) + \
             (composicion['cenizas'] / 100 * cp_cenizas(t))
    return cp_val

def calcular_k_alimento(t, composicion, Tf_input):
    agua_porcentaje = composicion['agua']
    Xi, Xu = calcular_fraccion_hielo(t, agua_porcentaje, Tf_input)

    k_val = (Xu * k_agua(t)) + \
            (Xi * k_agua(t - 0.0001)) + \
            (composicion['proteina'] / 100 * k_proteina(t)) + \
            (composicion['grasa'] / 100 * k_grasa(t)) + \
            (composicion['carbohidrato'] / 100 * k_carbohidrato(t)) + \
            (composicion['fibra'] / 100 * k_fibra(t)) + \
            (composicion['cenizas'] / 100 * k_cenizas(t))
    return k_val

def calcular_alpha_alimento(t, composicion, Tf_input):
    densidad = calcular_densidad_alimento(t, composicion, Tf_input)
    cp = calcular_cp_alimento(t, composicion, Tf_input)
    k = calcular_k_alimento(t, composicion, Tf_input)
    if densidad * cp == 0: # Evitar división por cero
        return 0
    return k / (densidad * cp)

# --- Funciones de Cálculo de Procesos ---

# Coeficientes A1 y lambda1 para Heisler (Primer término)
# Estos valores deben ser consistentes con tablas o soluciones de ecuaciones trascendentales.
# Se agrupan por geometría y rango de Bi. Se podrían hacer interpolaciones más precisas.
def get_heisler_coeffs(geometry, bi):
    if geometry == 'Placa Plana':
        # Valores aproximados para placa plana
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
        # Para Bi muy grandes (convección infinita), A1=1, lambda1=pi/2
        return 1.2732, 1.5708 # Límite para Bi -> inf (A1=4/pi, lambda1=pi/2)

    elif geometry == 'Cilindro':
        # Valores aproximados para cilindro
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
        # Para Bi muy grandes (convección infinita), A1=1.6018, lambda1=2.4048
        return 1.6018, 2.4048 # Límite para Bi -> inf

    elif geometry == 'Esfera':
        # Valores aproximados para esfera
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
        # Para Bi muy grandes (convección infinita), A1=1.5708, lambda1=pi
        return 1.5708, 3.1416 # Límite para Bi -> inf (A1=pi/2, lambda1=pi)
    return 1.0, 0.0 # Valores por defecto si la geometría no coincide

# Factor de posición X(x/Lc, lambda1) para Heisler
def get_heisler_position_factor(geometry, x_over_Lc, lambda1):
    if geometry == 'Placa Plana':
        return np.cos(lambda1 * x_over_Lc)
    elif geometry == 'Cilindro':
        if lambda1 * x_over_Lc == 0: # Caso especial para el centro del cilindro
            return 1.0
        return jv(0, lambda1 * x_over_Lc) # J0(lambda1 * x/Lc)
    elif geometry == 'Esfera':
        if lambda1 * x_over_Lc == 0: # Caso especial para el centro de la esfera
            return 1.0
        return np.sin(lambda1 * x_over_Lc) / (lambda1 * x_over_Lc)
    return 1.0 # Por defecto (centro)

# --- Funciones de Cálculo para la Interfaz ---

# Calculo de propiedades del alimento (para mostrar al usuario)
def calcular_propiedades_alimento(composicion, T_referencia, Tf_input):
    densidad = calcular_densidad_alimento(T_referencia, composicion, Tf_input)
    cp = calcular_cp_alimento(T_referencia, composicion, Tf_input)
    k = calcular_k_alimento(T_referencia, composicion, Tf_input)
    alpha = calcular_alpha_alimento(T_referencia, composicion, Tf_input)
    return densidad, cp, k, alpha

# Calculo de temperatura final en el punto frío (Heisler)
def calcular_temperatura_final_punto_frio(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a):
    if k_alimento_medio == 0 or h == 0:
        st.error("Error: La conductividad térmica o el coeficiente de convección no pueden ser cero para calcular el Bi. Por favor, revise las propiedades o los datos de entrada.")
        return None

    Lc = dimension_a # Para Heisler, Lc es el radio o el semi-espesor
    Bi = (h * Lc) / k_alimento_medio
    Fo = (alpha_alimento_medio * t_segundos) / (Lc**2)

    A1, lambda1 = get_heisler_coeffs(geometria, Bi)

    # Condición para la validez del primer término de Heisler
    if Fo < 0.2:
        st.warning(f"Advertencia: El número de Fourier (Fo = {Fo:.3f}) es menor que 0.2. La solución del primer término de la serie de Heisler puede no ser precisa. Considere tiempos de proceso más largos.")

    # Ecuación de Heisler para el centro (Theta_0)
    theta_0 = A1 * np.exp(-(lambda1**2) * Fo)

    T_final_centro = T_medio + theta_0 * (T_inicial_alimento - T_medio)
    return T_final_centro, Fo, Bi, A1, lambda1

# Cálculo del tiempo para alcanzar una temperatura final (Heisler)
def calcular_tiempo_para_temperatura(T_final_alimento, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a):
    if T_medio == T_inicial_alimento:
        st.error("Error: La temperatura del medio no puede ser igual a la temperatura inicial del alimento para este cálculo.")
        return None, None, None, None, None
    if T_medio == T_final_alimento:
        return 0, 0, 0, 0, 0 # Ya está a la temperatura objetivo

    Lc = dimension_a
    Bi = (h * Lc) / k_alimento_medio
    A1, lambda1 = get_heisler_coeffs(geometria, Bi)

    # Relación de temperatura no dimensional
    theta_0_target = (T_final_alimento - T_medio) / (T_inicial_alimento - T_medio)

    if theta_0_target <= 0 or theta_0_target >= A1: # Ajuste para logaritmo
         st.error(f"Error: La temperatura final objetivo ({T_final_alimento:.2f}°C) es inalcanzable o ya superada para las condiciones dadas.")
         st.info(f"La relación (Tf-Tinf)/(Ti-Tinf) debe ser menor a A1 ({A1:.4f}) y mayor a 0.")
         return None, None, None, None, None

    # Despejando Fo de la ecuación de Heisler
    try:
        Fo = -np.log(theta_0_target / A1) / (lambda1**2)
    except Exception as e:
        st.error(f"Error al calcular Fo: {e}. Puede que la temperatura objetivo sea inalcanzable con estos parámetros.")
        return None, None, None, None, None

    if Fo < 0.2:
        st.warning(f"Advertencia: El número de Fourier calculado (Fo = {Fo:.3f}) es menor que 0.2. La solución del primer término de la serie de Heisler puede no ser precisa.")
    elif Fo < 0:
        st.error("Error: El número de Fourier calculado es negativo, lo que indica un problema con las temperaturas de entrada (por ejemplo, el alimento ya está más frío/caliente que el objetivo).")
        return None, None, None, None, None

    t_segundos = (Fo * (Lc**2)) / alpha_alimento_medio
    t_minutos = t_segundos / 60
    return t_minutos, Fo, Bi, A1, lambda1

# Calculo de temperatura en posición específica (Heisler)
def calcular_temperatura_posicion(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a, posicion_x):
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

    # Calcular Theta_0 (temperatura no dimensional en el centro)
    theta_0 = A1 * np.exp(-(lambda1**2) * Fo)

    # Calcular Theta(x) (temperatura no dimensional en la posición x)
    x_over_Lc = posicion_x / Lc
    position_factor = get_heisler_position_factor(geometria, x_over_Lc, lambda1)
    theta_x = theta_0 * position_factor

    T_final_x = T_medio + theta_x * (T_inicial_alimento - T_medio)
    return T_final_x, Fo, Bi, A1, lambda1, position_factor

# Cálculo del tiempo de congelación (Plank)
def calcular_tiempo_congelacion_plank(Tf_input, T_ambiente_congelacion, h, k_congelado, L_efectivo, geometria, dimension_a):
    if Tf_input >= T_ambiente_congelacion:
        st.error("Error: La temperatura de congelación del alimento (Tf) debe ser mayor que la temperatura del medio de congelación (Ta) para que la congelación ocurra.")
        return None, None, None, None

    # Factores geométricos P y R para la ecuación de Plank
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

    # Asegurarse de que el denominador no sea cero
    if (Tf_input - T_ambiente_congelacion) == 0:
        st.error("Error: La diferencia de temperatura (Tf - Ta) no puede ser cero.")
        return None, None, None, None

    t_segundos = (L_efectivo / (Tf_input - T_ambiente_congelacion)) * \
                 ((P * dimension_a / h) + (R * dimension_a**2 / k_congelado))

    if t_segundos < 0:
        st.error("Error: El tiempo de congelación calculado es negativo. Revise las temperaturas o propiedades.")
        return None, None, None, None

    t_minutos = t_segundos / 60
    return t_minutos, P, R, L_efectivo

# --- Interfaz de Usuario Streamlit ---

st.title("Calculadora de Procesos Térmicos en Alimentos 🍎")

st.markdown("""
Esta aplicación permite calcular propiedades termofísicas de alimentos y simular procesos de calentamiento, enfriamiento y congelación utilizando modelos de la ingeniería de alimentos.
""")

st.sidebar.header("1. Composición del Alimento (%)")
st.sidebar.markdown("Introduce los porcentajes en peso de cada componente. La suma debe ser 100%.")

# Inputs de composición proximal
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

# Entrada para la temperatura de congelación inicial (Tf)
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
        "Tiempo de congelación (min)"
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

    # Evaluar propiedades a una temperatura media representativa para Heisler
    T_heisler_props_avg = (T_inicial_alimento + T_medio) / 2
    if T_heisler_props_avg < Tf_input:
        st.warning(f"La temperatura media para las propiedades ({T_heisler_props_avg:.1f}ºC) es menor que la de congelación ({Tf_input:.1f}ºC). Los modelos de Choi y Okos usados aquí asumen un comportamiento simple de congelación. Para procesos de congelación profundos, las propiedades pueden variar significativamente.")
    
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

    # Evaluar propiedades a una temperatura representativa para k_f y L_e en Plank
    # Temperatura para k_f: Típicamente a la mitad del rango de congelación, o a -5°C por ejemplo
    T_kf_plank = min(-5.0, (Tf_input + T_ambiente_congelacion) / 2) # Asegurarse de que esté en el rango de congelación
    if T_kf_plank > Tf_input: # Ajuste si la media es muy alta
         T_kf_plank = Tf_input - 2 # Un poco por debajo de Tf

    k_alimento_congelado = calcular_k_alimento(T_kf_plank, composicion, Tf_input)

    # Cálculo del calor latente efectivo (Le) para Plank
    # Se aproxima como el calor latente del agua inicial + calor sensible de agua y sólidos al Tf
    # L_e = X_agua * L_0 + C_p_no_agua * (Tf - T_final_deseada) + C_p_agua_congelada * (Tf - T_final_deseada)
    # Sin embargo, Plank se enfoca en el cambio de fase. Una simplificación común es:
    L_e = (composicion['agua'] / 100) * 333.6e3 # Solo el calor latente de congelación del agua
    st.info(f"Calor latente efectivo (Le) utilizado para Plank: {L_e/1000:.2f} kJ/kg (Basado solo en calor latente del agua).")


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

# --- Sección de Información Adicional ---
st.markdown("---") # Separador visual
st.markdown("<h4 style='font-size: 1.4em;'>Información Adicional</h4>", unsafe_allow_html=True)

# Usar st.tabs para organizar el contenido
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
    * **Singh, R. P., & Heldman, D. R. (2009).** *Introducción a la Ingeniería de los Alimentos* (2da ed.). Acribia.
    * **Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. (2007).** *Fundamentals of Heat and Mass Transfer* (6th ed.). John Wiley & Sons.
    * **Geankoplis, C. J. (2003).** *Transport Processes and Separation Process Principles* (4th ed.). Prentice Hall. (Para Ecuación de Plank)
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
    st.markdown("##### 2. Fracción de Hielo")
    st.markdown("""
    Para temperaturas por debajo del punto de congelación inicial ($T_f$), la **fracción de hielo** ($X_i$) se estima mediante la siguiente relación aproximada, asumiendo un equilibrio termodinámico:
    """)
    st.latex(r"""
    X_i = \frac{L_0}{C_{p,\text{agua}} \cdot (T_f - T)} \cdot X_{\text{agua, inicial}}
    """)
    st.markdown(r"""
    Donde $L_0$ es el calor latente de fusión del hielo a 0°C (333.6 kJ/kg), $C_{p,\text{agua}}$ es el calor específico del agua líquida (aprox. 4186 J/(kg·K)), $T_f$ es la temperatura inicial de congelación del alimento, $T$ es la temperatura actual y $X_{\text{agua, inicial}}$ es la fracción de agua inicial en el alimento.
    """)

    st.markdown("---")
    st.markdown("##### 3. Ecuación de Plank (Tiempo de Congelación)")
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
    st.markdown("##### 4. Ecuaciones de Heisler (Calentamiento/Enfriamiento Transitorio)")
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
    \frac{T_{centro}(t) - T_{\infty}}{T_i - T_{\infty}} = A_1 \cdot \exp(-\lambda_1^2 \cdot Fo)
    """)
    st.markdown(r"""
    Donde:
    * $T_{centro}(t)$: Temperatura en el centro al tiempo $t$ (°C)
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
    \frac{T(x,t) - T_{\infty}}{T_i - T_{\infty}} = \left( \frac{T_{centro}(t) - T_{\infty}}{T_i - T_{\infty}} \right) \cdot X(x/L_c, \lambda_1)
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
        $X(x/L_c, \lambda_1) = \frac{\sin(\lambda_1 \cdot x/L_c)}{\lambda_1 \cdot x/L_c}$
    """)
    st.markdown("""
    *Nota: Para el cálculo en cilindros, se requiere la función de Bessel de primera clase de orden cero ($J_0$), que se obtiene de librerías matemáticas como `scipy.special`.*
    """)
