import streamlit as st
import numpy as np
from scipy.special import jv as J0 # Para funciones de Bessel
# Eliminado: import plotly.graph_objects as go
# Eliminado: import plotly.express as px

# --- Funciones de Cálculo ---
def calcular_propiedades_alimento(composicion, T, Tf):
    """
    Calcula las propiedades termofísicas del alimento (densidad, Cp, k, alpha)
    usando las correlaciones de Choi y Okos (1986) de forma simplificada.
    Se asume que 'composicion' es un diccionario con 'agua', 'proteina', 'grasa',
    'carbohidratos', 'fibra', 'cenizas' en porcentaje.
    T: Temperatura de evaluación en °C
    Tf: Temperatura de congelación inicial en °C
    """
    T_K = T + 273.15
    Tf_K = Tf + 273.15

    # Propiedades del agua (base) - Correlaciones simplificadas
    rho_w = 997.18 - 0.0031439 * T - 0.0037574 * T**2 # kg/m3
    cp_w = 4180 - 0.5 * T # J/(kg.K)
    k_w = 0.56 + 0.0018 * T # W/(m.K)

    # Propiedades de los sólidos (valores constantes simplificados)
    rho_prot = 1300
    cp_prot = 1550
    k_prot = 0.25

    rho_fat = 920
    cp_fat = 1900
    k_fat = 0.18

    rho_carb = 1600
    cp_carb = 1550
    k_carb = 0.20

    rho_fiber = 1500
    cp_fiber = 1350
    k_fiber = 0.15

    rho_ash = 2000
    cp_ash = 820
    k_ash = 0.35

    if T >= Tf: # Fase no congelada (líquida)
        # Suma ponderada de las propiedades de los componentes
        densidad = (composicion['agua']/100 * rho_w +
                    composicion['proteina']/100 * rho_prot +
                    composicion['grasa']/100 * rho_fat +
                    composicion['carbohidratos']/100 * rho_carb +
                    composicion['fibra']/100 * rho_fiber +
                    composicion['cenizas']/100 * rho_ash)

        cp = (composicion['agua']/100 * cp_w +
              composicion['proteina']/100 * cp_prot +
              composicion['grasa']/100 * cp_fat +
              composicion['carbohidratos']/100 * cp_carb +
              composicion['fibra']/100 * cp_fiber +
              composicion['cenizas']/100 * cp_ash)

        k = (composicion['agua']/100 * k_w +
             composicion['proteina']/100 * k_prot +
             composicion['grasa']/100 * k_fat +
             composicion['carbohidratos']/100 * k_carb +
             composicion['fibra']/100 * k_fiber +
             composicion['cenizas']/100 * k_ash)

    else: # Fase congelada
        # Calor latente molar de fusión del agua (J/mol)
        lambda_val = 6010
        # Constante universal de los gases (J/(mol·K))
        R_gas = 8.314
        # Temperatura de fusión del hielo puro (K)
        T0_K = 273.15

        # Fracción molar de agua no congelada (actividad del agua)
        # Evitar log(0) o valores negativos extremos si T es muy bajo
        if T_K <= 0: # Cuando T es muy baja, XA tiende a 0
            XA = 0.0001
        else:
            try:
                XA = np.exp((lambda_val / R_gas) * (1 / T0_K - 1 / T_K))
            except OverflowError:
                XA = 0.0001
            if XA > 1: XA = 1 # Asegurar que no exceda 1

        # Fracción de masa de agua no congelada
        mu_agua_inicial = composicion['agua'] / 100.0
        fraccion_agua_no_congelada = mu_agua_inicial * XA
        fraccion_hielo = mu_agua_inicial - fraccion_agua_no_congelada

        # Propiedades del hielo (simplificadas y constantes para este modelo)
        rho_ice = 916.8 # kg/m3 a 0°C
        cp_ice = 2064 # J/(kg.K) a -10°C (promedio)
        k_ice = 2.22 # W/(m.K) a -10°C (promedio)

        # Propiedades de los sólidos
        fraccion_solidos = 1.0 - mu_agua_inicial

        # Densidad de la mezcla (modelo de mezcla ideal para volumen)
        sum_inv_rho_frac = (fraccion_hielo / rho_ice +
                            fraccion_agua_no_congelada / rho_w +
                            (composicion['proteina']/100 / rho_prot) +
                            (composicion['grasa']/100 / rho_fat) +
                            (composicion['carbohidratos']/100 / rho_carb) +
                            (composicion['fibra']/100 / rho_fiber) +
                            (composicion['cenizas']/100 / rho_ash))
        densidad = 1 / sum_inv_rho_frac if sum_inv_rho_frac > 0 else 0

        # Cp de la mezcla (suma ponderada por masa)
        cp = (fraccion_hielo * cp_ice +
              fraccion_agua_no_congelada * cp_w +
              composicion['proteina']/100 * cp_prot +
              composicion['grasa']/100 * cp_fat +
              composicion['carbohidratos']/100 * cp_carb +
              composicion['fibra']/100 * cp_fiber +
              composicion['cenizas']/100 * cp_ash)

        # k de la mezcla (simplificado como suma ponderada)
        k = (fraccion_hielo * k_ice +
             fraccion_agua_no_congelada * k_w +
             composicion['proteina']/100 * k_prot +
             composicion['grasa']/100 * k_fat +
             composicion['carbohidratos']/100 * k_carb +
             composicion['fibra']/100 * k_fiber +
             composicion['cenizas']/100 * k_ash)

    alpha_val = k / (densidad * cp) if (densidad * cp) > 0 else 0
    return densidad, cp, k, alpha_val

def calcular_lambda1_A1(Bi, geometria):
    """
    Calcula los coeficientes lambda1 y A1 para las ecuaciones de Heisler
    basados en el número de Biot y la geometría.
    Se usan valores interpolados de tablas estándar.
    """
    if Bi < 0.001: # Caso de capacitancia concentrada
        lambda1 = 0.001
        A1 = 1.0
    elif Bi > 1000: # Para Bi muy grandes, aprox a solución de temperatura constante en superficie
        if geometria == "Placa Plana":
            lambda1 = np.pi / 2
            A1 = 4 / np.pi
        elif geometria == "Cilindro":
            lambda1 = 2.4048 # Primera raíz de J0(x) = 0
            A1 = 2 / lambda1
        elif geometria == "Esfera":
            lambda1 = np.pi
            A1 = 2
    else:
        # Valores de tablas de Heisler (simplificados para rangos comunes)
        # Interpolación lineal para obtener valores más precisos
        if geometria == "Placa Plana":
            bi_vals = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 500.0, 1000.0])
            lambda1_vals = np.array([0.0998, 0.1410, 0.1987, 0.2425, 0.2791, 0.3111, 0.4328, 0.6115, 0.7496, 0.8603, 0.9408, 1.1593, 1.2746, 1.3525, 1.4078, 1.4961, 1.5202, 1.5585, 1.5694, 1.5707, 1.5707, 1.5708, 1.5708, 1.5708]) # pi/2 = 1.5708
            A1_vals = np.array([1.0000, 1.0001, 1.0002, 1.0003, 1.0005, 1.0006, 1.0016, 1.0063, 1.0140, 1.0247, 1.0385, 1.1145, 1.1895, 1.2621, 1.3315, 1.4800, 1.5471, 1.7240, 1.8080, 1.8540, 1.8840, 1.9780, 1.9999, 2.0000]) # 4/pi = 1.2732, pero las tablas dan hasta casi 2
            lambda1 = np.interp(Bi, bi_vals, lambda1_vals)
            A1 = np.interp(Bi, bi_vals, A1_vals)
        elif geometria == "Cilindro":
            bi_vals = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 500.0, 1000.0])
            lambda1_vals = np.array([0.1412, 0.1995, 0.2814, 0.3430, 0.3951, 0.4388, 0.6170, 0.8516, 0.9926, 1.1081, 1.2558, 1.4793, 1.6373, 1.7640, 1.8710, 2.0720, 2.1795, 2.3486, 2.3809, 2.3924, 2.3972, 2.4029, 2.4048, 2.4048]) # 2.4048
            A1_vals = np.array([1.0000, 1.0001, 1.0003, 1.0005, 1.0008, 1.0011, 1.0040, 1.0159, 1.0311, 1.0494, 1.0701, 1.1643, 1.2488, 1.3259, 1.3965, 1.5791, 1.6934, 1.9472, 2.0768, 2.1461, 2.1899, 2.3168, 2.4020, 2.4020]) # 2/lambda1 -> 2/2.4048 = 0.8316
            lambda1 = np.interp(Bi, bi_vals, lambda1_vals)
            A1 = np.interp(Bi, bi_vals, A1_vals)
        elif geometria == "Esfera":
            bi_vals = np.array([0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0, 500.0, 1000.0])
            lambda1_vals = np.array([0.1730, 0.2445, 0.3450, 0.4206, 0.4841, 0.5375, 0.7593, 1.0767, 1.3037, 1.4962, 1.5708, 2.0288, 2.2789, 2.4566, 2.5704, 2.7661, 2.8363, 2.9730, 3.0200, 3.0450, 3.0590, 3.0900, 3.1416, 3.1416]) # pi = 3.1416
            A1_vals = np.array([1.0000, 1.0001, 1.0003, 1.0006, 1.0010, 1.0015, 1.0059, 1.0232, 1.0505, 1.0852, 1.1275, 1.3090, 1.4793, 1.6373, 1.7820, 2.1130, 2.2980, 2.7560, 2.9340, 3.0260, 3.0800, 3.1400, 3.1416, 3.1416]) # pi = 3.1416
            lambda1 = np.interp(Bi, bi_vals, lambda1_vals)
            A1 = np.interp(Bi, bi_vals, A1_vals)

    return lambda1, A1

def calcular_temperatura_final_punto_frio(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a):
    """
    Calcula la temperatura en el centro de un alimento usando el primer término de la serie de Heisler.
    """
    if dimension_a == 0:
        st.error("La dimensión característica 'a' no puede ser cero.")
        return None

    if geometria == 'Placa Plana':
        Lc = dimension_a
    elif geometria == 'Cilindro' or geometria == 'Esfera':
        Lc = dimension_a # Para cilindro y esfera, Lc = radio 'a'

    # Evitar divisiones por cero o valores no válidos de k_alimento_medio
    if k_alimento_medio == 0:
        st.error("La conductividad térmica del alimento (k) es cero. Revisa la composición o el rango de temperatura.")
        return None

    Bi = (h * Lc) / k_alimento_medio
    Fo = (alpha_alimento_medio * t_segundos) / (Lc**2)

    if Fo < 0.2:
        st.warning("El Número de Fourier (Fo) es menor a 0.2. La aproximación del primer término de la serie de Heisler podría no ser precisa. Considera aumentar el tiempo de proceso para una mayor precisión.")

    lambda1, A1 = calcular_lambda1_A1(Bi, geometria)

    if (T_inicial_alimento - T_medio) == 0:
        st.warning("La temperatura inicial del alimento es igual a la temperatura del medio. No habrá transferencia de calor.")
        return T_inicial_alimento, Fo, Bi, A1, lambda1

    # Cálculo de la relación de temperatura θ/θi
    try:
        theta_theta_i = A1 * np.exp(-lambda1**2 * Fo)
        # Asegurarse de que theta_theta_i esté en un rango válido [0, 1]
        theta_theta_i = np.clip(theta_theta_i, 0, 1)

    except OverflowError:
        st.error("Error de cálculo (OverflowError). Revisa tus parámetros de entrada. El tiempo o las propiedades podrían llevar a valores extremos.")
        return None
    except RuntimeWarning: # Para np.exp de valores muy negativos
        theta_theta_i = 0.0 # Indica que la temperatura ya alcanzó el medio

    T_final_centro = T_medio + theta_theta_i * (T_inicial_alimento - T_medio)

    return T_final_centro, Fo, Bi, A1, lambda1

def calcular_tiempo_para_temperatura(T_final_alimento, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a):
    """
    Calcula el tiempo necesario para que el centro de un alimento alcance una temperatura específica
    usando el primer término de la serie de Heisler.
    """
    if dimension_a == 0:
        st.error("La dimensión característica 'a' no puede ser cero.")
        return None

    if (T_inicial_alimento - T_medio) == 0:
        st.warning("La temperatura inicial del alimento es igual a la temperatura del medio. No habrá transferencia de calor.")
        return 0, 0, 0, 0, 0

    # Determinar si es calentamiento o enfriamiento
    is_heating = (T_medio > T_inicial_alimento)
    is_cooling = (T_medio < T_inicial_alimento)

    if is_heating and T_final_alimento < T_inicial_alimento:
        st.error("Para calentamiento, la temperatura final deseada debe ser mayor que la inicial.")
        return None, None, None, None, None
    if is_heating and T_final_alimento > T_medio:
        st.error("Para calentamiento, la temperatura final deseada no puede ser mayor que la temperatura del medio.")
        return None, None, None, None, None
    if is_cooling and T_final_alimento > T_inicial_alimento:
        st.error("Para enfriamiento, la temperatura final deseada debe ser menor que la inicial.")
        return None, None, None, None, None
    if is_cooling and T_final_alimento < T_medio:
        st.error("Para enfriamiento, la temperatura final deseada no puede ser menor que la temperatura del medio.")
        return None, None, None, None, None


    if geometria == 'Placa Plana':
        Lc = dimension_a
    elif geometria == 'Cilindro' or geometria == 'Esfera':
        Lc = dimension_a # Para cilindro y esfera, Lc = radio 'a'

    if k_alimento_medio == 0:
        st.error("La conductividad térmica del alimento (k) es cero. Revisa la composición o el rango de temperatura.")
        return None, None, None, None, None

    Bi = (h * Lc) / k_alimento_medio
    lambda1, A1 = calcular_lambda1_A1(Bi, geometria)

    # Calcular la relación de temperatura (θ/θi)
    denom = (T_inicial_alimento - T_medio)
    if denom == 0:
        st.error("Error: Diferencia de temperatura inicial y media es cero. Revisa los parámetros.")
        return None, None, None, None, None
    theta_theta_i_target = (T_final_alimento - T_medio) / denom

    # Asegurarse de que el argumento del logaritmo sea positivo y válido
    log_arg = theta_theta_i_target / A1
    if log_arg <= 0:
        st.error("La temperatura final deseada no es alcanzable con los parámetros dados. Asegúrate de que T_final esté entre T_medio y T_inicial.")
        return None, None, None, None, None

    try:
        # Fo = -1/lambda1^2 * ln(theta/theta_i / A1)
        Fo = - (1 / (lambda1**2)) * np.log(log_arg)
        if Fo < 0:
            st.error("El tiempo calculado resulta negativo. Revisa los parámetros de temperatura y la geometría.")
            return None, None, None, None, None
    except Exception as e:
        st.error(f"Error al calcular el número de Fourier: {e}. Revisa tus parámetros de entrada.")
        return None, None, None, None, None

    if Fo < 0.2:
        st.warning(f"El Número de Fourier (Fo={Fo:.3f}) es menor a 0.2. La aproximación del primer término de la serie de Heisler podría no ser precisa. El tiempo calculado podría ser una subestimación.")

    if alpha_alimento_medio == 0:
        st.error("La difusividad térmica del alimento (alpha) es cero. Revisa la composición o el rango de temperatura.")
        return None, None, None, None, None

    t_segundos = (Fo * Lc**2) / alpha_alimento_medio
    t_minutos = t_segundos / 60

    return t_minutos, Fo, Bi, A1, lambda1

def calcular_temperatura_posicion(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio, k_alimento_medio, h, geometria, dimension_a, posicion_x):
    """
    Calcula la temperatura en una posición 'x' de un alimento usando el primer término de Heisler.
    """
    if dimension_a == 0:
        st.error("La dimensión característica 'a' no puede ser cero.")
        return None
    if posicion_x > dimension_a:
        st.error(f"La posición 'x' ({posicion_x:.4f} m) no puede ser mayor que la dimensión característica 'a' ({dimension_a:.4f} m).")
        return None
    if posicion_x < 0:
        st.error("La posición 'x' no puede ser negativa.")
        return None

    if geometria == 'Placa Plana':
        Lc = dimension_a
    elif geometria == 'Cilindro' or geometria == 'Esfera':
        Lc = dimension_a # Para cilindro y esfera, Lc = radio 'a'

    if k_alimento_medio == 0:
        st.error("La conductividad térmica del alimento (k) es cero. Revisa la composición o el rango de temperatura.")
        return None

    Bi = (h * Lc) / k_alimento_medio
    Fo = (alpha_alimento_medio * t_segundos) / (Lc**2)

    if Fo < 0.2:
        st.warning("El Número de Fourier (Fo) es menor a 0.2. La aproximación del primer término de la serie de Heisler podría no ser precisa.")

    lambda1, A1 = calcular_lambda1_A1(Bi, geometria)

    if (T_inicial_alimento - T_medio) == 0:
        st.warning("La temperatura inicial del alimento es igual a la temperatura del medio. No habrá transferencia de calor.")
        return T_inicial_alimento, Fo, Bi, A1, lambda1, 1.0

    # Cálculo de la relación de temperatura en el centro
    theta_theta_i_center = A1 * np.exp(-lambda1**2 * Fo)

    # Cálculo del factor de posición X(x/Lc, lambda1)
    if Lc == 0:
        st.error("La longitud característica (Lc) es cero, no se puede calcular el factor de posición.")
        return None

    x_over_Lc = posicion_x / Lc

    if geometria == 'Placa Plana':
        position_factor = np.cos(lambda1 * x_over_Lc)
    elif geometria == 'Cilindro':
        position_factor = J0(lambda1 * x_over_Lc) # scipy.special.jv(0, x)
    elif geometria == 'Esfera':
        # Evitar división por cero si x_over_Lc es extremadamente pequeño y lambda1 es cero
        if np.isclose(lambda1 * x_over_Lc, 0):
             position_factor = 1.0 # Limite cuando lambda1 * x/Lc -> 0
        else:
            position_factor = np.sin(lambda1 * x_over_Lc) / (lambda1 * x_over_Lc)
    else:
        position_factor = 1.0 # Valor por defecto si la geometría no es reconocida

    # Asegurarse de que position_factor esté en un rango válido [0, 1]
    position_factor = np.clip(position_factor, 0, 1)

    # Cálculo de la temperatura en la posición x
    T_final_x = T_medio + theta_theta_i_center * position_factor * (T_inicial_alimento - T_medio)

    return T_final_x, Fo, Bi, A1, lambda1, position_factor

def calcular_tiempo_congelacion_plank(Tf_input, T_ambiente_congelacion, h_congelacion, k_alimento_congelado, L_e, geometria_plank, dimension_a_plank):
    """
    Calcula el tiempo de congelación usando la ecuación de Plank.
    """
    if dimension_a_plank == 0:
        st.error("La dimensión característica 'a' no puede ser cero.")
        return None
    if Tf_input <= T_ambiente_congelacion:
        st.error("La temperatura de congelación inicial debe ser mayor que la temperatura del medio ambiente de congelación para que ocurra la congelación.")
        return None

    # Factores geométricos para Plank
    P_plank = 0
    R_plank = 0
    if geometria_plank == 'Placa Plana':
        P_plank = 0.5
        R_plank = 0.125
    elif geometria_plank == 'Cilindro':
        P_plank = 0.25
        R_plank = 0.0625
    elif geometria_plank == 'Esfera':
        P_plank = 0.1667 # 1/6
        R_plank = 0.0417 # 1/24

    # Diferencia de temperatura
    delta_T = Tf_input - T_ambiente_congelacion
    if delta_T <= 0:
        st.error("La temperatura de congelación debe ser mayor que la temperatura ambiente para calcular el tiempo de congelación.")
        return None

    try:
        # t = (Le / delta_T) * (P*a/h + R*a^2/kf)
        term1 = L_e / delta_T
        if h_congelacion == 0:
            st.error("El coeficiente de convección (h) no puede ser cero para el cálculo de Plank.")
            return None, None, None, None
        if k_alimento_congelado == 0:
            st.error("La conductividad térmica del alimento congelado (kf) no puede ser cero para el cálculo de Plank.")
            return None, None, None, None


        term2 = (P_plank * dimension_a_plank) / h_congelacion
        term3 = (R_plank * dimension_a_plank**2) / k_alimento_congelado

        t_segundos_plank = term1 * (term2 + term3)
        t_minutos_plank = t_segundos_plank / 60
    except ZeroDivisionError:
        st.error("División por cero en el cálculo de Plank. Revisa los valores de h, k_f o la diferencia de temperatura.")
        return None
    except Exception as e:
        st.error(f"Error en el cálculo del tiempo de congelación de Plank: {e}")
        return None

    return t_minutos_plank, P_plank, R_plank, L_e

def calcular_pm_solido_aparente(Tf_input, porcentaje_agua):
    """
    Calcula el peso molecular aparente del sólido basándose en Tf y el contenido de agua.
    Asume depresión crioscópica ideal.
    """
    Tf_K = Tf_input + 273.15
    T0_K = 273.15 # Temperatura de fusión del hielo puro en K
    lambda_val = 6010 # Calor latente molar de fusión del agua (J/mol)
    R_gas = 8.314 # Constante universal de los gases (J/(mol·K))
    PM_agua = 18.015 # g/mol

    if Tf_input >= 0:
        st.warning("La temperatura de congelación inicial debe ser menor a 0°C para calcular un PMs significativo. Para Tf >= 0°C, PMs tiende a infinito (agua pura).")
        return float('inf') # Indica que es agua pura o casi

    try:
        # ln XA = (lambda/R) * (1/T0 - 1/Tf)
        # Asegurarse de que Tf_K no sea cero o negativo si viene de un input.
        if Tf_K <= 0:
            st.error("Temperatura de congelación en Kelvin no válida para cálculo de PMs.")
            return None

        ln_XA = (lambda_val / R_gas) * (1 / T0_K - 1 / Tf_K)
        XA = np.exp(ln_XA)

        if XA >= 1: # Esto indica que el punto de congelación es 0°C o por encima (problema)
             st.warning("La fracción molar de agua calculada (XA) es >= 1, lo que sugiere que Tf es cercana o mayor a 0°C. El PMs será muy grande o infinito.")
             return float('inf')

        m_u = porcentaje_agua / 100.0 # Fracción de masa de agua
        m_s = 1.0 - m_u # Fracción de masa de sólidos

        if m_u <= 0 or m_s <= 0:
            st.error("La composición de agua o sólidos no es válida para calcular el PMs.")
            return None

        # PM_s = (XA * m_s * PM_agua) / (m_u * (1 - XA))
        # Asegurarse de que (1 - XA) no sea cero o negativo
        denominator = (m_u * (1 - XA))
        if denominator <= 0:
            st.error("Denominador cero o negativo en el cálculo de PMs. La fracción de agua o XA no es válida.")
            return None

        pm_s = (XA * m_s * PM_agua) / denominator

        if pm_s < 0:
            st.warning("El Peso Molecular Aparente del Sólido calculado es negativo, lo que indica un problema con los parámetros de entrada o la aplicabilidad del modelo.")
            return None
        return pm_s
    except ZeroDivisionError:
        st.error("División por cero en el cálculo del PMs. Revisa la temperatura de congelación inicial y el contenido de agua.")
        return None
    except Exception as e:
        st.error(f"Error en el cálculo del Peso Molecular Aparente del Sólido: {e}")
        return None

# Eliminada la función generar_graficos_propiedades

# --- Configuración de la página Streamlit ---
st.set_page_config(layout="wide", page_title="Calculadora de Propiedades y Procesos Térmicos de Alimentos")

# Título Principal con tamaño ajustado
st.markdown("<h1 style='font-size: 1.8em;'>🍎 Calculadora de Propiedades y Procesos Térmicos de Alimentos ❄️🔥</h1>", unsafe_allow_html=True)
st.markdown("¡Bienvenido a la calculadora interactiva para simular el comportamiento térmico de los alimentos!")
st.markdown("SMMiroE")
# --- Entrada de Composición del Alimento ---
st.markdown("---")
# Subtítulo 1 con tamaño ajustado
st.markdown("<h2 style='font-size: 1.4em;'>1. Composición Proximal del Alimento (%)</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    agua = st.number_input("Agua [%]", value=75.0, min_value=0.0, max_value=100.0, step=0.1, key="agua_input")
    proteina = st.number_input("Proteína [%]", value=10.0, min_value=0.0, max_value=100.0, step=0.1, key="proteina_input")
    grasa = st.number_input("Grasa [%]", value=5.0, min_value=0.0, max_value=100.0, step=0.1, key="grasa_input")
with col2:
    carbohidratos = st.number_input("Carbohidratos [%]", value=8.0, min_value=0.0, max_value=100.0, step=0.1, key="carbohidratos_input")
    fibra = st.number_input("Fibra [%]", value=1.0, min_value=0.0, max_value=100.0, step=0.1, key="fibra_input")
    cenizas = st.number_input("Cenizas [%]", value=1.0, min_value=0.0, max_value=100.0, step=0.1, key="cenizas_input")

composicion = {
    'agua': agua,
    'proteina': proteina,
    'grasa': grasa,
    'carbohidratos': carbohidratos,
    'fibra': fibra,
    'cenizas': cenizas
}

total_composicion = sum(composicion.values())
if abs(total_composicion - 100.0) > 0.01: # Usar una pequeña tolerancia para la comparación de flotantes
    st.warning(f"La suma total de la composición es {total_composicion:.1f}%. Por favor, ajusta los porcentajes para que sumen 100%.")
else:
    st.success("¡Composición ajustada al 100%!")

# --- Temperatura de Congelación Inicial ---
st.markdown("---")
# Subtítulo 2 con tamaño ajustado
st.markdown("<h2 style='font-size: 1.4em;'>2. Temperatura de Congelación Inicial (Tf)</h2>", unsafe_allow_html=True)
Tf_input = st.number_input("Temperatura de Congelación Inicial (Tf) [ºC]", value=-1.0, step=0.1, key="tf_input")
st.info(f"*Esta es la temperatura a la cual el alimento comienza a congelarse, puedes encontrarla en la bibliografia.*")


# --- Selección del Tipo de Cálculo ---
st.markdown("---")
# Subtítulo 3 con tamaño ajustado
st.markdown("<h2 style='font-size: 1.4em;'>3. Elige el Cálculo a Realizar</h2>", unsafe_allow_html=True)

calculation_type = st.radio(
    "", # Aquí se eliminó el texto del título del radio button
    (
     "Propiedades a T > 0°C",
     "Propiedades a T < 0°C",
     "Temperatura final en el punto frío (ºC)",
     "Tiempo de proceso para alcanzar una temperatura final (ºC)",
     "Temperatura en una posición específica (X) en el alimento (ºC)",
     "Tiempo de congelación (min)",
     "Peso Molecular Aparente del Sólido (PMs) [g/mol]"),
    key="calculation_type_radio"
)

# --- Inputs dinámicos según la selección ---
st.markdown("---")
# Subtítulo 4 con tamaño ajustado
st.markdown("<h2 style='font-size: 1.4em;'>4. Parámetros del Cálculo</h2>", unsafe_allow_html=True)

# Inicializar variables que podrían no ser usadas en todos los caminos
T_prop = 0.0
T_inicial_alimento = 0.0
T_medio = 0.0
h = 0.0
geometria = "Placa Plana"
dimension_a = 0.0
t_segundos = 0.0
posicion_x = 0.0
T_final_alimento = 0.0
T_ambiente_congelacion = 0.0
h_congelacion = 0.0
L_e = 0.0
geometria_plank = "Placa Plana"
dimension_a_plank = 0.0


if calculation_type == "Propiedades a T > 0°C":
    T_prop = st.number_input("Temperatura para el calculo de propiedades [ºC]", value=20.0, step=1.0, key="t_prop_gt0")
    if T_prop < Tf_input:
        st.warning(f"La temperatura de referencia ({T_prop}ºC) está en la zona de congelación inicial ({Tf_input}ºC). Las propiedades se calcularán para la fase congelada. Considera cambiar a 'Propiedades a T < 0ºC' si ese es tu objetivo principal.")
    # No se calcula aquí, sino en el botón "Realizar Cálculo"

elif calculation_type == "Propiedades a T < 0°C":
    T_prop = st.number_input("Temperatura para el calculo de propiedades [ºC]", value=-10.0, step=1.0, key="t_prop_lt0")
    if T_prop >= Tf_input:
        st.warning(f"La temperatura ({T_prop}ºC) es mayor o igual que la temperatura de congelación inicial ({Tf_input}ºC). Las propiedades se calcularán como si no hubiera hielo. Considera cambiar a 'Propiedades a T > 0ºC' si ese es tu objetivo principal.")
    # No se calcula aquí, sino en el botón "Realizar Cálculo"

elif calculation_type in ["Temperatura final en el punto frío (ºC)", "Tiempo de proceso para alcanzar una temperatura final (ºC)", "Temperatura en una posición específica (X) en el alimento (ºC)"]:
    T_inicial_alimento = st.number_input("Temperatura Inicial del Alimento [ºC]", value=20.0, step=1.0, key="t_inicial_alimento")
    T_medio = st.number_input("Temperatura del Medio Calefactor/Enfriador [ºC]", value=80.0, step=1.0, key="t_medio")
    h = st.number_input("Coeficiente de Convección (h) [W/(m²·K)]", value=100.0, min_value=0.0, step=5.0, key="h_heisler")
    if h == 0: st.warning("Un coeficiente de convección de 0 W/(m²·K) implica aislamiento perfecto o que no hay transferencia de calor por convección.")

    geometria = st.selectbox(
        "Geometría del Alimento:",
        ("Placa Plana", "Cilindro", "Esfera"), key="geometria_heisler"
    )
    if geometria == 'Placa Plana':
        st.info("Para placa plana, la 'Dimensión Característica a' es el semi-espesor.")
    elif geometria == 'Cilindro':
        st.info("Para cilindro, la 'Dimensión Característica a' es el radio.")
    elif geometria == 'Esfera':
        st.info("Para esfera, la 'Dimensión Característica a' es el radio.")
    dimension_a = st.number_input("Dimensión Característica 'a' [m]", value=0.02, min_value=0.0, format="%.4f", help="Radio (cilindro, esfera) o semi-espesor (placa).", key="dimension_a_heisler")
    if dimension_a == 0: st.error("La dimensión característica 'a' no puede ser cero para estos cálculos.")

    # Calculamos propiedades medias para Heisler.
    # Evitar un promedio que caiga en la zona no aplicable si T_inicial y T_medio son muy diferentes
    if T_medio >= Tf_input and T_inicial_alimento >= Tf_input:
        # Ambas en zona no congelada
        T_heisler_props_avg = (T_inicial_alimento + T_medio) / 2
    elif T_medio < Tf_input and T_inicial_alimento < Tf_input:
        # Ambas en zona congelada
        T_heisler_props_avg = (T_inicial_alimento + T_medio) / 2
    else:
        # Cruza la zona de congelación. Se evalúa en un punto "seguro" o se advierte.
        # Por simplicidad, se usa el promedio, pero se advierte.
        T_heisler_props_avg = (T_inicial_alimento + T_medio) / 2
        st.warning(rf"La **temperatura promedio** para la evaluación de las propiedades ($\mathbf{{T_{{heisler\_props\_avg}}}}$ºC) cae en la zona de congelación, siendo menor que la temperatura inicial de congelación ($\mathbf{{T_{{f\_input}}}}$ºC). Los modelos de Choi y Okos usados aquí asumen un comportamiento simple de congelación. Para procesos de congelación profundos, las propiedades pueden variar significativamente, afectando la precisión de Heisler en esa fase.")

    # Se calculan para la advertencia visual, pero se recalculan dentro de la función del botón
    alpha_alimento_medio = calcular_propiedades_alimento(composicion, T_heisler_props_avg, Tf_input)[3]
    k_alimento_medio = calcular_propiedades_alimento(composicion, T_heisler_props_avg, Tf_input)[2]


    if calculation_type == "Temperatura final en el punto frío (ºC)":
        t_minutos = st.number_input("Tiempo de Proceso [min]", value=30.0, min_value=0.0, step=1.0, key="t_minutos_final_temp")
        t_segundos = t_minutos * 60

    elif calculation_type == "Tiempo de proceso para alcanzar una temperatura final (ºC)":
        T_final_alimento = st.number_input("Temperatura Final deseada en el centro [ºC]", value=60.0, step=1.0, key="t_final_alimento_time")
        if (T_inicial_alimento <= T_medio and T_final_alimento < T_inicial_alimento) or \
           (T_inicial_alimento >= T_medio and T_final_alimento > T_inicial_alimento):
            st.error("La temperatura final deseada debe estar entre la temperatura inicial del alimento y la temperatura del medio.")


    elif calculation_type == "Temperatura en una posición específica (X) en el alimento (ºC)":
        t_minutos = st.number_input("Tiempo de Proceso [min]", value=30.0, min_value=0.0, step=1.0, key="t_minutos_pos_temp")
        t_segundos = t_minutos * 60
        posicion_x = st.number_input("Posición 'x' desde el centro [m]", value=0.01, min_value=0.0, format="%.4f", help="Distancia desde el centro (0) hasta el borde (a). Debe ser <= 'a'.", key="posicion_x")
        if posicion_x > dimension_a:
            st.error(f"La posición 'x' ({posicion_x:.4f} m) no puede ser mayor que la dimensión característica 'a' ({dimension_a:.4f} m).")


elif calculation_type == "Tiempo de congelación (min)":
    T_ambiente_congelacion = st.number_input("Temperatura del Medio de Congelación (Ta) [ºC]", value=-20.0, step=1.0, key="t_ambiente_congelacion")
    h_congelacion = st.number_input("Coeficiente de Convección (h) [W/(m²·K)]", value=20.0, min_value=0.0, step=1.0, help="Coeficiente de convección para el proceso de congelación.", key="h_congelacion")
    if h_congelacion == 0: st.error("El coeficiente de convección (h) no puede ser cero.")


    # Para k_f de Plank, se suele evaluar a una temperatura media entre Tf y Ta
    T_kf_plank = (Tf_input + T_ambiente_congelacion) / 2
    # Ajuste para asegurar que la temperatura de evaluación de k_f no esté por encima de Tf
    if T_kf_plank > Tf_input:
          T_kf_plank = Tf_input - 2.0 # Asegurarse de que esté en la zona congelada

    k_alimento_congelado = calcular_propiedades_alimento(composicion, T_kf_plank, Tf_input)[2] # Solo k

    L_e = (composicion['agua'] / 100) * 333.6e3 # J/kg (Calor latente de fusión del hielo a 0°C)
    st.info(f"Calor latente efectivo (Le) utilizado para Plank: {L_e/1000:.2f} kJ/kg (Basado solo en calor latente de fusión del agua inicial).")

    geometria_plank = st.selectbox(
        "Geometría del Alimento:",
        ("Placa Plana", "Cilindro", "Esfera"), key="geometria_plank"
    )
    if geometria_plank == 'Placa Plana':
        st.info("Para placa plana, la 'Dimensión Característica a' es el semi-espesor.")
    elif geometria_plank == 'Cilindro':
        st.info("Para cilindro, la 'Dimensión Característica a' es el radio.")
    elif geometria_plank == 'Esfera':
        st.info("Para esfera, la 'Dimensión Característica a' es el radio.")
    dimension_a_plank = st.number_input("Dimensión Característica 'a' [m]", value=0.02, min_value=0.0, format="%.4f", key="dimension_a_plank")
    if dimension_a_plank == 0: st.error("La dimensión característica 'a' no puede ser cero para el cálculo de Plank.")

elif calculation_type == "Peso Molecular Aparente del Sólido (PMs) [g/mol]":
    st.info("Este cálculo estima el peso molecular promedio del sólido basándose en la temperatura inicial de congelación del alimento y su contenido de agua.")


# --- Botón de cálculo y resultados ---
st.markdown("---")
if st.button("Realizar Cálculo", help="Haz clic para ejecutar el cálculo seleccionado."):
    st.markdown("<h2 style='font-size: 1.4em;'>5. Resultados del Cálculo</h2>", unsafe_allow_html=True)
    if abs(total_composicion - 100.0) > 0.01:
        st.error("Por favor, ajusta los porcentajes de composición para que sumen 100% antes de calcular.")
    else:
        if calculation_type == "Propiedades a T > 0°C" or calculation_type == "Propiedades a T < 0°C":
            densidad_val, cp_val, k_val, alpha_val = calcular_propiedades_alimento(composicion, T_prop, Tf_input)
            st.success(f"Propiedades Termofísicas del Alimento a {T_prop:.1f} °C:")
            st.write(f"**Densidad (ρ):** {densidad_val:.2f} kg/m³")
            st.write(f"**Calor Específico (Cp):** {cp_val:.2f} J/(kg·K)")
            st.write(f"**Conductividad Térmica (k):** {k_val:.4f} W/(m·K)")
            st.write(f"**Difusividad Térmica (α):** {alpha_val:.2e} m²/s")

        elif calculation_type == "Temperatura final en el punto frío (ºC)":
            # Recalcular propiedades medias justo antes del cálculo
            temp_avg_heisler = (T_inicial_alimento + T_medio) / 2
            alpha_alimento_medio_calc = calcular_propiedades_alimento(composicion, temp_avg_heisler, Tf_input)[3]
            k_alimento_medio_calc = calcular_propiedades_alimento(composicion, temp_avg_heisler, Tf_input)[2]

            result = calcular_temperatura_final_punto_frio(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio_calc, k_alimento_medio_calc, h, geometria, dimension_a)
            if result:
                T_final_centro, Fo, Bi, A1, lambda1 = result
                st.success(f"Temperatura en el centro al finalizar el proceso: **{T_final_centro:.2f} °C**")
                st.markdown("---")
                st.subheader("Parámetros Adicionales del Proceso:")
                st.write(f"**Número de Biot (Bi):** {Bi:.2f}")
                st.write(f"**Número de Fourier (Fo):** {Fo:.3f}")
                st.write(f"**Coeficiente A1:** {A1:.4f}")
                st.write(f"**Valor propio Lambda1 (λ1):** {lambda1:.4f}")
                st.write(f"*(Propiedades evaluadas a la temperatura media del proceso: {temp_avg_heisler:.1f} °C)*")

        elif calculation_type == "Tiempo de proceso para alcanzar una temperatura final (ºC)":
            # Recalcular propiedades medias justo antes del cálculo
            temp_avg_heisler = (T_inicial_alimento + T_medio) / 2
            alpha_alimento_medio_calc = calcular_propiedades_alimento(composicion, temp_avg_heisler, Tf_input)[3]
            k_alimento_medio_calc = calcular_propiedades_alimento(composicion, temp_avg_heisler, Tf_input)[2]

            result = calcular_tiempo_para_temperatura(T_final_alimento, T_inicial_alimento, T_medio, alpha_alimento_medio_calc, k_alimento_medio_calc, h, geometria, dimension_a)
            if result and result[0] is not None:
                t_minutos, Fo, Bi, A1, lambda1 = result
                st.success(f"Tiempo necesario para que el centro alcance {T_final_alimento:.1f} °C: **{t_minutos:.2f} minutos**")
                st.markdown("---")
                st.subheader("Parámetros Adicionales del Proceso:")
                st.write(f"**Número de Biot (Bi):** {Bi:.2f}")
                st.write(f"**Número de Fourier (Fo):** {Fo:.3f}")
                st.write(f"**Coeficiente A1:** {A1:.4f}")
                st.write(f"**Valor propio Lambda1 (λ1):** {lambda1:.4f}")
                st.write(f"*(Propiedades evaluadas a la temperatura media del proceso: {temp_avg_heisler:.1f} °C)*")

        elif calculation_type == "Temperatura en una posición específica (X) en el alimento (ºC)":
            # Recalcular propiedades medias justo antes del cálculo
            temp_avg_heisler = (T_inicial_alimento + T_medio) / 2
            alpha_alimento_medio_calc = calcular_propiedades_alimento(composicion, temp_avg_heisler, Tf_input)[3]
            k_alimento_medio_calc = calcular_propiedades_alimento(composicion, temp_avg_heisler, Tf_input)[2]

            result = calcular_temperatura_posicion(t_segundos, T_inicial_alimento, T_medio, alpha_alimento_medio_calc, k_alimento_medio_calc, h, geometria, dimension_a, posicion_x)
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
                st.write(f"*(Propiedades evaluadas a la temperatura media del proceso: {temp_avg_heisler:.1f} °C)*")

        elif calculation_type == "Tiempo de congelación (min)":
            # Recalcular k_alimento_congelado justo antes del cálculo
            T_kf_plank_calc = (Tf_input + T_ambiente_congelacion) / 2
            if T_kf_plank_calc > Tf_input:
                  T_kf_plank_calc = Tf_input - 2.0
            k_alimento_congelado_calc = calcular_propiedades_alimento(composicion, T_kf_plank_calc, Tf_input)[2]

            result = calcular_tiempo_congelacion_plank(Tf_input, T_ambiente_congelacion, h_congelacion, k_alimento_congelado_calc, L_e, geometria_plank, dimension_a_plank)
            if result:
                t_minutos_plank, P_plank, R_plank, Le_plank = result
                st.success(f"Tiempo de congelación estimado (Plank): **{t_minutos_plank:.2f} minutos**")
                st.markdown("---")
                st.subheader("Parámetros Adicionales del Proceso:")
                st.write(f"**Temperatura del medio (Ta):** {T_ambiente_congelacion:.1f} °C")
                st.write(f"**Coeficiente de convección (h):** {h_congelacion:.1f} W/(m²·K)")
                st.write(f"**Conductividad del alimento congelado (kf):** {k_alimento_congelado_calc:.4f} W/(m·K) *(evaluada a {T_kf_plank_calc:.1f}°C)*")
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


# Eliminada la sección de Gráficos

# --- Sección de Información Adicional ---
st.markdown("---")
st.markdown("<h4 style='font-size: 1.4em;'>Información Adicional</h4>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Guía Rápida de Uso", "Referencias Bibliográficas", "Bases de Datos de Composición de Alimentos", "Ecuaciones Utilizadas", "Contacto"])

with tab1:
    st.markdown("<h5 style='font-size: 1.2em;'>Guía Rápida de Uso</h5>", unsafe_allow_html=True)
    st.markdown("""
    Para utilizar esta herramienta de simulación de procesos térmicos, sigue estos sencillos pasos:

   1.  **Define la Composición Proximal:** En la sección "1. Composición Proximal del Alimento (%)", ingresa los porcentajes de **Agua, Proteína, Grasa, Carbohidratos, Fibra** y **Cenizas** de tu alimento. Asegúrate de que la suma total sea **100%**. La aplicación te indicará si necesitas ajustar los valores.

2.  **Define la Temperatura de Congelación (Tf):** En la sección "2. Temperatura de Congelación Inicial (Tf)", introduce la temperatura a la cual el alimento comienza a congelarse.

3.  **Selecciona el Tipo de Cálculo:** En la sección "3. Elige el Cálculo a Realizar", usa las opciones para seleccionar la simulación que deseas ejecutar (por ejemplo, propiedades termofísicas, temperatura final, tiempo de proceso o tiempo de congelación).

4.  **Ingresa los Parámetros Específicos:** En la sección "4. Parámetros del Cálculo", aparecerán los campos de entrada relevantes para tu simulación (temperaturas, coeficientes de convección, geometría, dimensiones, etc.). Completa todos los datos necesarios.

5.  **Realiza el Cálculo:** Haz clic en el botón **"Realizar Cálculo"** ubicado en la parte inferior de la sección de parámetros. Los **resultados** se mostrarán inmediatamente en la sección "5. Resultados del Cálculo", junto con cualquier parámetro adicional relevante.
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
    El tiempo de congelación se calcula utilizando la **ecuación de Plank**, un modelo semi-empírico que estima el tiempo aproximado para congelar un alimento:
    """)
    st.latex(r"""
    t = \frac{L_e}{T_f - T_a} \left( \frac{P \cdot a}{h} + \frac{R \cdot a^2}{k_f} \right)
    """)
    st.markdown(r"""
    Donde:
    * $t$: Tiempo de congelación (s)
    * $L_e$: **Calor latente efectivo** (J/kg), que considera tanto el calor latente de congelación del agua como el calor sensible involucrado en el proceso.
    * $T_f$: **Temperatura inicial de congelación** del alimento (°C)
    * $T_a$: **Temperatura del medio ambiente** de congelación (°C)
    * $P, R$: **Factores geométricos** específicos para cada forma (ver tabla a continuación)
    * $a$: **Dimensión característica** (radio para cilindro/esfera, semiespesor para placa) (m)
    * $h$: **Coeficiente de transferencia de calor por convección** (W/(m²·K))
    * $k_f$: **Conductividad térmica del alimento congelado** (W/(m·K)), evaluada típicamente a la temperatura media del proceso de congelación.
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
    * $Fo$: **Número de Fourier**, representa la relación entre la conducción de calor y el almacenamiento de energía dentro de un cuerpo.
    * $Bi$: **Número de Biot**, indica la relación entre la resistencia a la transferencia de calor por convección en la superficie y la resistencia a la conducción dentro del cuerpo.
    * $\alpha$: **Difusividad térmica** del alimento (m²/s)
    * $t$: **Tiempo** (s)
    * $L_c$: **Longitud característica** (m)
    * $h$: **Coeficiente de transferencia de calor por convección** (W/(m²·K))
    * $k$: **Conductividad térmica** del alimento (W/(m·K))
    """)

    st.markdown("""
    **a) Temperatura Final en el Punto Frío (Centro, $x=0$):**
    """)
    st.markdown("""
    Esta ecuación se usa para encontrar la temperatura en el **centro** del alimento a un tiempo dado.
    """)
    st.latex(r"""
    \frac{T_{\text{centro}}(t) - T_{\infty}}{T_i - T_{\infty}} = A_1 \cdot \exp(-\lambda_1^2 \cdot Fo)
    """)
    st.markdown(r"""
    Donde:
    * $T_{\text{centro}}(t)$: **Temperatura en el centro** al tiempo $t$ (°C)
    * $T_i$: **Temperatura inicial uniforme** del alimento (°C)
    * $T_{\infty}$: **Temperatura del medio ambiente** (°C)
    * $A_1, \lambda_1$: **Coeficientes y valores propios** del primer término, que dependen de la geometría y del número de Biot ($Bi$). Se obtienen de tablas o soluciones numéricas.
    """)

    st.markdown("""
    **b) Tiempo de Proceso para Alcanzar una Temperatura Final:**
    """)
    st.markdown("""
    Para determinar el **tiempo ($t$)** necesario para que el **centro** del alimento alcance una temperatura específica ($T_{final}$), se despeja $t$ de la ecuación anterior. Esto aplica tanto para **calentamiento** como para **enfriamiento hasta una temperatura superior a la de congelación ($T > T_f$)**:
    """)
    st.latex(r"""
    t = -\frac{L_c^2}{\alpha \cdot \lambda_1^2} \cdot \ln \left( \frac{1}{A_1} \cdot \frac{T_{final} - T_{\infty}}{T_i - T_{\infty}} \right)
    """)

    st.markdown("""
    **c) Temperatura en una Posición Específica (X) en el Alimento:**
    """)
    st.markdown("""
    La temperatura en una posición $x$ (distancia desde el centro) se calcula multiplicando la relación de temperatura del centro por un **factor de posición $X(x/L_c, \lambda_1)$**:
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
        $X(x/L_c, \lambda_1) = J_0(\lambda_1 \cdot x/L_c)$ (donde $J_0$ es la **función de Bessel de primera clase, orden cero**)
    * **Esfera:**
    """)
    st.latex(r"""
    X(x/L_c, \lambda_1) = \frac{\sin(\lambda_1 \cdot x/L_c)}{\lambda_1 \cdot x/L_c}
    """)
    st.markdown("""
    *Nota: Para el cálculo en cilindros, se requiere la función de Bessel de primera clase de orden cero ($J_0$), que se obtiene de librerías matemáticas como `scipy.special`.*
    """)

# Nueva pestaña de Contacto
with tab5:
    st.markdown("<h5 style='font-size: 1.2em;'>Contacto</h5>", unsafe_allow_html=True)
    st.markdown("""
    *Dra. Mg. Ing. Química Silvia Marcela Miro Erdmann* 🔬
    * smmiroer@gmail.com
    * Profesor Adjunto
    * Facultad de Ingeniería y Ciencias Agropecuarias - Universidad Nacional de San Luis (FICA-UNSL)
    * Escuela de Ingeniería - Universidad Nacional de Villa Mercedes (EI-UNVM)
    """)
