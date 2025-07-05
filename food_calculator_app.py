import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0 # Importar j0 para cilindros

# --- 1. Funciones para calcular la PROPIEDAD DE CADA COMPONENTE en función de la TEMPERATURA ---

# --- DENSIDAD (rho) en kg/m^3 ---
def densidad_agua(t):
    """Calcula la densidad del agua en kg/m^3 a la temperatura t (°C)."""
    if t >= 0: # Agua líquida
        return 997.18 + 3.1439e-3 * t - 3.7574e-3 * t**2
    else: # Hielo
        return 916.89 - 0.13071 * t

def densidad_proteina(t):
    """Calcula la densidad de la proteína en kg/m^3 a la temperatura t (°C)."""
    return 1329.9 - 0.5184 * t

def densidad_grasa(t):
    """Calcula la densidad de la grasa en kg/m^3 a la temperatura t (°C)."""
    return 925.59 - 0.41757 * t

def densidad_carbohidrato(t):
    """Calcula la densidad del carbohidrato en kg/m^3 a la temperatura t (°C)."""
    return 1599.1 - 0.31046 * t

def densidad_fibra(t):
    """Calcula la densidad de la fibra en kg/m^3 a la temperatura t (°C)."""
    return 1311.5 - 0.36589 * t

def densidad_cenizas(t):
    """Calcula la densidad de las cenizas en kg/m^3 a la temperatura t (°C)."""
    return 2423.8 - 0.28063 * t

# --- CALOR ESPECÍFICO (Cp) en J/(kg·K) o J/(kg·°C) ---
def cp_agua(t):
    """Calcula el calor específico del agua en J/(kg·K) a la temperatura t (°C)."""
    if t >= 0: # Agua líquida
        return 4176.2 - 9.0864e-2 * t + 5.4731e-3 * t**2
    else: # Hielo
        return 2062.3 + 6.0769 * t

def cp_proteina(t):
    """Calcula el calor específico de la proteína en J/(kg·K) a la temperatura t (°C)."""
    return 2008.2 + 1.2089 * t - 1.3129e-3 * t**2

def cp_grasa(t):
    """Calcula el calor específico de la grasa en J/(kg·K) a la temperatura t (°C)."""
    return 1984.2 + 1.4733 * t - 4.8008e-3 * t**2

def cp_carbohidrato(t):
    """Calcula el calor específico del carbohidrato en J/(kg·K) a la temperatura t (°C)."""
    return 1548.8 + 1.9625 * t - 5.9399e-3 * t**2

def cp_fibra(t):
    """Calcula el calor específico de la fibra en J/(kg·K) a la temperatura t (°C)."""
    return 1845.9 + 1.8306 * t - 4.6509e-3 * t**2

def cp_cenizas(t):
    """Calcula el calor específico de las cenizas en J/(kg·K) a la temperatura t (°C)."""
    return 1092.6 + 1.8896 * t - 3.6817e-3 * t**2

# --- CONDUCTIVIDAD TÉRMICA (k) en W/(m·K) ---
def k_agua(t):
    """Calcula la conductividad térmica del agua en W/(m·K) a la temperatura t (°C)."""
    if t >= 0: # Agua líquida
        return 0.57109 + 1.7625e-3 * t - 6.7036e-6 * t**2
    else: # Hielo
        return 2.2196 - 6.2489e-3 * t + 1.0154e-4 * t**2

def k_proteina(t):
    """Calcula la conductividad térmica de la proteína en W/(m·K) a la temperatura t (°C)."""
    return 0.17881 + 1.1958e-3 * t - 2.7178e-6 * t**2

def k_grasa(t):
    """Calcula la conductividad térmica de la grasa en W/(m·K) a la temperatura t (°C)."""
    return 0.18071 - 2.7604e-4 * t - 1.7749e-7 * t**2

def k_carbohidrato(t):
    """Calcula la conductividad térmica del carbohidrato en W/(m·K) a la temperatura t (°C)."""
    return 0.20141 + 1.3874e-3 * t - 4.3312e-6 * t**2

def k_fibra(t):
    """Calcula la conductividad térmica de la fibra en W/(m·K) a la temperatura t (°C)."""
    return 0.18331 + 1.2497e-3 * t - 3.1683e-6 * t**2

def k_cenizas(t):
    """Calcula la conductividad térmica de las cenizas en W/(m·K) a la temperatura t (°C)."""
    return 0.32962 + 1.4011e-3 * t - 2.9069e-6 * t**2

# --- DIFUSIVIDAD TÉRMICA (alpha) en m^2/s ---
def alpha_agua(t):
    """Calcula la difusividad térmica del agua en m^2/s a la temperatura t (°C)."""
    if t >= 0: # Agua líquida
        return 1.3168e-7 + 6.2477e-10 * t - 2.4022e-12 * t**2
    else: # Hielo
        return 1.1756e-6 - 6.0833e-9 * t + 9.5037e-11 * t**2

def alpha_proteina(t):
    """Calcula la difusividad térmica de la proteína en m^2/s a la temperatura t (°C)."""
    return 9.8777e-8 - 1.2569e-11 * t - 3.8286e-14 * t**2

def alpha_grasa(t):
    """Calcula la difusividad térmica de la grasa en m^2/s a la temperatura t (°C)."""
    return 6.8714e-8 + 4.7578e-10 * t - 1.4646e-12 * t**2

def alpha_carbohidrato(t):
    """Calcula la difusividad térmica del carbohidrato en m^2/s a la temperatura t (°C)."""
    return 8.0842e-8 + 5.3052e-10 * t - 2.3218e-12 * t**2

def alpha_fibra(t):
    """Calcula la difusividad térmica de la fibra en m^2/s a la temperatura t (°C)."""
    return 7.3976e-8 + 5.1902e-10 * t - 2.2202e-12 * t**2

def alpha_cenizas(t):
    """Calcula la difusividad térmica de las cenizas en m^2/s a la temperatura t (°C)."""
    return 1.2461e-7 + 3.7321e-10 * t - 1.2244e-12 * t**2

# --- Función para calcular la fracción de hielo ---
def calcular_fraccion_hielo(t, agua_porcentaje, Tf_input):
    """
    Calcula la fracción de hielo (Xi) en un alimento a una temperatura t (°C)
    dada la temperatura inicial de congelación (Tf_input)
    y el porcentaje de agua inicial.
    Asume calor latente de fusión del hielo de 333.6 kJ/kg.
    """
    L0 = 333.6 * 1000 # Calor latente de fusión del hielo a 0°C en J/kg (333.6 kJ/kg)

    if t >= Tf_input:
        return 0.0 # No hay hielo si la temperatura es mayor o igual a la de congelación inicial
    elif t < Tf_input:
        # Ecuación simplificada para fracción de hielo (asumiendo propiedades de solución diluida)
        # Esto es una aproximación y puede variar según el modelo de congelación
        # Se asume Cp del agua liquida aprox 4186 J/(kg.K) para el termino del calor sensible
        if (Tf_input - t) == 0: # Evitar división por cero si t == Tf_input
            return 0.0
        Xi = (L0 / (4186 * (Tf_input - t))) * (agua_porcentaje / 100) # Ajustado a 4186 J/(kg.K)
        return min(max(0.0, Xi), agua_porcentaje / 100) # Asegura que esté entre 0 y el contenido total de agua
    else:
        return 0.0


# --- 2. Funciones para calcular la PROPIEDAD DEL ALIMENTO COMPLETO ---

def calcular_densidad_alimento(t, composicion, Tf_input):
    """
    Calcula la densidad del alimento usando las ecuaciones de Choi y Okos,
    considerando la fracción de hielo si la temperatura es de congelación.
    """
    if abs(sum(composicion.values()) - 100) > 0.01: # Usar una pequeña tolerancia para la suma
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100 # Fracción de agua inicial
    Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0), Tf_input) # Fracción de hielo
    Xu = Xw_inicial - Xi # Fracción de agua no congelada (líquida)

    # Convertir porcentajes a fracciones de masa
    Xp = composicion.get('proteina', 0) / 100
    Xf = composicion.get('grasa', 0) / 100
    Xc = composicion.get('carbohidrato', 0) / 100
    Xfi = composicion.get('fibra', 0) / 100
    Xa = composicion.get('cenizas', 0) / 100

    rho_alimento_inv = (Xu / densidad_agua(t)) + \
                       (Xi / densidad_agua(t)) + \
                       (Xp / densidad_proteina(t)) + \
                       (Xf / densidad_grasa(t)) + \
                       (Xc / densidad_carbohidrato(t)) + \
                       (Xfi / densidad_fibra(t)) + \
                       (Xa / densidad_cenizas(t))
    return 1 / rho_alimento_inv


def calcular_cp_alimento(t, composicion, Tf_input):
    """
    Calcula el calor específico del alimento usando las ecuaciones de Choi y Okos,
    considerando la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100
    Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0), Tf_input)
    Xu = Xw_inicial - Xi

    Xp = composicion.get('proteina', 0) / 100
    Xf = composicion.get('grasa', 0) / 100
    Xc = composicion.get('carbohidrato', 0) / 100
    Xfi = composicion.get('fibra', 0) / 100
    Xa = composicion.get('cenizas', 0) / 100

    cp_alimento = (Xu * cp_agua(t)) + \
                  (Xi * cp_agua(t)) + \
                  (Xp * cp_proteina(t)) + \
                  (Xf * cp_grasa(t)) + \
                  (Xc * cp_carbohidrato(t)) + \
                  (Xfi * cp_fibra(t)) + \
                  (Xa * cp_cenizas(t))
    return cp_alimento


def calcular_k_alimento(t, composicion, Tf_input):
    """
    Calcula la conductividad térmica del alimento usando las ecuaciones de Choi y Okos,
    considerando la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100
    Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0), Tf_input)
    Xu = Xw_inicial - Xi

    Xp = composicion.get('proteina', 0) / 100
    Xf = composicion.get('grasa', 0) / 100
    Xc = composicion.get('carbohidrato', 0) / 100
    Xfi = composicion.get('fibra', 0) / 100
    Xa = composicion.get('cenizas', 0) / 100

    k_alimento = (Xu * k_agua(t)) + \
                 (Xi * k_agua(t)) + \
                 (Xp * k_proteina(t)) + \
                 (Xf * k_grasa(t)) + \
                 (Xc * k_carbohidrato(t)) + \
                 (Xfi * k_fibra(t)) + \
                 (Xa * k_cenizas(t))
    return k_alimento


def calcular_alpha_alimento(t, composicion, Tf_input):
    """
    Calcula la difusividad térmica del alimento usando las ecuaciones de Choi y Okos,
    considerando la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    # Recalcula las propiedades auxiliares (rho, Cp, k) ya que la difusividad depende de ellas
    # y deben considerar la fracción de hielo.
    densidad = calcular_densidad_alimento(t, composicion, Tf_input)
    cp = calcular_cp_alimento(t, composicion, Tf_input)
    k = calcular_k_alimento(t, composicion, Tf_input)

    if densidad * cp == 0: # Evitar división por cero
        return 0.0
    return k / (densidad * cp)

# --- Función para calcular el tiempo de congelación (Ecuación de Plank) ---
def calcular_tiempo_congelacion(composicion, T0, Ta, h, geometria, dimension_a, Tf_input):
    """
    Calcula el tiempo de congelación usando la Ecuación de Plank.
    :param composicion: Diccionario con porcentajes de los componentes.
    :param T0: Temperatura inicial del alimento (°C).
    :param Ta: Temperatura del medio ambiente de congelación (°C).
    :param h: Coeficiente de transferencia de calor por convección (W/(m²·K)).
    :param geometria: Tipo de geometría ('Placa', 'Cilindro', 'Esfera').
    :param dimension_a: Dimensión característica del alimento (m).
    :param Tf_input: Temperatura inicial de congelación del alimento (°C) ingresada por el usuario.
    :return: Tiempo de congelación en horas.
    """
    # Constantes
    L0 = 333.6 * 1000 # Calor latente de fusión del hielo a 0°C en J/kg

    # Validaciones
    if Ta >= Tf_input: # Usa Tf_input aquí
        st.warning("La temperatura del medio ambiente de congelación (Ta) debe ser menor que la temperatura de congelación inicial del alimento (Tf).")
        return None
    if h <= 0:
        st.warning("El coeficiente de transferencia de calor (h) debe ser un valor positivo.")
        return None
    if dimension_a <= 0:
        st.warning("La dimensión característica (a) debe ser un valor positivo.")
        return None

    # Calcular propiedades promedio del alimento congelado (a una T de referencia)
    # Usaremos una temperatura ligeramente por debajo de Tf_input para asegurar propiedades de hielo.
    temp_prop_congelado = max(Ta, Tf_input - 5) # Usa Tf_input aquí
    rho_f = calcular_densidad_alimento(temp_prop_congelado, composicion, Tf_input) # Pasa Tf_input
    k_f = calcular_k_alimento(temp_prop_congelado, composicion, Tf_input) # Pasa Tf_input

    # Factores de forma P y R según la geometría
    P, R = 0, 0
    if geometria == 'Placa':
        P = 0.5
        R = 0.125
    elif geometria == 'Cilindro':
        P = 0.25
        R = 0.0625
    elif geometria == 'Esfera':
        P = 0.166667 # 1/6
        R = 0.041667 # 1/24
    else:
        st.error("Geometría no válida seleccionada para el cálculo del tiempo de congelación.")
        return None

    # Calor latente efectivo (considera solo la fracción de agua inicial que se congela)
    L_efectivo = L0 * (composicion.get('agua', 0) / 100)

    # Ecuación de Plank (tiempo en segundos)
    # No incluye el pre-enfriamiento de T0 a Tf en esta versión simple.
    if (Tf_input - Ta) == 0: # Usa Tf_input aquí
        return float('inf')

    tiempo_segundos = (L_efectivo / (Tf_input - Ta)) * ((P * dimension_a / h) + (R * dimension_a**2 / k_f))

    return tiempo_segundos / 3600 # Convertir segundos a horas


# --- Funciones para el cálculo de tiempo de escaldado y perfil de temperatura ---

def get_heisler_coeffs(bi, geometry):
    """
    Obtiene los coeficientes A1 y lambda1 para la solución del primer término de Heisler.
    Basado en tablas de transferencia de calor (aproximaciones para Bi comunes).
    """
    # Estos valores son aproximaciones o interpolaciones de tablas estándar.
    # Para mayor precisión, se requeriría una interpolación más robusta o la solución numérica de las raíces.
    # Bi = h*Lc/k
    if geometry == 'Placa':
        if bi <= 0.01: return 1.0000, 0.0998
        if bi <= 0.02: return 1.0000, 0.1410
        if bi <= 0.05: return 1.0001, 0.2217
        if bi <= 0.1: return 1.0000, 0.3111
        if bi <= 0.2: return 1.0010, 0.4328
        if bi <= 0.3: return 1.0024, 0.5218
        if bi <= 0.4: return 1.0045, 0.5932
        if bi <= 0.5: return 1.0069, 0.6533
        if bi <= 0.6: return 1.0095, 0.7051
        if bi <= 0.7: return 1.0122, 0.7506
        if bi <= 0.8: return 1.0149, 0.7910
        if bi <= 0.9: return 1.0177, 0.8274
        if bi <= 1.0: return 1.0202, 0.8603
        if bi <= 2.0: return 1.0476, 1.0765
        if bi <= 3.0: return 1.0699, 1.1925
        if bi <= 4.0: return 1.0858, 1.2646
        if bi <= 5.0: return 1.0980, 1.3138
        if bi <= 10.0: return 1.1578, 1.4289
        return 1.2732, 1.5708 # Bi -> inf (resistencia conductiva dominante)

    elif geometry == 'Cilindro':
        if bi <= 0.01: return 1.0000, 0.1412
        if bi <= 0.02: return 1.0000, 0.1995
        if bi <= 0.05: return 1.0001, 0.3142
        if bi <= 0.1: return 1.0000, 0.4417
        if bi <= 0.2: return 1.0020, 0.6170
        if bi <= 0.3: return 1.0050, 0.7496
        if bi <= 0.4: return 1.0089, 0.8516
        if bi <= 0.5: return 1.0135, 0.9372
        if bi <= 0.6: return 1.0185, 1.0116
        if bi <= 0.7: return 1.0238, 1.0776
        if bi <= 0.8: return 1.0292, 1.1373
        if bi <= 0.9: return 1.0346, 1.1924
        if bi <= 1.0: return 1.0399, 1.2427
        if bi <= 2.0: return 1.0967, 1.5994
        if bi <= 3.0: return 1.1411, 1.7887
        if bi <= 4.0: return 1.1750, 1.9081
        if bi <= 5.0: return 1.2029, 1.9969
        if bi <= 10.0: return 1.2650, 2.2305
        return 1.6020, 2.4048 # Bi -> inf

    elif geometry == 'Esfera':
        if bi <= 0.01: return 1.0000, 0.1730
        if bi <= 0.02: return 1.0000, 0.2445
        if bi <= 0.05: return 1.0001, 0.3870
        if bi <= 0.1: return 1.0000, 0.5482
        if bi <= 0.2: return 1.0030, 0.7593
        if bi <= 0.3: return 1.0075, 0.9208
        if bi <= 0.4: return 1.0132, 1.0528
        if bi <= 0.5: return 1.0197, 1.1656
        if bi <= 0.6: return 1.0269, 1.2644
        if bi <= 0.7: return 1.0345, 1.3525
        if bi <= 0.8: return 1.0423, 1.4320
        if bi <= 0.9: return 1.0503, 1.5044
        if bi <= 1.0: return 1.0581, 1.5708
        if bi <= 2.0: return 1.1827, 2.0288
        if bi <= 3.0: return 1.2801, 2.2889
        if bi <= 4.0: return 1.3473, 2.4556
        if bi <= 5.0: return 1.3978, 2.5704
        if bi <= 10.0: return 1.5441, 2.8363
        return 2.0000, 3.1416 # Bi -> inf
    return None, None # En caso de geometría no reconocida


def calcular_tiempo_escaldado(T_inicial_alimento, T_final_alimento_centro, T_medio_escaldado, h_escaldado, k_alimento_medio, alpha_alimento_medio, geometria, dimension_a):
    """
    Calcula el tiempo de escaldado para que el centro del alimento alcance una temperatura objetivo.
    Usa la solución del primer término de la serie de Fourier (cartas de Heisler).
    :param T_inicial_alimento: Temperatura inicial uniforme del alimento (°C).
    :param T_final_alimento_centro: Temperatura objetivo en el centro del alimento (°C).
    :param T_medio_escaldado: Temperatura del medio de calentamiento (°C).
    :param h_escaldado: Coeficiente de transferencia de calor por convección para escaldado (W/(m²·K)).
    :param k_alimento_medio: Conductividad térmica del alimento a la temperatura media (W/(m·K)).
    :param alpha_alimento_medio: Difusividad térmica del alimento a la temperatura media (m²/s).
    :param geometria: Tipo de geometría ('Placa', 'Cilindro', 'Esfera').
    :param dimension_a: Dimensión característica del alimento (m).
    :return: Tiempo de escaldado en segundos.
    """
    if T_medio_escaldado <= T_final_alimento_centro:
        st.warning("La temperatura del medio de escaldado debe ser mayor que la temperatura final deseada en el centro del alimento.")
        return None
    if h_escaldado <= 0 or k_alimento_medio <= 0 or alpha_alimento_medio <= 0 or dimension_a <= 0:
        st.warning("Los valores de h, k, alpha y dimensión 'a' deben ser positivos para el cálculo del tiempo de escaldado.")
        return None

    # Longitud característica Lc
    Lc = dimension_a # Para placa es L, para cilindro/esfera es R

    # Número de Biot (Bi)
    if k_alimento_medio == 0: # Evitar división por cero
        st.error("Conductividad térmica del alimento es cero, no se puede calcular el número de Biot.")
        return None
    Bi = (h_escaldado * Lc) / k_alimento_medio

    # Obtener coeficientes A1 y lambda1
    A1, lambda1 = get_heisler_coeffs(Bi, geometria)

    if A1 is None or lambda1 is None:
        st.error(f"No se encontraron coeficientes A1 y lambda1 para Bi={Bi:.2f} y geometría {geometria}. Revise los rangos.")
        return None

    # Relación de temperatura no dimensional en el centro
    theta_0 = (T_final_alimento_centro - T_medio_escaldado) / (T_inicial_alimento - T_medio_escaldado)

    # Validaciones adicionales para logaritmo
    if theta_0 <= 0 or A1 <= 0:
        st.warning("La relación de temperatura o A1 no son válidos para calcular el tiempo. Revise las temperaturas.")
        return None
    if theta_0 / A1 <= 0:
        st.warning("El argumento del logaritmo es no positivo. Esto puede indicar que la temperatura final deseada es inalcanzable con la temperatura inicial dada, o que hay un problema con A1. Revise las temperaturas.")
        return None

    # Calcular el número de Fourier (Fo)
    if lambda1 == 0: # Evitar división por cero
        st.warning("Lambda1 es cero, no se puede calcular el número de Fourier.")
        return None
    
    try:
        Fo = - (1 / (lambda1**2)) * np.log(theta_0 / A1)
    except Exception as e:
        st.error(f"Error inesperado al calcular Fo: {e}. Asegúrese de que T_final_alimento_centro sea alcanzable y que las temperaturas sean lógicas.")
        return None


    # Calcular el tiempo (t) a partir de Fo
    if alpha_alimento_medio == 0: # Evitar división por cero
        st.warning("Difusividad térmica del alimento es cero, no se puede calcular el tiempo.")
        return None
    
    tiempo_segundos = (Fo * Lc**2) / alpha_alimento_medio
    return tiempo_segundos


def calcular_perfil_temperatura(t_final_segundos, T_inicial_alimento, T_medio_escaldado, alpha_alimento_medio, k_alimento_medio, h_escaldado, geometria, dimension_a, num_puntos=50):
    """
    Calcula el perfil de temperatura a través del alimento en un tiempo dado.
    Usa la solución del primer término de la serie de Fourier.
    :param t_final_segundos: Tiempo final en segundos.
    :param T_inicial_alimento: Temperatura inicial uniforme del alimento (°C).
    :param T_medio_escaldado: Temperatura del medio de calentamiento (°C).
    :param alpha_alimento_medio: Difusividad térmica del alimento a la temperatura media (m²/s).
    :param k_alimento_medio: Conductividad térmica del alimento a la temperatura media (W/(m·K)).
    :param h_escaldado: Coeficiente de transferencia de calor por convección para escaldado (W/(m²·K)).
    :param geometria: Tipo de geometría ('Placa', 'Cilindro', 'Esfera').
    :param dimension_a: Dimensión característica del alimento (m).
    :param num_puntos: Número de puntos para la gráfica del perfil.
    :return: Tupla de (posiciones_adimensionales, temperaturas_en_puntos).
    """
    Lc = dimension_a
    
    if k_alimento_medio == 0:
        st.error("Conductividad térmica es cero, no se puede calcular el Número de Biot para el perfil.")
        return None, None
    Bi = (h_escaldado * Lc) / k_alimento_medio

    A1, lambda1 = get_heisler_coeffs(Bi, geometria)

    if A1 is None or lambda1 is None:
        st.error(f"No se pudieron obtener coeficientes A1 y lambda1 para el perfil de temperatura. Bi={Bi:.2f}, Geometría={geometria}")
        return None, None
    
    if alpha_alimento_medio == 0 or Lc == 0:
        st.error("Difusividad térmica o longitud característica son cero, no se puede calcular el Número de Fourier para el perfil.")
        return None, None

    Fo = (alpha_alimento_medio * t_final_segundos) / Lc**2

    # Generar puntos de posición adimensional (x/L o r/R)
    posiciones_adimensionales = np.linspace(0, 1, num_puntos)
    term_posicion = None # Inicializar para evitar errores de referencia

    if geometria == 'Placa':
        term_posicion = np.cos(lambda1 * posiciones_adimensionales) / np.cos(lambda1)
        
    elif geometry == 'Cilindro':
        try:
            # j0 es la función de Bessel de primera clase de orden cero
            term_posicion = j0(lambda1 * posiciones_adimensionales) / j0(lambda1)
        except Exception as e:
            st.error(f"Error al calcular perfil para Cilindro (J0): {e}. Asegúrese de que 'scipy' esté en requirements.txt.")
            return None, None

    elif geometry == 'Esfera':
        try:
            term_posicion_sin = np.sin(lambda1 * posiciones_adimensionales)
            term_posicion_denom = (lambda1 * posiciones_adimensionales)
            
            # Manejar el caso donde el denominador es cero (en el centro, r/R = 0), el límite es 1.0
            term_posicion = np.where(term_posicion_denom == 0, 1.0, term_posicion_sin / term_posicion_denom)
            # Normalizar por el valor en la superficie (sin(lambda1)/lambda1) para la expresión de Heisler
            if (lambda1 * np.sin(lambda1)) != 0:
                 term_posicion = term_posicion * (lambda1 * np.sin(lambda1)) / (np.sin(lambda1)) # esto no es una división
                 # La forma correcta es (sin(lambda1 * r/R) / (lambda1 * r/R)) / (sin(lambda1) / lambda1)
                 term_posicion = term_posicion / (np.sin(lambda1) / lambda1)
            else: # Manejo si sin(lambda1)/lambda1 es cero o indefinido
                st.warning("Problema con lambda1 para Esfera en la normalización del perfil. Revisa los valores.")
                return None, None

        except Exception as e:
            st.error(f"Error al calcular el perfil de temperatura para Esfera: {e}. Se requiere un manejo robusto del límite.")
            return None, None
    else: # Bloque else final para geometría no válida
        st.error("Geometría no válida seleccionada para el perfil de temperatura.")
        return None, None

    # Si por alguna razón term_posicion sigue siendo None (ej. en caso de error en los try/except)
    if term_posicion is None:
        return None, None

    # Calcular la temperatura adimensional en cada punto
    # Theta(x,t) = (T(x,t) - T_inf) / (T_i - T_inf)
    # Theta(x,t) = A1 * exp(-lambda1^2 * Fo) * X(x/L, lambda1)
    
    exp_term = np.exp(-lambda1**2 * Fo)

    theta_x_t = A1 * exp_term * term_posicion

    # Convertir a temperatura real
    temperaturas_en_puntos = T_medio_escaldado + theta_x_t * (T_inicial_alimento - T_medio_escaldado)

    return posiciones_adimensionales, temperaturas_en_puntos


# --- CONFIGURACIÓN DE LA INTERFAZ CON STREAMLIT ---

st.set_page_config(
    page_title="Calculador de Propiedades de Alimentos",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🍔 Calculador de Propiedades Termofísicas de Alimentos 🌡️")
st.markdown("Dra. Silvia Marcela Miró Erdmann - Profesor Adjunto UNSL/ UNViMe")
st.markdown("Calcula densidad, calor específico, conductividad y difusividad térmica usando las ecuaciones de Choi y Okos (1986).")

st.sidebar.header("Datos de Entrada")

# Entrada de Temperatura (ahora como number_input)
temperatura = st.sidebar.number_input("Temperatura de Propiedades (°C)", min_value=-40.0, max_value=150.0, value=25.0, step=0.1,
                                     help="Temperatura a la que se calcularán las propiedades termofísicas (densidad, Cp, k, alpha).")

st.sidebar.subheader("Composición Proximal (%)")

# Entradas de Composición
agua = st.sidebar.number_input("Agua (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1)
proteina = st.sidebar.number_input("Proteína (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)
grasa = st.sidebar.number_input("Grasa (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
carbohidrato = st.sidebar.number_input("Carbohidratos (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1)
fibra = st.sidebar.number_input("Fibra (%)", min_value=0.0, max_value=100.0, value=0.5, step=0.1)
cenizas = st.sidebar.number_input("Cenizas (%)", min_value=0.0, max_value=100.0, value=0.5, step=0.1)

composicion_total = agua + proteina + grasa + carbohidrato + fibra + cenizas

st.sidebar.write(f"Suma de la composición: **{composicion_total:.1f}%**")

if abs(composicion_total - 100) > 0.01:
    st.sidebar.error("La suma de los porcentajes debe ser 100%. Por favor, ajuste la composición.")

# --- CAMPOS PARA TIEMPO DE CONGELACIÓN ---
st.sidebar.header("Datos para Tiempo de Congelación")

# Temperatura inicial de congelación (Tf_input)
Tf_input = st.sidebar.number_input("Temperatura Inicial de Congelación (Tf) [°C]", min_value=-50.0, max_value=0.0, value=-1.8, step=0.1,
                                   help="Temperatura a la que comienza la congelación del alimento. Depende de la concentración de solutos.")

T0 = st.sidebar.number_input("Temperatura Inicial del Alimento (°C)", min_value=-40.0, max_value=150.0, value=20.0, step=0.1,
                             help="Temperatura del alimento antes de iniciar la congelación.")
Ta = st.sidebar.number_input("Temperatura del Medio de Congelación (°C)", min_value=-60.0, max_value=0.0, value=-20.0, step=0.1,
                             help="Temperatura del aire o medio refrigerante.")
h = st.sidebar.number_input("Coeficiente de Convección (h) [W/(m²·K)] (Congelación)", min_value=1.0, max_value=1000.0, value=15.0, step=0.1,
                            help="Coeficiente de transferencia de calor por convección en la superficie del alimento durante la congelación.")

geometria = st.sidebar.selectbox("Geometría del Alimento", ['Placa', 'Cilindro', 'Esfera'],
                                 help="Selecciona la forma geométrica del alimento.")

dimension_a = st.sidebar.number_input("Dimensión Característica 'a' (m)", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f",
                                      help="Para Placa: mitad del espesor; para Cilindro/Esfera: radio.")

# --- NUEVOS CAMPOS PARA ESCALDADO ---
st.sidebar.header("Datos para Escaldado")
temp_inicial_escaldado = st.sidebar.number_input("Temperatura Inicial Escaldado (°C)", min_value=0.0, max_value=100.0, value=20.0, step=0.1,
                                                help="Temperatura inicial del alimento antes del escaldado.")
temp_final_escaldado = st.sidebar.number_input("Temperatura Final Escaldado (°C)", min_value=0.0, max_value=100.0, value=85.0, step=0.1,
                                              help="Temperatura deseada del alimento después del escaldado (en el centro).")
T_medio_escaldado = st.sidebar.number_input("Temperatura del Medio de Escaldado (T∞) [°C]", min_value=0.0, max_value=150.0, value=95.0, step=0.1,
                                            help="Temperatura del medio de calentamiento (ej. agua caliente, vapor).")
h_escaldado = st.sidebar.number_input("Coeficiente de Convección (h) [W/(m²·K)] (Escaldado)", min_value=1.0, max_value=5000.0, value=100.0, step=0.1,
                                     help="Coeficiente de transferencia de calor por convección en la superficie del alimento durante el escaldado.")


# Botón de cálculo
if st.sidebar.button("Calcular Propiedades y Tiempos"):
    if abs(composicion_total - 100) > 0.01:
        st.error("Por favor, corrija la composición antes de calcular (debe sumar 100%).")
    else:
        composicion = {
            'agua': agua,
            'proteina': proteina,
            'grasa': grasa,
            'carbohidrato': carbohidrato,
            'fibra': fibra,
            'cenizas': cenizas
        }

        with st.spinner("Calculando..."):
            try:
                # --- Sección de Propiedades Termofísicas Generales ---
                st.subheader("Resultados de Propiedades Termofísicas Generales")
                st.write(f"**Temperatura de Propiedades:** {temperatura}°C")
                st.write("---")
                densidad = calcular_densidad_alimento(temperatura, composicion, Tf_input)
                cp = calcular_cp_alimento(temperatura, composicion, Tf_input)
                k = calcular_k_alimento(temperatura, composicion, Tf_input)
                alpha = calcular_alpha_alimento(temperatura, composicion, Tf_input)

                st.metric(label="Densidad (ρ)", value=f"{densidad:.2f} kg/m³")
                st.metric(label="Calor Específico (Cp)", value=f"{cp:.2f} J/(kg·K)")
                st.metric(label="Conductividad Térmica (k)", value=f"{k:.4f} W/(m·K)")
                st.metric(label="Difusividad Térmica (α)", value=f"{alpha:.2e} m²/s")

                fraccion_hielo_actual = calcular_fraccion_hielo(temperatura, composicion.get('agua', 0), Tf_input)
                st.info(f"Fracción de Hielo a {temperatura}°C: {fraccion_hielo_actual:.3f} (kg hielo / kg alimento)")


                # --- Sección de Tiempo de Congelación ---
                st.write("---")
                st.subheader("Tiempo de Congelación (Ecuación de Plank)")

                tiempo_congelacion_horas = calcular_tiempo_congelacion(composicion, T0, Ta, h, geometria, dimension_a, Tf_input)

                if tiempo_congelacion_horas is not None:
                    st.metric(label="Tiempo de Congelación", value=f"{tiempo_congelacion_horas:.2f} horas")
                else:
                    st.warning("No se pudo calcular el tiempo de congelación. Revise los datos de entrada para esta sección.")


                # --- NUEVA SECCIÓN: Propiedades y Tiempo para Escaldado ---
                st.write("---")
                st.subheader("Propiedades Termofísicas y Tiempo para Escaldado")

                if temp_inicial_escaldado >= temp_final_escaldado:
                    st.warning("La Temperatura Final de Escaldado debe ser mayor que la Temperatura Inicial.")
                elif T_medio_escaldado <= temp_final_escaldado:
                    st.warning("La Temperatura del Medio de Escaldado debe ser mayor que la Temperatura Final deseada del alimento.")
                else:
                    # Usar la temperatura media entre la inicial del alimento y la final deseada en el centro para las propiedades
                    temperatura_media_escaldado = (temp_inicial_escaldado + temp_final_escaldado) / 2
                    st.write(f"**Temperatura Media de Escaldado para Propiedades:** {temperatura_media_escaldado:.2f}°C")

                    # Calcular propiedades a la temperatura media de escaldado
                    # Para escaldado, asumimos que no hay congelación, por lo que Tf_input no es relevante aquí
                    # y podemos usar un valor por encima de 0 para asegurar que no se calcule hielo.
                    densidad_escaldado = calcular_densidad_alimento(temperatura_media_escaldado, composicion, 0.0)
                    cp_escaldado = calcular_cp_alimento(temperatura_media_escaldado, composicion, 0.0)
                    k_escaldado = calcular_k_alimento(temperatura_media_escaldado, composicion, 0.0)
                    alpha_escaldado = calcular_alpha_alimento(temperatura_media_escaldado, composicion, 0.0)

                    st.metric(label="Densidad (ρ) Media", value=f"{densidad_escaldado:.2f} kg/m³")
                    st.metric(label="Calor Específico (Cp) Medio", value=f"{cp_escaldado:.2f} J/(kg·K)")
                    st.metric(label="Conductividad Térmica (k) Media", value=f"{k_escaldado:.4f} W/(m·K)")
                    st.metric(label="Difusividad Térmica (α) Media", value=f"{alpha_escaldado:.2e} m²/s")

                    # Calcular tiempo de escaldado
                    tiempo_escaldado_segundos = calcular_tiempo_escaldado(
                        temp_inicial_escaldado, temp_final_escaldado, T_medio_escaldado,
                        h_escaldado, k_escaldado, alpha_escaldado, geometria, dimension_a
                    )

                    if tiempo_escaldado_segundos is not None:
                        tiempo_escaldado_minutos = tiempo_escaldado_segundos / 60
                        st.metric(label="Tiempo de Escaldado (Centro)", value=f"{tiempo_escaldado_minutos:.2f} minutos")

                        # --- Gráfica del Perfil de Temperatura ---
                        st.write("---")
                        st.subheader("Perfil de Temperatura al Final del Escaldado")

                        posiciones, temperaturas = calcular_perfil_temperatura(
                            tiempo_escaldado_segundos, temp_inicial_escaldado, T_medio_escaldado,
                            alpha_escaldado, k_escaldado, h_escaldado, geometria, dimension_a
                        )

                        if posiciones is not None and temperaturas is not None:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            ax.plot(posiciones, temperaturas, marker='o', linestyle='-', markersize=4)
                            ax.set_xlabel("Posición Adimensional (x/L o r/R)")
                            ax.set_ylabel("Temperatura (°C)")
                            ax.set_title(f"Perfil de Temperatura en {geometria} (t = {tiempo_escaldado_minutos:.2f} min)")
                            ax.grid(True)
                            ax.set_ylim(min(temp_inicial_escaldado, T_medio_escaldado) - 5, max(temp_inicial_escaldado, T_medio_escaldado) + 5)
                            
                            # Marcar la temperatura final en el centro
                            ax.axhline(y=temp_final_escaldado, color='r', linestyle='--', label=f'T centro objetivo ({temp_final_escaldado}°C)')
                            ax.legend()
                            
                            st.pyplot(fig)
                            plt.close(fig) # Importante para liberar memoria

                        else:
                            st.warning("No se pudo generar el perfil de temperatura. Revise los parámetros de escaldado y asegúrese de que SciPy esté en requirements.txt para Cilindro/Esfera.")
                    else:
                        st.warning("No se pudo calcular el tiempo de escaldado. Revise los datos de entrada para esta sección.")

            except Exception as e:
                st.error(f"Ocurrió un error durante el cálculo: {e}")
                st.warning("Asegúrese de que los valores de entrada sean válidos y que la suma de la composición sea 100%.")
