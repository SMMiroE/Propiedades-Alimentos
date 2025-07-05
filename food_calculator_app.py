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

def calcular_densidad_alimento(t, composicion, Tf_input=0.0): # Tf_input con valor por defecto
    """
    Calcula la densidad del alimento usando las ecuaciones de Choi y Okos.
    Si la temperatura es de congelación, considera la fracción de hielo.
    Tf_input es la temperatura de inicio de congelación. Si t > Tf_input, no hay hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01: # Usar una pequeña tolerancia para la suma
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100
    
    # Solo calcular fracción de hielo si la temperatura está en el rango de congelación
    if t < Tf_input:
        Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0), Tf_input)
        Xu = Xw_inicial - Xi
    else:
        Xi = 0.0
        Xu = Xw_inicial

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


def calcular_cp_alimento(t, composicion, Tf_input=0.0): # Tf_input con valor por defecto
    """
    Calcula el calor específico del alimento usando las ecuaciones de Choi y Okos.
    Si la temperatura es de congelación, considera la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100
    
    # Solo calcular fracción de hielo si la temperatura está en el rango de congelación
    if t < Tf_input:
        Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0), Tf_input)
        Xu = Xw_inicial - Xi
    else:
        Xi = 0.0
        Xu = Xw_inicial

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


def calcular_k_alimento(t, composicion, Tf_input=0.0): # Tf_input con valor por defecto
    """
    Calcula la conductividad térmica del alimento usando las ecuaciones de Choi y Okos.
    Si la temperatura es de congelación, considera la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100
    
    # Solo calcular fracción de hielo si la temperatura está en el rango de congelación
    if t < Tf_input:
        Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0), Tf_input)
        Xu = Xw_inicial - Xi
    else:
        Xi = 0.0
        Xu = Xw_inicial

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


def calcular_alpha_alimento(t, composicion, Tf_input=0.0): # Tf_input con valor por defecto
    """
    Calcula la difusividad térmica del alimento usando las ecuaciones de Choi y Okos.
    Si la temperatura es de congelación, considera la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    # Recalcula las propiedades auxiliares (rho, Cp, k) ya que la difusividad depende de ellas.
    # Estas funciones internas ya manejan la lógica de Tf_input.
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
    :return: Tiempo de congelación en segundos.
    """
    # Constantes
    L0 = 333.6 * 1000 # Calor latente de fusión del hielo a 0°C en J/kg

    # Validaciones
    if Ta >= Tf_input:
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
    temp_prop_congelado = max(Ta, Tf_input - 5)
    rho_f = calcular_densidad_alimento(temp_prop_congelado, composicion, Tf_input)
    k_f = calcular_k_alimento(temp_prop_congelado, composicion, Tf_input)

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
    if (Tf_input - Ta) == 0:
        return float('inf')

    tiempo_segundos = (L_efectivo / (Tf_input - Ta)) * ((P * dimension_a / h) + (R * dimension_a**2 / k_f))

    return tiempo_segundos # Ahora retorna segundos, la conversión a minutos se hará en la UI


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
    :param h_escaldado: Coeficiente de transferencia de calor por convección para escaldado (W/(m·K)).
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
        
    elif geometria == 'Cilindro':
        try:
            # j0 es la función de Bessel de primera clase de orden cero
            term_posicion = j0(lambda1 * posiciones_adimensionales) / j0(lambda1)
        except Exception as e:
            st.error(f"Error al calcular perfil para Cilindro (J0): {e}. Asegúrese de que 'scipy' esté en requirements.txt.")
            return None, None

    elif geometria == 'Esfera':
        try:
            # La expresión para Esfera es (sin(lambda*pos))/(lambda*pos) / (sin(lambda)/lambda)
            # Primero calculamos el numerador (sin(lambda*r/R) / (lambda*r/R))
            term_numerador = np.where(lambda1 * posiciones_adimensionales == 0, 1.0, np.sin(lambda1 * posiciones_adimensionales) / (lambda1 * posiciones_adimensionales))
            
            # Luego el denominador (sin(lambda)/lambda)
            term_denominador = np.where(lambda1 == 0, 1.0, np.sin(lambda1) / lambda1)

            if term_denominador == 0: # Evitar división por cero en la normalización
                st.warning("El término del denominador para Esfera es cero. No se puede calcular el perfil. Revise los valores de lambda1.")
                return None, None
            
            term_posicion = term_numerador / term_denominador

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

def calcular_temperatura_centro_vs_tiempo(T_inicial_alimento, T_medio_escaldado, alpha_alimento_medio, k_alimento_medio, h_escaldado, geometria, dimension_a, max_tiempo_segundos, num_puntos_tiempo=100):
    """
    Calcula la temperatura del centro del alimento a lo largo del tiempo.
    :return: Tupla de (tiempos_segundos, temperaturas_centro).
    """
    Lc = dimension_a
    if k_alimento_medio == 0:
        st.error("Conductividad térmica es cero, no se puede calcular el Número de Biot para el perfil de tiempo.")
        return None, None
    Bi = (h_escaldado * Lc) / k_alimento_medio

    A1, lambda1 = get_heisler_coeffs(Bi, geometria)

    if A1 is None or lambda1 is None:
        st.error(f"No se pudieron obtener coeficientes A1 y lambda1 para el perfil de tiempo. Bi={Bi:.2f}, Geometría={geometria}")
        return None, None
    
    if alpha_alimento_medio == 0 or Lc == 0:
        st.error("Difusividad térmica o longitud característica son cero, no se puede calcular el Número de Fourier para el perfil de tiempo.")
        return None, None

    tiempos_segundos = np.linspace(0, max_tiempo_segundos, num_puntos_tiempo)
    temperaturas_centro = []

    for t_sec in tiempos_segundos:
        Fo = (alpha_alimento_medio * t_sec) / Lc**2
        
        # Theta_0 = (T_0 - T_inf) / (T_i - T_inf)
        theta_0_t = A1 * np.exp(-lambda1**2 * Fo)
        
        T_center = T_medio_escaldado + theta_0_t * (T_inicial_alimento - T_medio_escaldado)
        temperaturas_centro.append(T_center)

    return tiempos_segundos, np.array(temperaturas_centro)


# --- CONFIGURACIÓN DE LA INTERFAZ CON STREAMLIT ---

st.set_page_config(
    page_title="Herramienta de Simulación de Procesos Térmicos en Alimentos",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Contenido principal de la aplicación ---

# Título principal de la aplicación, centrado
st.markdown("<h3 style='text-align: center;'>PROCESOS TÉRMICOS EN ALIMENTOS</h3>", unsafe_allow_html=True) # h3 para reducir tamaño
st.markdown("<p style='text-align: center; font-size: 1em;'>Herramienta de simulación</p>", unsafe_allow_html=True) # p con font-size reducido

st.markdown("""
<p style='font-size: 0.85em;'>Esta aplicación interactiva permite calcular <b>propiedades termofísicas de alimentos</b> (densidad, calor específico, conductividad y difusividad térmica) basadas en su composición proximal, utilizando las ecuaciones de <b>Choi y Okos (1986)</b>. Además, facilita la estimación del <b>tiempo de congelación</b> mediante la ecuación de Plank y la simulación de procesos de <b>escaldado</b>, incluyendo el cálculo del tiempo necesario y la visualización del <b>perfil de temperatura</b> dentro del alimento, utilizando la solución del primer término de la serie de Fourier.</p>
""", unsafe_allow_html=True)

st.markdown("---") # Separador visual

# --- Sección de Composición Proximal (en la pantalla principal) ---
st.markdown("<h5 style='text-align: left;'>Introduce la composición del alimento (%)</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
col1, col2 = st.columns(2) # Usamos columnas para una mejor organización de inputs
with col1:
    agua = st.number_input("Agua (%)", min_value=0.0, max_value=100.0, value=75.0, step=0.1, key="agua_main")
    proteina = st.number_input("Proteína (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1, key="proteina_main")
    grasa = st.number_input("Grasa (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1, key="grasa_main")
with col2:
    carbohidrato = st.number_input("Carbohidratos (%)", min_value=0.0, max_value=100.0, value=4.0, step=0.1, key="carbo_main")
    fibra = st.number_input("Fibra (%)", min_value=0.0, max_value=100.0, value=0.5, step=0.1, key="fibra_main")
    cenizas = st.number_input("Cenizas (%)", min_value=0.0, max_value=100.0, value=0.5, step=0.1, key="cenizas_main")

composicion_total = agua + proteina + grasa + carbohidrato + fibra + cenizas
if abs(composicion_total - 100) > 0.01:
    st.error("La suma de los porcentajes debe ser 100%. Por favor, ajuste la composición.")
else:
    st.success("La suma de la composición es 100%. ¡Perfecto!")

st.markdown("---") # Separador visual

# --- Sección de Selección de Cálculo (en la pantalla principal) ---
st.markdown("<h5 style='text-align: left;'>Elige el cálculo que quieras realizar:</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
opcion_calculo = st.radio(
    " ", # Un espacio para que el label no sea visible pero el radio button funcione
    (
        "Propiedades a T > 0°C",
        "Propiedades a T < 0°C",
        "Tiempo de escaldado (min)",
        "Tiempo de congelación (min)"
    ),
    key="main_opcion_calculo" # Key único para este radio button
)

st.markdown("---") # Separador visual

# --- Contenedores para entradas dinámicas (AHORA EN LA COLUMNA PRINCIPAL) ---
# Default values for inputs that might not be shown, initialized here for scope
temperatura_calculo = 25.0
Tf_input = -1.8 # Initial freezing point
geometria = 'Placa'
dimension_a = 0.05
temp_inicial_escaldado = 20.0
temp_final_escaldado = 85.0
T_medio_escaldado = 95.0
h_escaldado = 100.0
T0_congelacion = 20.0
Ta_congelacion = -20.0
h_congelacion = 15.0

# **IMPORTANTE:** Inicializar temp_medio_escaldado aquí también
# para que siempre esté definida antes de cualquier uso posterior
temp_media_escaldado = 0.0 # Valor predeterminado que se actualizará si se elige escaldado


# Mostrar los campos de entrada relevantes en la sección principal
# Modificación de los títulos de las secciones de parámetros
if opcion_calculo == "Propiedades a T > 0°C":
    st.markdown("<h5 style='text-align: left;'>Parámetros para el cálculo:</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
elif opcion_calculo == "Propiedades a T < 0°C":
    st.markdown("<h5 style='text-align: left;'>Parámetros para el cálculo:</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
elif opcion_calculo == "Tiempo de escaldado (min)":
    st.markdown("<h5 style='text-align: left;'>Parámetros para el cálculo:</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
elif opcion_calculo == "Tiempo de congelación (min)":
    st.markdown("<h5 style='text-align: left;'>Parámetros para el cálculo:</h5>", unsafe_allow_html=True) # h5 para reducir tamaño


if opcion_calculo == "Propiedades a T > 0°C":
    temperatura_calculo = st.number_input(
        "Temperatura (°C)",
        min_value=0.0, max_value=150.0, value=25.0, step=0.1,
        help="Temperatura a la que se calcularán las propiedades termofísicas. Solo para temperaturas por encima de la congelación."
    )

elif opcion_calculo == "Propiedades a T < 0°C":
    temperatura_calculo = st.number_input(
        "Temperatura (°C)", # Texto cambiado de "Temperatura de Cálculo" a "Temperatura"
        min_value=-50.0, max_value=0.0, value=-5.0, step=0.1,
        help="Temperatura a la que se calcularán las propiedades termofísicas, incluyendo la formación de hielo."
    )
    Tf_input = st.number_input(
        "Temperatura Inicial de Congelación (Tf) [°C]",
        min_value=-50.0, max_value=0.0, value=-1.8, step=0.1,
        help="Temperatura a la que el agua en el alimento comienza a congelarse."
    )
    if temperatura_calculo >= Tf_input:
        st.warning(f"La temperatura de cálculo ({temperatura_calculo}°C) debe ser menor que la temperatura inicial de congelación ({Tf_input}°C) para observar la formación de hielo.")


elif opcion_calculo == "Tiempo de escaldado (min)":
    temp_inicial_escaldado = st.number_input("Temperatura Inicial del Alimento (°C)", min_value=0.0, max_value=100.0, value=20.0, step=0.1) # Texto cambiado
    temp_final_escaldado = st.number_input("Temperatura Final en el Centro del Alimento (°C)", min_value=0.0, max_value=100.0, value=85.0, step=0.1) # Texto cambiado
    T_medio_escaldado = st.number_input("Temperatura del Medio Calefactor (°C)", min_value=0.0, max_value=150.0, value=95.0, step=0.1) # Texto cambiado
    h_escaldado = st.number_input("Coeficiente de Convección (h) [W/(m²·K)]", min_value=1.0, max_value=5000.0, value=100.0, step=0.1)
    
    geometria = st.selectbox("Geometría del Alimento", ['Placa', 'Cilindro', 'Esfera'], key="geom_escaldado_main")
    dimension_a = st.number_input("Dimensión Característica 'a' (m)", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f", key="dim_escaldado_main")

elif opcion_calculo == "Tiempo de congelación (min)":
    Tf_input = st.number_input("Temperatura Inicial de Congelación (Tf) [°C]", min_value=-50.0, max_value=0.0, value=-1.8, step=0.1)
    T0_congelacion = st.number_input("Temperatura Inicial del Alimento (°C)", min_value=-40.0, max_value=150.0, value=20.0, step=0.1)
    Ta_congelacion = st.number_input("Temperatura del Medio Refrigerante (°C)", min_value=-60.0, max_value=0.0, value=-20.0, step=0.1) # Texto cambiado
    h_congelacion = st.number_input("Coeficiente de Convección (h) [W/(m²·K)]", min_value=1.0, max_value=1000.0, value=15.0, step=0.1)
    
    geometria = st.selectbox("Geometría del Alimento", ['Placa', 'Cilindro', 'Esfera'], key="geom_congelacion_main")
    dimension_a = st.number_input("Dimensión Característica 'a' (m)", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f", key="dim_congelacion_main")


st.markdown("---") # Separador visual

# --- Botón de cálculo (en la pantalla principal) ---
if st.button("Realizar Cálculo"):
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
                if opcion_calculo == "Propiedades a T > 0°C":
                    st.markdown("<h5 style='text-align: left;'>Propiedades Termofísicas Calculadas</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
                    st.markdown("""
                    <small>Estas propiedades se calculan asumiendo que el agua se encuentra en estado líquido y son aproximadas a las propiedades del alimento de composición homogénea.</small>
                    """, unsafe_allow_html=True) # Leyenda movida y texto modificado
                    st.write(f"**Temperatura de Cálculo:** {temperatura_calculo}°C")
                    st.markdown("---")
                    
                    densidad = calcular_densidad_alimento(temperatura_calculo, composicion, 0.0) 
                    cp = calcular_cp_alimento(temperatura_calculo, composicion, 0.0)
                    k = calcular_k_alimento(temperatura_calculo, composicion, 0.0)
                    alpha = calcular_alpha_alimento(temperatura_calculo, composicion, 0.0)

                    # Formato para la difusividad térmica (VOLVER A e-X)
                    alpha_str = f"{alpha:.2e}"
                    
                    # Usar st.metric con un estilo para el valor para reducir el tamaño
                    st.metric(label="Densidad (ρ)", value=f"{densidad:.2f} kg/m³")
                    st.metric(label="Calor Específico (Cp)", value=f"{cp:.2f} J/(kg·K)")
                    st.metric(label="Conductividad Térmica (k)", value=f"{k:.4f} W/(m·K)")
                    st.metric(label="Difusividad Térmica (α)", value=f"{alpha_str} m²/s") # Notación cambiada

                elif opcion_calculo == "Propiedades a T < 0°C":
                    st.markdown("<h5 style='text-align: left;'>Resultados de Propiedades Termofísicas (Con Hielo)</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
                    st.markdown("""
                    <small>Estas propiedades se calculan considerando la fracción de hielo presente a la temperatura especificada, basándose en la temperatura inicial de congelación (Tf).</small>
                    """, unsafe_allow_html=True) # Leyenda movida
                    st.write(f"**Temperatura de Cálculo:** {temperatura_calculo}°C")
                    st.write(f"**Temperatura Inicial de Congelación (Tf):** {Tf_input}°C")
                    st.markdown("---")

                    if temperatura_calculo >= Tf_input:
                        st.warning("La temperatura de cálculo debe ser menor que la temperatura inicial de congelación (Tf) para que se forme hielo. Ajuste los parámetros.")
                    else:
                        densidad = calcular_densidad_alimento(temperatura_calculo, composicion, Tf_input)
                        cp = calcular_cp_alimento(temperatura_calculo, composicion, Tf_input)
                        k = calcular_k_alimento(temperatura_calculo, composicion, Tf_input)
                        alpha = calcular_alpha_alimento(temperatura_calculo, composicion, Tf_input)
                        
                        Xi_fraccion = calcular_fraccion_hielo(temperatura_calculo, composicion.get('agua', 0), Tf_input)
                        
                        # Formato para la difusividad térmica (VOLVER A e-X)
                        alpha_str = f"{alpha:.2e}"

                        st.metric(label="Fracción de Hielo (Xi)", value=f"{Xi_fraccion:.3f} (kg hielo/kg alimento)")
                        st.metric(label="Densidad (ρ)", value=f"{densidad:.2f} kg/m³")
                        st.metric(label="Calor Específico (Cp)", value=f"{cp:.2f} J/(kg·K)")
                        st.metric(label="Conductividad Térmica (k)", value=f"{k:.4f} W/(m·K)")
                        st.metric(label="Difusividad Térmica (α)", value=f"{alpha_str} m²/s") # Notación cambiada


                elif opcion_calculo == "Tiempo de escaldado (min)": # Opción renombrada
                    st.markdown("<h5 style='text-align: left;'>Tiempo de Escaldado Calculado</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
                    # Eliminadas las propiedades termofísicas medias de esta sección

                    if temp_inicial_escaldado >= temp_final_escaldado:
                        st.warning("La Temperatura Final deseada en el centro del alimento debe ser mayor que la Temperatura Inicial del alimento.")
                    elif T_medio_escaldado <= temp_final_escaldado:
                        st.warning("La Temperatura del Medio de Escaldado debe ser estrictamente mayor que la Temperatura Final deseada en el centro del alimento.")
                    else:
                        # La variable 'temp_media_escaldado' ya está definida globalmente
                        # y se actualiza aquí, luego se usa para calcular las propiedades.
                        temp_media_escaldado = (temp_inicial_escaldado + T_medio_escaldado) / 2                                                                                                    
                        
                        # No mostrar las propiedades medias aquí, pero se calculan para el tiempo
                        densidad_escaldado = calcular_densidad_alimento(temp_media_escaldado, composicion, 0.0)
                        cp_escaldado = calcular_cp_alimento(temp_media_escaldado, composicion, 0.0)
                        k_escaldado = calcular_k_alimento(temp_media_escaldado, composicion, 0.0)
                        alpha_escaldado = calcular_alpha_alimento(temp_media_escaldado, composicion, 0.0)

                        tiempo_escaldado_segundos = calcular_tiempo_escaldado(
                            temp_inicial_escaldado, temp_final_escaldado, T_medio_escaldado,
                            h_escaldado, k_escaldado, alpha_escaldado, geometria, dimension_a
                        )

                        if tiempo_escaldado_segundos is not None:
                            tiempo_escaldado_minutos = tiempo_escaldado_segundos / 60
                            st.metric(label="Tiempo de Escaldado (Centro)", value=f"{tiempo_escaldado_minutos:.2f} minutos")

                            st.write("---")
                            st.markdown("<h5 style='text-align: left;'>Perfil de Temperatura al Final del Escaldado</h5>", unsafe_allow_html=True) # h5 para reducir tamaño

                            posiciones, temperaturas = calcular_perfil_temperatura(
                                tiempo_escaldado_segundos, temp_inicial_escaldado, T_medio_escaldado,
                                alpha_escaldado, k_escaldado, h_escaldado, geometria, dimension_a
                            )

                            if posiciones is not None and temperaturas is not None:
                                fig, ax = plt.subplots(figsize=(8, 5))
                                ax.plot(posiciones * dimension_a * 100, temperaturas, marker='o', linestyle='-', markersize=4) # Convertir a cm para el eje x
                                ax.set_xlabel(f"Posición desde el centro (cm) para {geometria}")
                                ax.set_ylabel("Temperatura (°C)")
                                ax.set_title(f"Perfil de Temperatura en {geometria} (t = {tiempo_escaldado_minutos:.2f} min)")
                                ax.grid(True)
                                min_temp_plot = min(temp_inicial_escaldado, T_medio_escaldado, np.min(temperaturas)) - 5
                                max_temp_plot = max(temp_inicial_escaldado, T_medio_escaldado, np.max(temperaturas)) + 5
                                ax.set_ylim(min_temp_plot, max_temp_plot)
                                
                                ax.axhline(y=temp_final_escaldado, color='r', linestyle='--', label=f'T centro ({temp_final_escaldado}°C)') # Texto cambiado
                                ax.legend()
                                
                                st.pyplot(fig)
                                plt.close(fig)

                            else:
                                st.warning("No se pudo generar el perfil de temperatura. Revise los parámetros de escaldado y asegúrese de que SciPy esté en requirements.txt para Cilindro/Esfera.")

                            # Nuevo gráfico: Temperatura del centro vs tiempo
                            st.markdown("<h5 style='text-align: left;'>Temperatura del Centro vs. Tiempo</h5>", unsafe_allow_html=True) # h5 para reducir tamaño
                            
                            # MODIFICACIÓN CLAVE AQUÍ:
                            # Se usa el tiempo_escaldado_segundos calculado como el tiempo máximo para la gráfica.
                            max_plot_time = tiempo_escaldado_segundos 
                            
                            tiempos_plot, temps_center_plot = calcular_temperatura_centro_vs_tiempo(
                                temp_inicial_escaldado, T_medio_escaldado, alpha_escaldado, k_escaldado, h_escaldado,
                                geometria, dimension_a, max_tiempo_segundos=max_plot_time
                            )

                            if tiempos_plot is not None and temps_center_plot is not None:
                                fig_time, ax_time = plt.subplots(figsize=(8, 5))
                                ax_time.plot(tiempos_plot / 60, temps_center_plot, label='Temperatura del Centro') # Tiempo en minutos
                                ax_time.axhline(y=T_medio_escaldado, color='g', linestyle=':', label='Temperatura del Medio')
                                ax_time.axhline(y=temp_final_escaldado, color='r', linestyle='--', label=f'T centro objetivo ({temp_final_escaldado}°C)')
                                ax_time.set_xlabel("Tiempo (min)")
                                ax_time.set_ylabel("Temperatura (°C)")
                                ax_time.set_title("Temperatura del Centro del Alimento a lo largo del Tiempo")
                                ax_time.grid(True)
                                ax_time.legend()
                                st.pyplot(fig_time)
                                plt.close(fig_time)
                            else:
                                st.warning("No se pudo generar el gráfico de Temperatura del centro vs. Tiempo.")

                        else:
                            st.warning("No se pudo calcular el tiempo de escaldado. Revise los datos de entrada para esta sección.")
                
                elif opcion_calculo == "Tiempo de congelación (min)": # Opción renombrada
                    st.markdown("<h5 style='text-align: left;'>Tiempo de Congelación (Ecuación de Plank)</h5>", unsafe_allow_html=True) # h5 para reducir tamaño

                    tiempo_congelacion_segundos = calcular_tiempo_congelacion(composicion, T0_congelacion, Ta_congelacion, h_congelacion, geometria, dimension_a, Tf_input)

                    if tiempo_congelacion_segundos is not None:
                        tiempo_congelacion_minutos = tiempo_congelacion_segundos / 60 # Convertir a minutos
                        st.metric(label="Tiempo de Congelación", value=f"{tiempo_congelacion_minutos:.2f} minutos") # Unidad cambiada
                    else:
                        st.warning("No se pudo calcular el tiempo de congelación. Revise los datos de entrada para esta sección.")

            except Exception as e:
                st.error(f"Ocurrió un error durante el cálculo: {e}")
                st.warning("Asegúrese de que los valores de entrada sean válidos y que la suma de la composición sea 100%.")

# --- Contenido de la barra lateral (SIMPLIFICADA) ---
st.sidebar.header("Información de la Aplicación")
st.sidebar.markdown("""
Para más detalles sobre la aplicación, su uso y referencias, consulte las pestañas en la sección "Información Adicional" de la pantalla principal.
""", unsafe_allow_html=True)

# **AGREGAMOS TU INFORMACIÓN DE CONTACTO AQUÍ, CON UN ESTILO MÁS PEQUEÑO**
st.sidebar.markdown("<p style='font-size: 0.8em;'>Dra. Silvia Marcela Miró Erdmann</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-size: 0.8em;'>✉️ smmiroer@gmail.com</p>", unsafe_allow_html=True)

# --- Sección de Información Adicional (EN LA PANTALLA PRINCIPAL CON PESTAÑAS) ---
st.markdown("---") # Separador visual
st.markdown("<h5 style='text-align: left;'>Información Adicional</h5>", unsafe_allow_html=True) # h5 para reducir tamaño

# Usar st.tabs para organizar el contenido
tab1, tab2, tab3 = st.tabs(["Guía Rápida de Uso", "Referencias Bibliográficas", "Bases de Datos de Composición de Alimentos"])

with tab1:
    st.markdown("<h6 style='text-align: left;'>Guía Rápida de Uso</h6>", unsafe_allow_html=True) # h6 para reducir tamaño
    st.markdown("""
    <p style='font-size: 0.8em;'>Para utilizar esta herramienta de simulación de procesos térmicos, sigue estos sencillos pasos:</p>

    <p style='font-size: 0.8em;'>1.  <b>Define la Composición Proximal:</b></p>
        <ul style='font-size: 0.8em;'>
            <li>En la sección "Introduce la composición del alimento" de la pantalla principal, ingresa los porcentajes de <b>Agua, Proteína, Grasa, Carbohidratos, Fibra</b> y <b>Cenizas</b> de tu alimento.</li>
            <li>Asegúrate de que la suma total sea <b>100%</b>. La aplicación te indicará si necesitas ajustar los valores.</li>
        </ul>

    <p style='font-size: 0.8em;'>2.  <b>Selecciona el Tipo de Cálculo:</b></p>
        <ul style='font-size: 0.8em;'>
            <li>En la sección "Elige el cálculo que quieras realizar" de la pantalla principal, usa las opciones de radio button para elegir la simulación que deseas.</li>
        </ul>

    <p style='font-size: 0.8em;'>3.  <b>Ingresa los Parámetros Específicos:</b></p>
        <ul style='font-size: 0.8em;'>
            <li>Debajo de la selección de cálculo, aparecerán los campos de entrada relevantes para tu simulación (temperaturas, coeficientes, geometría, etc.). Completa todos los datos necesarios.</li>
        </ul>

    <p style='font-size: 0.8em;'>4.  <b>Realiza el Cálculo:</b></p>
        <ul style='font-size: 0.8em;'>
            <li>Haz clic en el botón <b>"Realizar Cálculo"</b> en la parte inferior de la pantalla principal.</li>
            <li>Los resultados se mostrarán en la sección principal, junto con gráficas si aplica (para escaldado).</li>
        </ul>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("<h6 style='text-align: left;'>Referencias Bibliográficas</h6>", unsafe_allow_html=True) # h6 para reducir tamaño
    st.markdown("""
    <ul style='font-size: 0.8em;'>
        <li><b>Choi, Y., & Okos, M. R. (1986).</b> <i>Thermal Properties of Foods</i>. In M. R. Okos (Ed.), Physical Properties of Food Materials (pp. 93-112). Purdue University.</li>
        <li><b>Singh, R. P., & Heldman, D. D. (2009).</b> <i>Introducción a la Ingeniería de los Alimentos</i> (2da ed.). Acribia.</li>
        <li><b>Incropera, F. P., DeWitt, D. P., Bergman, T. L., & Lavine, A. S. (2007).</b> <i>Fundamentals of Heat and Mass Transfer</i> (6th ed.). John Wiley & Sons.</li>
    </ul>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("<h6 style='text-align: left;'>Bases de Datos de Composición de Alimentos</h6>", unsafe_allow_html=True) # h6 para reducir tamaño
    st.markdown("""
    <p style='font-size: 0.8em;'>Aquí puedes encontrar enlaces a bases de datos confiables para consultar la composición proximal de diversos alimentos:</p>

    <ul style='font-size: 0.8em;'>
        <li><b>USDA FoodData Central (Estados Unidos):</b><br>[https://fdc.nal.usda.gov/](https://fdc.nal.usda.gov/)</li>
        <li><b>BEDCA - Base de Datos Española de Composición de Alimentos (España):</b><br>[http://www.bedca.net/](http://www.bedca.net/)</li>
        <li><b>Tabla de Composición de Alimentos del INTA (Argentina):</b><br>[https://inta.gob.ar/documentos/tablas-de-composicion-de-alimentos](https://inta.gob.ar/documentos/tablas-de-composicion-de-alimentos)</li>
        <li><b>FAO/INFOODS (Internacional):</b><br>[https://www.fao.org/infoods/infoods/es/](https://www.fao.org/infoods/infoods/es/)</li>
        <li><b>Food Composition Databases (EUFIC - Europa):</b><br>[https://www.eufic.org/en/food-composition/article/food-composition-databases](https://www.eufic.org/en/food-composition/article/food-composition-databases)</li>
    </ul>
    """, unsafe_allow_html=True)
