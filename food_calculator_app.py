import streamlit as st
import numpy as np
import matplotlib.pyplot as plt # Reincorporado para la gráfica

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
