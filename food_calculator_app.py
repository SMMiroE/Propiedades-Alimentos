import streamlit as st
import numpy as np

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


# --- 2. Funciones para calcular la PROPIEDAD DEL ALIMENTO COMPLETO ---

def calcular_densidad_alimento(t, composicion):
    """
    Calcula la densidad del alimento usando las ecuaciones de Choi y Okos.
    :param t: Temperatura en °C.
    :param composicion: Diccionario con porcentajes de los componentes.
                        Ej: {'agua': 70, 'proteina': 10, 'grasa': 5, 'carbohidrato': 10, 'fibra': 3, 'cenizas': 2}
    :return: Densidad del alimento en kg/m^3.
    """
    if abs(sum(composicion.values()) - 100) > 0.01: # Usar una pequeña tolerancia para la suma
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop() # Detiene la ejecución si la suma no es 100

    # Convertir porcentajes a fracciones de masa
    Xw = composicion.get('agua', 0) / 100
    Xp = composicion.get('proteina', 0) / 100
    Xf = composicion.get('grasa', 0) / 100
    Xc = composicion.get('carbohidrato', 0) / 100
    Xfi = composicion.get('fibra', 0) / 100
    Xa = composicion.get('cenizas', 0) / 100

    rho_w_val = densidad_agua(t)
    rho_p_val = densidad_proteina(t)
    rho_f_val = densidad_grasa(t)
    rho_c_val = densidad_carbohidrato(t)
    rho_fi_val = densidad_fibra(t)
    rho_a_val = densidad_cenizas(t)

    rho_alimento_inv = (Xw / rho_w_val) + \
                       (Xp / rho_p_val) + \
                       (Xf / rho_f_val) + \
                       (Xc / rho_c_val) + \
                       (Xfi / rho_fi_val) + \
                       (Xa / rho_a_val)
    return 1 / rho_alimento_inv


def calcular_cp_alimento(t, composicion):
    """
    Calcula el calor específico del alimento usando las ecuaciones de Choi y Okos.
    :param t: Temperatura en °C.
    :param composicion: Diccionario con porcentajes de los componentes.
    :return: Calor específico del alimento en J/(kg·K).
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw = composicion.get('agua', 0) / 100
    Xp = composicion.get('proteina', 0) / 100
    Xf = composicion.get('grasa', 0) / 100
    Xc = composicion.get('carbohidrato', 0) / 100
    Xfi = composicion.get('fibra', 0) / 100
    Xa = composicion.get('cenizas', 0) / 100

    cp_alimento = (Xw * cp_agua(t)) + \
                  (Xp * cp_proteina(t)) + \
                  (Xf * cp_grasa(t)) + \
                  (Xc * cp_carbohidrato(t)) + \
                  (Xfi * cp_fibra(t)) + \
                  (Xa * cp_cenizas(t))
    return cp_alimento


def calcular_k_alimento(t, composicion):
    """
    Calcula la conductividad térmica del alimento usando las ecuaciones de Choi y Okos.
    :param t: Temperatura en °C.
    :param composicion: Diccionario con porcentajes de los componentes.
    :return: Conductividad térmica del alimento en W/(m·K).
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw = composicion.get('agua', 0) / 100
    Xp = composicion.get('proteina', 0) / 100
    Xf = composicion.get('grasa', 0) / 100
    Xc = composicion.get('carbohidrato', 0) / 100
    Xfi = composicion.get('fibra', 0) / 100
    Xa = composicion.get('cenizas', 0) / 100

    k_alimento = (Xw * k_agua(t)) + \
                 (Xp * k_proteina(t)) + \
                 (Xf * k_grasa(t)) + \
                 (Xc * k_carbohidrato(t)) + \
                 (Xfi * k_fibra(t)) + \
                 (Xa * k_cenizas(t))
    return k_alimento


def calcular_alpha_alimento(t, composicion):
    """
    Calcula la difusividad térmica del alimento usando las ecuaciones de Choi y Okos.
    :param t: Temperatura en °C.
    :param composicion: Diccionario con porcentajes de los componentes.
    :return: Difusividad térmica del alimento en m^2/s.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw = composicion.get('agua', 0) / 100
    Xp = composicion.get('proteina', 0) / 100
    Xf = composicion.get('grasa', 0) / 100
    Xc = composicion.get('carbohidrato', 0) / 100
    Xfi = composicion.get('fibra', 0) / 100
    Xa = composicion.get('cenizas', 0) / 100

    alpha_alimento = (Xw * alpha_agua(t)) + \
                     (Xp * alpha_proteina(t)) + \
                     (Xf * alpha_grasa(t)) + \
                     (Xc * alpha_carbohidrato(t)) + \
                     (Xfi * alpha_fibra(t)) + \
                     (Xa * alpha_cenizas(t))
    return alpha_alimento


# --- CONFIGURACIÓN DE LA INTERFAZ CON STREAMLIT ---

st.set_page_config(
    page_title="Calculador de Propiedades de Alimentos",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🍔 Calculador de Propiedades Termofísicas de Alimentos 🌡️")
st.markdown("Calcula densidad, calor específico, conductividad y difusividad térmica usando las ecuaciones de Choi y Okos (1986).")

st.sidebar.header("Datos de Entrada")

# Entrada de Temperatura
temperatura = st.sidebar.slider("Temperatura (°C)", -40.0, 150.0, 25.0, step=0.1)

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

# Botón de cálculo
if st.sidebar.button("Calcular Propiedades"):
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
                densidad = calcular_densidad_alimento(temperatura, composicion)
                cp = calcular_cp_alimento(temperatura, composicion)
                k = calcular_k_alimento(temperatura, composicion)
                alpha = calcular_alpha_alimento(temperatura, composicion)

                st.subheader("Resultados Calculados")
                st.write(f"**Temperatura:** {temperatura}°C")
                st.write("---")
                st.metric(label="Densidad (ρ)", value=f"{densidad:.2f} kg/m³")
                st.metric(label="Calor Específico (Cp)", value=f"{cp:.2f} J/(kg·K)")
                st.metric(label="Conductividad Térmica (k)", value=f"{k:.4f} W/(m·K)")
                st.metric(label="Difusividad Térmica (α)", value=f"{alpha:.2e} m²/s")

            except Exception as e:
                st.error(f"Ocurrió un error durante el cálculo: {e}")
                st.warning("Asegúrese de que los valores de entrada sean válidos para las ecuaciones de Choi y Okos.")
