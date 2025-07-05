
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

# --- Función para calcular la fracción de hielo ---
def calcular_fraccion_hielo(t, agua_porcentaje):
    """
    Calcula la fracción de hielo (Xi) en un alimento a una temperatura t (°C)
    dada la temperatura inicial de congelación (-1.8 °C por defecto para muchos alimentos)
    y el porcentaje de agua inicial.
    Asume una temperatura de congelación inicial de -1.8 °C y calor latente de fusión del hielo de 333.6 kJ/kg.
    """
    Tf = -1.8 # Temperatura inicial de congelación en °C (valor típico para muchos alimentos)
    L0 = 333.6 * 1000 # Calor latente de fusión del hielo a 0°C en J/kg (333.6 kJ/kg)

    if t >= Tf:
        return 0.0 # No hay hielo si la temperatura es mayor o igual a la de congelación inicial
    elif t < Tf:
        # Ecuación simplificada para fracción de hielo (asumiendo propiedades de solución diluida)
        # Esto es una aproximación y puede variar según el modelo de congelación
        Xi = (L0 / (4.186 * (Tf - t))) * (agua_porcentaje / 100)
        return min(max(0.0, Xi), agua_porcentaje / 100) # Asegura que esté entre 0 y el contenido total de agua
    else:
        return 0.0 # Caso por defecto, aunque el 'elif' anterior debería cubrirlo


# --- 2. Funciones para calcular la PROPIEDAD DEL ALIMENTO COMPLETO ---

def calcular_densidad_alimento(t, composicion):
    """
    Calcula la densidad del alimento usando las ecuaciones de Choi y Okos,
    considerando la fracción de hielo si la temperatura es de congelación.
    """
    if abs(sum(composicion.values()) - 100) > 0.01: # Usar una pequeña tolerancia para la suma
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100 # Fracción de agua inicial
    Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0)) # Fracción de hielo
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


def calcular_cp_alimento(t, composicion):
    """
    Calcula el calor específico del alimento usando las ecuaciones de Choi y Okos,
    considerando la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100
    Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0))
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


def calcular_k_alimento(t, composicion):
    """
    Calcula la conductividad térmica del alimento usando las ecuaciones de Choi y Okos,
    considerando la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    Xw_inicial = composicion.get('agua', 0) / 100
    Xi = calcular_fraccion_hielo(t, composicion.get('agua', 0))
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


def calcular_alpha_alimento(t, composicion):
    """
    Calcula la difusividad térmica del alimento usando las ecuaciones de Choi y Okos,
    considerando la fracción de hielo.
    """
    if abs(sum(composicion.values()) - 100) > 0.01:
        st.error("La suma de los porcentajes de los componentes debe ser 100%. Por favor, verifique.")
        st.stop()

    # Recalcula las propiedades auxiliares (rho, Cp, k) ya que la difusividad depende de ellas
    # y deben considerar la fracción de hielo.
    densidad = calcular_densidad_alimento(t, composicion)
    cp = calcular_cp_alimento(t, composicion)
    k = calcular_k_alimento(t, composicion)

    if densidad * cp == 0: # Evitar división por cero
        return 0.0
    return k / (densidad * cp)

# --- Función para calcular el tiempo de congelación (Ecuación de Plank) ---
def calcular_tiempo_congelacion(composicion, T0, Ta, h, geometria, dimension_a):
    """
    Calcula el tiempo de congelación usando la Ecuación de Plank.
    :param composicion: Diccionario con porcentajes de los componentes.
    :param T0: Temperatura inicial del alimento (°C).
    :param Ta: Temperatura del medio ambiente de congelación (°C).
    :param h: Coeficiente de transferencia de calor por convección (W/(m²·K)).
    :param geometria: Tipo de geometría ('Placa', 'Cilindro', 'Esfera').
    :param dimension_a: Dimensión característica del alimento (m).
    :return: Tiempo de congelación en horas.
    """
    # Constantes
    Tf = -1.8 # Temperatura de congelación inicial en °C (valor típico)
    L0 = 333.6 * 1000 # Calor latente de fusión del hielo a 0°C en J/kg

    # Validaciones
    if Ta >= Tf:
        st.warning("La temperatura del medio ambiente de congelación (Ta) debe ser menor que la temperatura de congelación inicial del alimento (Tf, aprox. -1.8°C) para que ocurra la congelación.")
        return None
    if h <= 0:
        st.warning("El coeficiente de transferencia de calor (h) debe ser un valor positivo.")
        return None
    if dimension_a <= 0:
        st.warning("La dimensión característica (a) debe ser un valor positivo.")
        return None

    # Calcular propiedades promedio del alimento congelado (a una T de referencia)
    # Usaremos una temperatura ligeramente por debajo de Tf para asegurar propiedades de hielo.
    temp_prop_congelado = max(Ta, Tf - 5)
    rho_f = calcular_densidad_alimento(temp_prop_congelado, composicion)
    k_f = calcular_k_alimento(temp_prop_congelado, composicion)

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
    if (Tf - Ta) == 0:
        return float('inf')

    tiempo_segundos = (L_efectivo / (Tf - Ta)) * ((P * dimension_a / h) + (R * dimension_a**2 / k_f))

    return tiempo_segundos / 3600 # Convertir segundos a horas


# --- CONFIGURACIÓN DE LA INTERFAZ CON STREAMLIT ---

st.set_page_config(
    page_title="Calculador de Propiedades de Alimentos",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🍔 Calculador de Propiedades Termofísicas de Alimentos 🌡️")
st.markdown("Dra. Silvia Marcela Miró Erdmann - Profesor Adjunto UNSL/ UNViMe") # Tu nombre y afiliación
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

T0 = st.sidebar.number_input("Temperatura Inicial del Alimento (°C)", min_value=-40.0, max_value=150.0, value=20.0, step=0.1,
                             help="Temperatura del alimento antes de iniciar la congelación.")
Ta = st.sidebar.number_input("Temperatura del Medio de Congelación (°C)", min_value=-60.0, max_value=0.0, value=-20.0, step=0.1,
                             help="Temperatura del aire o medio refrigerante.")
h = st.sidebar.number_input("Coeficiente de Convección (h) [W/(m²·K)]", min_value=1.0, max_value=1000.0, value=15.0, step=0.1,
                            help="Coeficiente de transferencia de calor por convección en la superficie del alimento.")

geometria = st.sidebar.selectbox("Geometría del Alimento", ['Placa', 'Cilindro', 'Esfera'],
                                 help="Selecciona la forma geométrica del alimento.")

dimension_a = st.sidebar.number_input("Dimensión Característica 'a' (m)", min_value=0.001, max_value=1.0, value=0.05, step=0.001, format="%.3f",
                                      help="Para Placa: mitad del espesor; para Cilindro/Esfera: radio.")


# Botón de cálculo
if st.sidebar.button("Calcular Propiedades y Tiempo de Congelación"):
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
                # Calcular propiedades termofísicas generales
                densidad = calcular_densidad_alimento(temperatura, composicion)
                cp = calcular_cp_alimento(temperatura, composicion)
                k = calcular_k_alimento(temperatura, composicion)
                alpha = calcular_alpha_alimento(temperatura, composicion)

                st.subheader("Resultados de Propiedades Termofísicas")
                st.write(f"**Temperatura de Propiedades:** {temperatura}°C")
                st.write("---")
                st.metric(label="Densidad (ρ)", value=f"{densidad:.2f} kg/m³")
                st.metric(label="Calor Específico (Cp)", value=f"{cp:.2f} J/(kg·K)")
                st.metric(label="Conductividad Térmica (k)", value=f"{k:.4f} W/(m·K)")
                st.metric(label="Difusividad Térmica (α)", value=f"{alpha:.2e} m²/s")

                # Mostrar fracción de hielo a la temperatura de propiedades
                fraccion_hielo_actual = calcular_fraccion_hielo(temperatura, composicion.get('agua', 0))
                st.info(f"Fracción de Hielo a {temperatura}°C: {fraccion_hielo_actual:.3f} (kg hielo / kg alimento)")

                st.write("---")
                st.subheader("Tiempo de Congelación (Ecuación de Plank)")

                # Calcular tiempo de congelación
                tiempo_congelacion_horas = calcular_tiempo_congelacion(composicion, T0, Ta, h, geometria, dimension_a)

                if tiempo_congelacion_horas is not None:
                    st.metric(label="Tiempo de Congelación", value=f"{tiempo_congelacion_horas:.2f} horas")
                else:
                    st.warning("No se pudo calcular el tiempo de congelación. Revise los datos de entrada para esta sección.")

            except Exception as e:
                st.error(f"Ocurrió un error durante el cálculo: {e}")
                st.warning("Asegúrese de que los valores de entrada sean válidos y que la suma de la composición sea 100%.")
