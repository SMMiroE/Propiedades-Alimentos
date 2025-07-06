import streamlit as st
import numpy as np
from scipy.special import jv
import pandas as pd

# --- Constantes globales para la ecuación de fracción de hielo y PMs ---
L_molar_fusion_agua = 6010.0 # J/mol (333.6 J/g * 18.015 g/mol)
R_gas = 8.314 # J/(mol·K)
T0_ref = 273.15 # K (0°C)
PM_agua = 18.015 # g/mol (o kg/kmol)

# ... (otras funciones de cálculo: densidad_agua, cp_agua, k_agua, etc.
# y las funciones principales de cálculo: calcular_fraccion_hielo,
# calcular_densidad_alimento, calcular_cp_alimento, calcular_k_alimento,
# calcular_alpha_alimento, get_heisler_coeffs, get_heisler_position_factor,
# calcular_temperatura_final_punto_frio, calcular_tiempo_para_temperatura,
# calcular_temperatura_posicion, calcular_tiempo_congelacion_plank,
# calcular_pm_solido_aparente) ...
# No incluyo todo el código aquí para no repetir, asumo que las funciones
# mencionadas en la respuesta anterior están presentes y actualizadas.

# --- Interfaz de Usuario Streamlit ---

# ... (todo el código de la interfaz de usuario: st.set_page_config,
# st.title, st.markdown, st.sidebar.header, inputs, etc.,
# hasta la sección de Información Adicional) ...

# Asegúrate de que las definiciones de funciones y las constantes
# (L_molar_fusion_agua, R_gas, T0_ref, PM_agua)
# estén en la parte superior de tu script antes de la interfaz de Streamlit,
# como se mostró en la respuesta anterior.

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
