"""
Capítulo 2 - Filtros de Sharpening
Demostración del código 03_sharpening.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 2 - Filtros de Sharpening", layout="wide")

# Título
st.title("🔍 Capítulo 2: Filtros de Sharpening")
st.markdown("**Demostración del código: `03_sharpening.py`**")

def apply_sharpening_filters(img):
    """Aplica los filtros de sharpening del código original"""
    
    # Definir los kernels de sharpening (del código original)
    # Kernel 1: Sharpening básico
    kernel_sharpen_1 = np.array([[-1,-1,-1], 
                                 [-1, 9,-1], 
                                 [-1,-1,-1]]) 
    
    # Kernel 2: Sharpening intenso
    kernel_sharpen_2 = np.array([[1, 1, 1], 
                                 [1,-7, 1], 
                                 [1, 1, 1]]) 
    
    # Kernel 3: Realce de bordes (Edge Enhancement)
    kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1], 
                                 [-1, 2, 2, 2,-1], 
                                 [-1, 2, 8, 2,-1], 
                                 [-1, 2, 2, 2,-1], 
                                 [-1,-1,-1,-1,-1]]) / 8.0 
    
    # Aplicar los diferentes kernels a la imagen
    output_1 = cv2.filter2D(img, -1, kernel_sharpen_1) 
    output_2 = cv2.filter2D(img, -1, kernel_sharpen_2) 
    output_3 = cv2.filter2D(img, -1, kernel_sharpen_3) 
    
    return output_1, output_2, output_3, kernel_sharpen_1, kernel_sharpen_2, kernel_sharpen_3

def load_image():
    """Carga la imagen de entrada (preferentemente house_input.png)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de imágenes disponibles (prioridad a house_input.png como en el código original)
    image_files = ['house_input.png', 'geometrics_input.png', 'train_input.png', 'text_input.png']
    
    for img_file in image_files:
        img_path = os.path.join(script_dir, 'images', img_file)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_file
    
    # Crear imagen de ejemplo si no existe ninguna
    img = np.ones((400, 600, 3), dtype=np.uint8) * 128
    cv2.rectangle(img, (50, 50), (550, 350), (200, 200, 200), -1)
    cv2.rectangle(img, (100, 100), (500, 300), (100, 100, 100), 3)
    cv2.circle(img, (300, 200), 60, (50, 50, 50), -1)
    cv2.putText(img, 'SHARPENING', (200, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img, "imagen_generada.png"

def cv2_to_pil(cv2_img):
    """Convierte imagen de OpenCV (BGR) a PIL (RGB)"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    """Convierte imagen de PIL (RGB) a OpenCV (BGR)"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

# Sidebar para seleccionar fuente de imagen
st.sidebar.header("📁 Selección de Imagen")
image_source = st.sidebar.radio(
    "Elige la fuente de la imagen:",
    ["🖼️ Imagen de ejemplo", "📤 Cargar mi propia imagen"],
    help="Selecciona si quieres usar una imagen de ejemplo del proyecto o cargar tu propia imagen"
)

# Cargar imagen según la opción seleccionada
img = None
img_name = ""

if image_source == "📤 Cargar mi propia imagen":
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu imagen:",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Formatos soportados: PNG, JPG, JPEG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        try:
            # Cargar imagen desde el archivo subido
            pil_image = Image.open(uploaded_file)
            img = pil_to_cv2(pil_image)
            img_name = uploaded_file.name
            st.sidebar.success(f"✅ Imagen cargada: {img_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar la imagen: {str(e)}")
    else:
        st.sidebar.info("👆 Sube una imagen para procesarla")
        
else:
    # Usar imagen de ejemplo
    img, img_name = load_image()
    st.sidebar.success(f"✅ Usando imagen: {img_name}")

# Mostrar información de la imagen si está cargada
if img is not None:
    height, width = img.shape[:2]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Información de la Imagen:**")
    st.sidebar.write(f"• **Nombre:** {img_name}")
    st.sidebar.write(f"• **Dimensiones:** {width} x {height} píxeles")
    st.sidebar.write(f"• **Canales:** {img.shape[2] if len(img.shape) > 2 else 1}")
    
    # Botón para resetear a imagen de ejemplo
    if image_source == "📤 Cargar mi propia imagen":
        if st.sidebar.button("🔄 Usar imagen de ejemplo"):
            st.rerun()

if img is not None:
    # Validación de la imagen
    try:
        # Aplicar filtros de sharpening
        sharp1, sharp2, sharp3, kernel1, kernel2, kernel3 = apply_sharpening_filters(img)
        
        # Mostrar estado de validación
        st.success("✅ **Imagen procesada correctamente con OpenCV**")
        
        # Información adicional sobre el procesamiento
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filtro Básico", "OK ✅", f"{sharp1.shape[0]}x{sharp1.shape[1]}")
        with col2:
            st.metric("Filtro Intenso", "OK ✅", f"{sharp2.shape[0]}x{sharp2.shape[1]}")
        with col3:
            st.metric("Realce Bordes", "OK ✅", f"{sharp3.shape[0]}x{sharp3.shape[1]}")
        
    except Exception as e:
        st.error(f"❌ **Error al procesar la imagen:** {str(e)}")
        st.info("💡 Intenta con una imagen diferente o verifica el formato")
        st.stop()
    
    # Mostrar código original
    st.subheader("📄 Código Original:")
    st.code("""
# Código del archivo 03_sharpening.py
import cv2 
import numpy as np 

# Definir los kernels de sharpening
# Kernel 1: Sharpening básico
kernel_sharpen_1 = np.array([[-1,-1,-1], 
                             [-1, 9,-1], 
                             [-1,-1,-1]]) 

# Kernel 2: Sharpening intenso
kernel_sharpen_2 = np.array([[1, 1, 1], 
                             [1,-7, 1], 
                             [1, 1, 1]]) 

# Kernel 3: Realce de bordes (Edge Enhancement)
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1], 
                             [-1, 2, 2, 2,-1], 
                             [-1, 2, 8, 2,-1], 
                             [-1, 2, 2, 2,-1], 
                             [-1,-1,-1,-1,-1]]) / 8.0 

# Aplicar los diferentes kernels a la imagen
output_1 = cv2.filter2D(img, -1, kernel_sharpen_1) 
output_2 = cv2.filter2D(img, -1, kernel_sharpen_2) 
output_3 = cv2.filter2D(img, -1, kernel_sharpen_3) 
""", language="python")
    
    # Mostrar imagen original
    st.subheader("🖼️ Imagen Original:")
    st.image(cv2_to_pil(img), caption=f"Imagen: {img_name}", width="stretch")
    
    # Mostrar resultados de los filtros
    st.subheader("🔍 Resultados de Sharpening:")
    
    # Crear tres columnas para mostrar los resultados
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Sharpening Básico**")
        st.image(cv2_to_pil(sharp1), caption="Kernel 3x3 básico", width="stretch")
    
    with col2:
        st.markdown("**Sharpening Intenso**") 
        st.image(cv2_to_pil(sharp2), caption="Kernel negativo intenso", width="stretch")
    
    with col3:
        st.markdown("**Realce de Bordes**")
        st.image(cv2_to_pil(sharp3), caption="Kernel 5x5 normalizado", width="stretch")
    
    # Mostrar información técnica de los kernels
    st.subheader("📊 Información de los Kernels:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Kernel Sharpening Básico (3x3):**")
        st.code(f"""
{kernel1}
        """)
        st.info("💡 Kernel clásico con centro positivo fuerte (+9) y valores negativos alrededor (-1)")
    
    with col2:
        st.markdown("**Kernel Sharpening Intenso (3x3):**")
        st.code(f"""
{kernel2}
        """)
        st.info("💡 Kernel con centro negativo fuerte (-7) que crea un efecto más agresivo")
    
    with col3:
        st.markdown("**Kernel Realce de Bordes (5x5):**")
        st.code(f"""
{kernel3}
        """)
        st.info("💡 Kernel más grande y normalizado (÷8) para un realce suave y controlado")
    
    # Explicación técnica
    st.subheader("📝 Explicación Técnica:")
    
    st.markdown("""
    ### 🔍 **Filtros de Sharpening (Realce de Detalles)**
    
    Los filtros de sharpening son kernels de convolución que realzan los detalles y bordes de una imagen:
    
    #### **1. Sharpening Básico:**
    - **Propósito**: Realza contornos de forma equilibrada
    - **Funcionamiento**: Centro positivo alto (+9) rodeado de valores negativos (-1)
    - **Resultado**: Aumento de contraste en bordes sin exceso de ruido
    
    #### **2. Sharpening Intenso:**
    - **Propósito**: Realce agresivo de detalles finos
    - **Funcionamiento**: Centro negativo (-7) con valores positivos alrededor
    - **Resultado**: Efecto más dramático, puede introducir artefactos
    
    #### **3. Realce de Bordes:**
    - **Propósito**: Realce suave y controlado
    - **Funcionamiento**: Kernel 5x5 con normalización (÷8)
    - **Resultado**: Mejora sutil de bordes sin distorsión excesiva
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🎯 Conceptos Clave:**")
        st.write("• **Convolución**: Operación matemática entre imagen y kernel")
        st.write("• **Kernel**: Matriz pequeña que define la transformación")
        st.write("• **Centro positivo**: Resalta el píxel central")
        st.write("• **Valores negativos**: Restan información de píxeles vecinos")
        st.write("• **Normalización**: División para controlar la intensidad")
    
    with col2:
        st.markdown("**📈 Aplicaciones Prácticas:**")
        st.write("• **Fotografía digital**: Mejorar nitidez de imágenes")
        st.write("• **Imágenes médicas**: Realzar estructuras anatómicas")
        st.write("• **Inspección industrial**: Detectar defectos y bordes")
        st.write("• **Análisis de documentos**: Mejorar legibilidad de texto")
        st.write("• **Visión por computador**: Preprocesamiento para detección")
    
    st.info("💡 **Nota**: El sharpening excesivo puede introducir ruido y artefactos. Es importante encontrar el equilibrio adecuado para cada aplicación.")

else:
    st.error("❌ No se pudo cargar ninguna imagen de ejemplo")