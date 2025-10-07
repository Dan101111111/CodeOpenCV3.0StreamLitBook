"""
Capítulo 3 - Efectos de Cartoonización
Demostración del código 05_cartoonizing.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 3 - Efectos de Cartoonización", layout="wide")

# Título
st.title("🎨 Capítulo 3: Efectos de Cartoonización")
st.markdown("**Demostración del código: `05_cartoonizing.py`**")

def cartoonize_image(img, ksize=5, sketch_mode=False):
    """Función de cartoonización del código original"""
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4 
    
    # Convert image to grayscale 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
    # Apply median filter to the grayscale image 
    img_gray = cv2.medianBlur(img_gray, 7) 
 
    # Detect edges in the image and threshold it 
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize) 
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV) 
 
    # 'mask' is the sketch of the image 
    if sketch_mode: 
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
 
    # Resize the image to a smaller size for faster computation 
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, interpolation=cv2.INTER_AREA)
 
    # Apply bilateral filter the image multiple times 
    for i in range(num_repetitions): 
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space) 
 
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_LINEAR) 
 
    # Ensure mask and output image have compatible dimensions
    if len(img_output.shape) == 3:  # Color image
        h, w = img_output.shape[:2]
    else:  # Grayscale image  
        h, w = img_output.shape
    
    # Resize mask to match output image dimensions if needed
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h))
 
    # Add the thick boundary lines to the image using 'AND' operator 
    dst = cv2.bitwise_and(img_output, img_output, mask=mask) 
    return dst

def load_image():
    """Carga la imagen de entrada (preferentemente blue_carpet.png)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de imágenes disponibles (prioridad a blue_carpet.png)
    image_files = ['blue_carpet.png', 'green_dots.png']
    
    for img_file in image_files:
        img_path = os.path.join(script_dir, 'images', img_file)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_file
    
    # Crear imagen de ejemplo si no existe ninguna
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Crear contenido visual para cartoonización
    # Fondo degradado
    for y in range(400):
        for x in range(600):
            img[y, x] = [150 + int(50 * np.sin(x/50)), 
                        180 + int(30 * np.cos(y/40)), 
                        120 + int(80 * np.sin((x+y)/60))]
    
    # Añadir formas geométricas
    cv2.rectangle(img, (100, 100), (500, 300), (80, 120, 200), -1)
    cv2.circle(img, (300, 200), 80, (200, 80, 120), -1)
    cv2.ellipse(img, (450, 150), (60, 40), 45, 0, 360, (120, 200, 80), -1)
    cv2.putText(img, 'CARTOON', (220, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
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
        # Aplicar efectos de cartoonización
        cartoon_sketch = cartoonize_image(img, ksize=5, sketch_mode=True)
        cartoon_color = cartoonize_image(img, ksize=5, sketch_mode=False)
        
        # Mostrar estado de validación
        st.success("✅ **Imagen procesada correctamente con OpenCV**")
        
        # Información adicional sobre el procesamiento
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Imagen Original", "OK ✅", f"{img.shape[1]}x{img.shape[0]}")
        with col2:
            st.metric("Modo Sketch", "OK ✅", f"{cartoon_sketch.shape[1]}x{cartoon_sketch.shape[0]}")
        with col3:
            st.metric("Modo Color", "OK ✅", f"{cartoon_color.shape[1]}x{cartoon_color.shape[0]}")
        
    except Exception as e:
        st.error(f"❌ **Error al procesar la imagen:** {str(e)}")
        st.info("💡 Intenta con una imagen diferente o verifica el formato")
        st.stop()
    
    # Mostrar código original
    st.subheader("📄 Código Original:")
    st.code("""
# Código del archivo 05_cartoonizing.py
import cv2 
import numpy as np 

def cartoonize_image(img, ksize=5, sketch_mode=False):
    num_repetitions, sigma_color, sigma_space, ds_factor = 10, 5, 7, 4 
    # Convert image to grayscale 
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
 
    # Apply median filter to the grayscale image 
    img_gray = cv2.medianBlur(img_gray, 7) 
 
    # Detect edges in the image and threshold it 
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize) 
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV) 
 
    # 'mask' is the sketch of the image 
    if sketch_mode: 
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) 
 
    # Resize the image to a smaller size for faster computation 
    img_small = cv2.resize(img, None, fx=1.0/ds_factor, fy=1.0/ds_factor, 
                          interpolation=cv2.INTER_AREA)
 
    # Apply bilateral filter the image multiple times 
    for i in range(num_repetitions): 
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space) 
 
    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, 
                           interpolation=cv2.INTER_LINEAR) 
 
    dst = np.zeros(img_gray.shape) 
 
    # Add the thick boundary lines to the image using 'AND' operator 
    dst = cv2.bitwise_and(img_output, img_output, mask=mask) 
    return dst
""", language="python")
    
    # Mostrar imagen original
    st.subheader("🖼️ Imagen Original:")
    st.image(cv2_to_pil(img), caption=f"Imagen: {img_name}", width="stretch")
    
    # Mostrar resultados de los efectos
    st.subheader("🎨 Resultados de Cartoonización:")
    
    # Crear dos columnas para mostrar los resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Modo Sketch (Sin Color)**")
        st.image(cv2_to_pil(cartoon_sketch), caption="sketch_mode=True", width="stretch")
        st.info("💡 Detecta bordes y crea un efecto de dibujo en blanco y negro")
    
    with col2:
        st.markdown("**Modo Color (Cartoonización)**")
        st.image(cv2_to_pil(cartoon_color), caption="sketch_mode=False", width="stretch") 
        st.info("💡 Aplica filtros bilaterales múltiples para crear efecto cartoon")
    
    # Mostrar información técnica del algoritmo
    st.subheader("📊 Información del Algoritmo:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Parámetros del Algoritmo:**")
        st.code("""
num_repetitions = 10    # Iteraciones bilateral
sigma_color = 5         # Filtrado de color  
sigma_space = 7         # Filtrado espacial
ds_factor = 4          # Factor de reducción
ksize = 5              # Tamaño del kernel
        """)
        
        st.markdown("**Pasos del Proceso:**")
        st.write("1. Conversión a escala de grises")
        st.write("2. Filtro mediano (ksize=7)")  
        st.write("3. Detección de bordes (Laplacian)")
        st.write("4. Umbralización binaria (threshold=100)")
        
    with col2:
        st.markdown("**Operaciones por Modo:**")
        
        st.markdown("**🖤 Modo Sketch:**")
        st.write("• Retorna directamente la máscara de bordes")
        st.write("• Convierte a BGR para compatibilidad")
        st.write("• Efecto: Dibujo líneal simple")
        
        st.markdown("**🎨 Modo Color:**")
        st.write("• Redimensiona imagen (÷4) para optimización")
        st.write("• Aplica 10 filtros bilaterales iterativos")  
        st.write("• Restaura tamaño original (×4)")
        st.write("• Combina con máscara usando AND bitwise")
    
    # Explicación técnica detallada
    st.subheader("📝 Explicación Técnica:")
    
    st.markdown("""
    ### 🎨 **Algoritmo de Cartoonización**
    
    Este algoritmo combina **detección de bordes** y **suavizado bilateral** para crear efectos artísticos:
    
    #### **Paso 1: Detección de Bordes**
    - **Conversión a escala de grises**: Simplifica el procesamiento
    - **Filtro mediano (7x7)**: Reduce ruido preservando bordes
    - **Operador Laplaciano**: Detecta cambios bruscos de intensidad
    - **Umbralización binaria**: Convierte bordes a máscara blanco/negro
    
    #### **Paso 2: Suavizado (Solo Modo Color)**
    - **Reducción de resolución**: Acelera el procesamiento (÷4)
    - **Filtro bilateral múltiple**: 10 iteraciones para suavizado extremo
        - `sigma_color=5`: Controla similitud de colores
        - `sigma_space=7`: Controla distancia espacial
    - **Restauración de resolución**: Vuelve al tamaño original (×4)
    
    #### **Paso 3: Combinación Final**
    - **AND bitwise**: Combina imagen suavizada con máscara de bordes
    - **Resultado**: Imagen con colores planos y bordes definidos
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Técnicas Utilizadas:**")
        st.write("• **Filtro Mediano**: Reducción de ruido")
        st.write("• **Operador Laplaciano**: Detección de bordes")
        st.write("• **Umbralización Binaria**: Segmentación")
        st.write("• **Filtro Bilateral**: Suavizado preservando bordes")
        st.write("• **Operaciones Bitwise**: Combinación de imágenes")
        st.write("• **Redimensionado**: Optimización de rendimiento")
    
    with col2:
        st.markdown("**🎯 Aplicaciones Prácticas:**")
        st.write("• **Efectos fotográficos**: Estilo cartoon/anime")
        st.write("• **Preprocesamiento**: Simplificación de imágenes")
        st.write("• **Arte digital**: Conversión automática a ilustración")
        st.write("• **Compresión visual**: Reducción de detalles")
        st.write("• **Interfaces gráficas**: Avatares estilizados")
        st.write("• **Educación**: Demostración de filtros")
    
    st.info("💡 **Nota**: El filtro bilateral preserva bordes importantes mientras suaviza áreas uniformes, creando el característico efecto de colores planos del estilo cartoon.")

else:
    st.error("❌ No se pudo cargar ninguna imagen de ejemplo")