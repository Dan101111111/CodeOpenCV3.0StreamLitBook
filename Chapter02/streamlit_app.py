"""
Aplicación Streamlit - Filtros de Sharpening
Aplicación interactiva para aplicar diferentes kernels de sharpening a imágenes
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# Configuración de la página
st.set_page_config(
    page_title="Filtros de Sharpening Interactivos",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔍 Filtros de Sharpening Interactivos")
st.markdown("**Explora diferentes técnicas de sharpening en tus imágenes**")

# Funciones auxiliares
@st.cache_data
def load_sample_image():
    """Carga una imagen de ejemplo desde la carpeta images"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de imágenes disponibles
    sample_images = ['house_input.png', 'geometrics_input.png', 'train_input.png']
    
    for img_name in sample_images:
        img_path = os.path.join(script_dir, 'images', img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_name
    
    # Si no se encuentra ninguna imagen, crear una de ejemplo
    img = np.ones((400, 600, 3), dtype=np.uint8) * 128
    cv2.rectangle(img, (50, 50), (550, 350), (255, 255, 255), -1)
    cv2.rectangle(img, (100, 100), (500, 300), (0, 0, 0), 2)
    cv2.putText(img, 'IMAGEN DE EJEMPLO', (150, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img, "imagen_generada.png"

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def apply_sharpening_filter(image, filter_type, intensity=1.0):
    """
    Aplica filtros de sharpening a la imagen
    
    Args:
        image: Imagen en formato OpenCV (BGR)
        filter_type: Tipo de filtro ('basico', 'intenso', 'bordes')
        intensity: Intensidad del efecto (0.1 - 2.0)
    
    Returns:
        Imagen procesada
    """
    
    # Definir los kernels de sharpening
    kernels = {
        'basico': np.array([[-1,-1,-1], 
                           [-1, 9,-1], 
                           [-1,-1,-1]], dtype=np.float32),
        
        'intenso': np.array([[1, 1, 1], 
                            [1,-7, 1], 
                            [1, 1, 1]], dtype=np.float32),
        
        'bordes': np.array([[-1,-1,-1,-1,-1], 
                           [-1, 2, 2, 2,-1], 
                           [-1, 2, 8, 2,-1], 
                           [-1, 2, 2, 2,-1], 
                           [-1,-1,-1,-1,-1]], dtype=np.float32) / 8.0
    }
    
    kernel = kernels[filter_type]
    
    # Ajustar intensidad
    if filter_type in ['basico', 'intenso']:
        # Para kernels básicos, interpolamos entre identidad y kernel completo
        if kernel.shape == (3, 3):
            identity = np.array([[0,0,0], [0,1,0], [0,0,0]], dtype=np.float32)
        else:
            identity = np.zeros((5,5), dtype=np.float32)
            identity[2,2] = 1
        
        kernel = identity + intensity * (kernel - identity)
    else:
        # Para edge enhancement, multiplicamos por intensidad
        kernel = kernel * intensity
    
    # Aplicar el filtro
    result = cv2.filter2D(image, -1, kernel)
    
    # Asegurar rango válido
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

# Sidebar - Controles
st.sidebar.header("⚙️ Controles")

# Upload de imagen
uploaded_file = st.sidebar.file_uploader(
    "📁 Sube tu imagen", 
    type=['png', 'jpg', 'jpeg'],
    help="Formatos soportados: PNG, JPG, JPEG"
)

# Cargar imagen
if uploaded_file is not None:
    # Imagen subida por el usuario
    pil_image = Image.open(uploaded_file)
    image = pil_to_cv2(pil_image)
    image_source = f"📁 {uploaded_file.name}"
else:
    # Imagen de ejemplo
    image, sample_name = load_sample_image()
    image_source = f"🖼️ {sample_name}"

# Información de la imagen
h, w = image.shape[:2]
st.sidebar.info(f"**Imagen actual:** {image_source}")
st.sidebar.info(f"**Dimensiones:** {w} x {h} píxeles")

st.sidebar.markdown("---")

# Controles de filtrado
st.sidebar.subheader("🎛️ Configuraciones del Filtro")

filter_type = st.sidebar.selectbox(
    "Tipo de Filtro:",
    options=['basico', 'intenso', 'bordes'],
    format_func=lambda x: {
        'basico': '🔍 Sharpening Básico',
        'intenso': '⚡ Sharpening Intenso', 
        'bordes': '🔲 Realce de Bordes'
    }[x],
    help="Selecciona el tipo de filtro de sharpening"
)

intensity = st.sidebar.slider(
    "Intensidad:",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Ajusta la intensidad del efecto"
)

# Opciones de visualización
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Opciones de Vista")

show_comparison = st.sidebar.checkbox(
    "Comparación Lado a Lado",
    value=True,
    help="Muestra original y procesada juntas"
)

show_all_filters = st.sidebar.checkbox(
    "Mostrar Todos los Filtros",
    value=False,
    help="Muestra los tres filtros simultáneamente"
)

# Área principal
if show_all_filters:
    st.subheader("🔍 Comparación de Todos los Filtros")
    
    # Aplicar todos los filtros
    result_basico = apply_sharpening_filter(image, 'basico', intensity)
    result_intenso = apply_sharpening_filter(image, 'intenso', intensity)
    result_bordes = apply_sharpening_filter(image, 'bordes', intensity)
    
    # Layout en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🖼️ Imagen Original**")
        st.image(cv2_to_pil(image), use_container_width=True)
        
    with col2:
        st.write("**🔍 Sharpening Básico**")
        st.image(cv2_to_pil(result_basico), use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.write("**⚡ Sharpening Intenso**")
        st.image(cv2_to_pil(result_intenso), use_container_width=True)
        
    with col4:
        st.write("**🔲 Realce de Bordes**")
        st.image(cv2_to_pil(result_bordes), use_container_width=True)

else:
    # Aplicar filtro seleccionado
    processed_image = apply_sharpening_filter(image, filter_type, intensity)
    
    filter_names = {
        'basico': '🔍 Sharpening Básico',
        'intenso': '⚡ Sharpening Intenso',
        'bordes': '🔲 Realce de Bordes'
    }
    
    if show_comparison:
        st.subheader(f"{filter_names[filter_type]} - Comparación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🖼️ Original**")
            st.image(cv2_to_pil(image), use_container_width=True)
        
        with col2:
            st.write(f"**{filter_names[filter_type]}**")
            st.image(cv2_to_pil(processed_image), use_container_width=True)
    else:
        st.subheader(f"Resultado: {filter_names[filter_type]}")
        st.image(cv2_to_pil(processed_image), use_container_width=True)
    
    # Botón de descarga
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        processed_pil = cv2_to_pil(processed_image)
        buf = io.BytesIO()
        processed_pil.save(buf, format='PNG')
        
        st.download_button(
            label="📥 Descargar Imagen Procesada",
            data=buf.getvalue(),
            file_name=f"sharpened_{filter_type}_{intensity:.1f}.png",
            mime="image/png",
            use_container_width=True
        )

# Información sobre los filtros
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Información")

with st.sidebar.expander("📚 Sobre los Filtros"):
    st.markdown("""
    **🔍 Sharpening Básico:**
    - Realza detalles suavemente
    - Kernel 3x3 balanceado
    - Ideal para uso general
    
    **⚡ Sharpening Intenso:**
    - Efecto más dramático
    - Puede introducir artefactos
    - Para imágenes muy suaves
    
    **🔲 Realce de Bordes:**
    - Enfatiza contornos
    - Kernel 5x5 especializado
    - Mejor para detección de bordes
    """)

with st.sidebar.expander("🔧 Kernels Utilizados"):
    st.markdown("""
    **Básico:**
    ```
    [[-1, -1, -1],
     [-1,  9, -1],
     [-1, -1, -1]]
    ```
    
    **Intenso:**
    ```
    [[ 1,  1,  1],
     [ 1, -7,  1],
     [ 1,  1,  1]]
    ```
    
    **Bordes:**
    ```
    [[-1, -1, -1, -1, -1],
     [-1,  2,  2,  2, -1],
     [-1,  2,  8,  2, -1],
     [-1,  2,  2,  2, -1],
     [-1, -1, -1, -1, -1]] / 8
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎨 <strong>Filtros de Sharpening Interactivos</strong> | Desarrollado con Streamlit y OpenCV</p>
        <p><small>Sube una imagen o usa las de ejemplo para experimentar con diferentes filtros</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)