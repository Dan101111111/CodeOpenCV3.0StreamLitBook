"""
Aplicación Streamlit - Efectos de Cartoonización Interactivos
Aplicación educativa para explorar diferentes efectos de cartoonización y filtros artísticos
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# Configuración de la página
st.set_page_config(
    page_title="Efectos de Cartoonización Interactivos",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎨 Efectos de Cartoonización Interactivos")
st.markdown("**Explora diferentes efectos artísticos y de cartoonización en imágenes**")

# Funciones auxiliares
@st.cache_data
def load_sample_image():
    """Carga una imagen de ejemplo desde la carpeta images"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de imágenes disponibles
    sample_images = ['blue_carpet.png', 'green_dots.png']
    
    for img_name in sample_images:
        img_path = os.path.join(script_dir, 'images', img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_name
    
    # Si no se encuentra ninguna imagen, crear una de ejemplo
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Crear contenido visual para cartoonización
    # Fondo degradado
    for y in range(400):
        for x in range(600):
            img[y, x] = [150 + int(50 * np.sin(x/50)), 
                        180 + int(30 * np.cos(y/40)), 
                        120 + int(80 * np.sin((x+y)/60))]
    
    # Agregar formas geométricas
    cv2.circle(img, (150, 150), 80, (255, 100, 100), -1)
    cv2.circle(img, (450, 150), 80, (100, 255, 100), -1)
    cv2.rectangle(img, (200, 250), (400, 350), (100, 100, 255), -1)
    cv2.ellipse(img, (300, 320), (80, 40), 0, 0, 360, (255, 255, 100), -1)
    
    # Agregar texto
    cv2.putText(img, 'CARTOON', (220, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, 'EFFECTS', (220, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    return img, "imagen_generada.png"

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

# Funciones de efectos de cartoonización
def cartoonize_image(img, ksize=5, sketch_mode=False, num_repetitions=10, 
                    sigma_color=5, sigma_space=7, ds_factor=4, 
                    edge_threshold=100, blur_kernel=7):
    """
    Aplica efecto de cartoonización a la imagen (versión mejorada del original)
    
    Args:
        img: Imagen de entrada
        ksize: Tamaño del kernel para detección de bordes
        sketch_mode: Si True, retorna solo el sketch en blanco y negro
        num_repetitions: Número de repeticiones del filtro bilateral
        sigma_color: Parámetro sigma_color del filtro bilateral
        sigma_space: Parámetro sigma_space del filtro bilateral
        ds_factor: Factor de reducción de tamaño para optimización
        edge_threshold: Umbral para detección de bordes
        blur_kernel: Tamaño del kernel para median blur
    
    Returns:
        Imagen cartoonizada
    """
    
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro mediano para reducir ruido
    img_gray = cv2.medianBlur(img_gray, blur_kernel)
    
    # Detectar bordes usando Laplaciano
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=ksize)
    ret, mask = cv2.threshold(edges, edge_threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Si es modo sketch, retornar solo la máscara
    if sketch_mode:
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Reducir tamaño para optimización
    h, w = img.shape[:2]
    img_small = cv2.resize(img, (w//ds_factor, h//ds_factor), interpolation=cv2.INTER_AREA)
    
    # Aplicar filtro bilateral múltiples veces para suavizar
    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, ksize, sigma_color, sigma_space)
    
    # Restaurar tamaño original
    img_output = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Aplicar máscara de bordes
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    
    return dst

def apply_pencil_sketch(img, blur_value=21):
    """Aplica efecto de dibujo a lápiz"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.medianBlur(img_gray, blur_value)
    edges = cv2.adaptiveThreshold(img_gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                 cv2.THRESH_BINARY, 9, 9)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def apply_watercolor_effect(img, sigma_s=50, sigma_r=0.2):
    """Aplica efecto acuarela"""
    # Usar edge-preserving filter
    img_filtered = cv2.edgePreservingFilter(img, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
    return img_filtered

def apply_oil_painting_effect(img, size=7, dynRatio=1):
    """Aplica efecto de pintura al óleo"""
    try:
        # Intentar usar la función de OpenCV si está disponible
        return cv2.xphoto.oilPainting(img, size, dynRatio)
    except:
        # Fallback: usar filtro bilateral intenso
        result = img.copy()
        for _ in range(3):
            result = cv2.bilateralFilter(result, 15, 80, 80)
        return result

def apply_stylization(img, sigma_s=100, sigma_r=0.25):
    """Aplica efecto de estilización"""
    return cv2.stylization(img, sigma_s=sigma_s, sigma_r=sigma_r)

def apply_edge_preserving_filter(img, flags=1, sigma_s=50, sigma_r=0.4):
    """Aplica filtro que preserva bordes"""
    return cv2.edgePreservingFilter(img, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)

def apply_detail_enhance(img, sigma_s=10, sigma_r=0.15):
    """Aplica realce de detalles"""
    return cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)

def apply_color_quantization(img, k=8):
    """Aplica cuantización de colores para efecto poster"""
    # Reshape para k-means
    data = img.reshape((-1, 3))
    data = np.float32(data)
    
    # Criterios y aplicar k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convertir de vuelta a uint8 y reshape
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_img = segmented_data.reshape(img.shape)
    
    return segmented_img

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
    pil_image = Image.open(uploaded_file)
    image = pil_to_cv2(pil_image)
    image_source = f"📁 {uploaded_file.name}"
else:
    image, sample_name = load_sample_image()
    image_source = f"🖼️ {sample_name}"

# Información de la imagen
h, w = image.shape[:2]
st.sidebar.info(f"**Imagen actual:** {image_source}")
st.sidebar.info(f"**Dimensiones:** {w} x {h} píxeles")

st.sidebar.markdown("---")

# Menú de efectos
st.sidebar.subheader("🎨 Tipo de Efecto")

effect_type = st.sidebar.selectbox(
    "Selecciona el efecto artístico:",
    options=['cartoonize', 'sketch', 'pencil_sketch', 'watercolor', 'oil_painting', 
             'stylization', 'edge_preserving', 'detail_enhance', 'color_quantization'],
    format_func=lambda x: {
        'cartoonize': '🎭 Cartoonización',
        'sketch': '✏️ Sketch B&N',
        'pencil_sketch': '✏️ Dibujo a Lápiz',
        'watercolor': '🌊 Acuarela',
        'oil_painting': '🖌️ Pintura al Óleo',
        'stylization': '🎨 Estilización',
        'edge_preserving': '🔲 Preservar Bordes',
        'detail_enhance': '⚡ Realce de Detalles',
        'color_quantization': '🎯 Cuantización de Color'
    }[x],
    help="Selecciona el tipo de efecto artístico a aplicar"
)

st.sidebar.markdown("---")

# Controles específicos según el efecto
if effect_type in ['cartoonize', 'sketch']:
    st.sidebar.subheader("🎭 Parámetros de Cartoonización")
    
    ksize = st.sidebar.slider("Tamaño kernel bordes:", 3, 15, 5, 2)
    num_repetitions = st.sidebar.slider("Repeticiones filtro:", 1, 20, 10, 1)
    sigma_color = st.sidebar.slider("Sigma color:", 1, 50, 5, 1)
    sigma_space = st.sidebar.slider("Sigma espacio:", 1, 50, 7, 1)
    ds_factor = st.sidebar.slider("Factor reducción:", 2, 8, 4, 1)
    edge_threshold = st.sidebar.slider("Umbral bordes:", 50, 200, 100, 5)
    blur_kernel = st.sidebar.slider("Suavizado inicial:", 3, 15, 7, 2)
    
    sketch_mode = effect_type == 'sketch'
    processed_image = cartoonize_image(image, ksize, sketch_mode, num_repetitions, 
                                     sigma_color, sigma_space, ds_factor, 
                                     edge_threshold, blur_kernel)

elif effect_type == 'pencil_sketch':
    st.sidebar.subheader("✏️ Parámetros de Lápiz")
    blur_value = st.sidebar.slider("Suavizado:", 3, 51, 21, 2)
    
    processed_image = apply_pencil_sketch(image, blur_value)

elif effect_type == 'watercolor':
    st.sidebar.subheader("🌊 Parámetros de Acuarela")
    sigma_s = st.sidebar.slider("Tamaño vecindario:", 10, 200, 50, 5)
    sigma_r = st.sidebar.slider("Promediado colores:", 0.0, 1.0, 0.2, 0.05)
    
    processed_image = apply_watercolor_effect(image, sigma_s, sigma_r)

elif effect_type == 'oil_painting':
    st.sidebar.subheader("🖌️ Parámetros Óleo")
    size = st.sidebar.slider("Tamaño pincel:", 1, 15, 7, 1)
    dynRatio = st.sidebar.slider("Ratio dinámico:", 1, 10, 1, 1)
    
    processed_image = apply_oil_painting_effect(image, size, dynRatio)

elif effect_type == 'stylization':
    st.sidebar.subheader("🎨 Parámetros Estilización")
    sigma_s = st.sidebar.slider("Suavizado:", 50, 300, 100, 10)
    sigma_r = st.sidebar.slider("Similaridad:", 0.0, 1.0, 0.25, 0.05)
    
    processed_image = apply_stylization(image, sigma_s, sigma_r)

elif effect_type == 'edge_preserving':
    st.sidebar.subheader("🔲 Parámetros Preservar Bordes")
    flags = st.sidebar.selectbox("Tipo algoritmo:", [1, 2], format_func=lambda x: "Normalizado" if x == 1 else "Recursivo")
    sigma_s = st.sidebar.slider("Tamaño vecindario:", 10, 200, 50, 5)
    sigma_r = st.sidebar.slider("Diferencia color:", 0.0, 1.0, 0.4, 0.05)
    
    processed_image = apply_edge_preserving_filter(image, flags, sigma_s, sigma_r)

elif effect_type == 'detail_enhance':
    st.sidebar.subheader("⚡ Parámetros Realce")
    sigma_s = st.sidebar.slider("Suavizado:", 5, 50, 10, 1)
    sigma_r = st.sidebar.slider("Contraste:", 0.0, 0.5, 0.15, 0.01)
    
    processed_image = apply_detail_enhance(image, sigma_s, sigma_r)

elif effect_type == 'color_quantization':
    st.sidebar.subheader("🎯 Cuantización de Color")
    k = st.sidebar.slider("Número de colores:", 2, 32, 8, 1)
    
    processed_image = apply_color_quantization(image, k)

# Opciones de visualización
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Opciones de Vista")

show_comparison = st.sidebar.checkbox("Comparación Lado a Lado", True)
show_all_effects = st.sidebar.checkbox("Mostrar Galería de Efectos", False)

# Área principal
effect_names = {
    'cartoonize': '🎭 Cartoonización',
    'sketch': '✏️ Sketch B&N',
    'pencil_sketch': '✏️ Dibujo a Lápiz',
    'watercolor': '🌊 Acuarela',
    'oil_painting': '🖌️ Pintura al Óleo',
    'stylization': '🎨 Estilización',
    'edge_preserving': '🔲 Preservar Bordes',
    'detail_enhance': '⚡ Realce de Detalles',
    'color_quantization': '🎯 Cuantización de Color'
}

if show_all_effects:
    st.subheader("🖼️ Galería de Efectos Artísticos")
    
    # Generar todos los efectos con parámetros por defecto
    effects_gallery = {
        '🎭 Original': image,
        '🎭 Cartoonización': cartoonize_image(image),
        '✏️ Sketch': cartoonize_image(image, sketch_mode=True),
        '✏️ Lápiz': apply_pencil_sketch(image),
        '🌊 Acuarela': apply_watercolor_effect(image),
        '🖌️ Óleo': apply_oil_painting_effect(image),
        '🎨 Estilización': apply_stylization(image),
        '🔲 Preservar Bordes': apply_edge_preserving_filter(image),
        '⚡ Realce': apply_detail_enhance(image)
    }
    
    # Layout en grid 3x3
    cols = st.columns(3)
    for i, (name, img) in enumerate(effects_gallery.items()):
        with cols[i % 3]:
            st.write(f"**{name}**")
            st.image(cv2_to_pil(img), use_container_width=True)

else:
    if show_comparison:
        st.subheader(f"{effect_names[effect_type]} - Comparación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🖼️ Imagen Original**")
            st.image(cv2_to_pil(image), use_container_width=True)
        
        with col2:
            st.write(f"**{effect_names[effect_type]}**")
            st.image(cv2_to_pil(processed_image), use_container_width=True)
    else:
        st.subheader(f"Resultado: {effect_names[effect_type]}")
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
            file_name=f"artistic_{effect_type}.png",
            mime="image/png",
            use_container_width=True
        )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("🎨 Sobre los Efectos"):
    st.markdown("""
    **🎭 Cartoonización:**
    - Combina detección de bordes y filtro bilateral
    - Reduce colores manteniendo bordes definidos
    - Efecto similar a dibujos animados
    
    **✏️ Sketch:**
    - Versión en blanco y negro de cartoonización
    - Enfatiza contornos y estructuras
    - Simula dibujo a mano alzada
    
    **🌊 Acuarela:**
    - Preserva bordes mientras suaviza texturas
    - Efecto de pintura fluida
    - Mantiene detalles importantes
    
    **🖌️ Pintura al Óleo:**
    - Textura espesa y pinceladas visibles
    - Suavizado intenso de colores
    - Efecto artístico tradicional
    
    **🎨 Estilización:**
    - Simplifica imagen manteniendo estructura
    - Reduce ruido y detalles menores
    - Aspecto limpio y artístico
    """)

with st.sidebar.expander("🔧 Parámetros Técnicos"):
    st.markdown("""
    **Filtro Bilateral:**
    - Sigma Color: controla diferencias de color
    - Sigma Space: controla distancia espacial
    - Preserva bordes mientras suaviza
    
    **Detección de Bordes:**
    - Laplaciano para detectar cambios de intensidad
    - Umbralización para crear máscara binaria
    - Combinación con imagen suavizada
    
    **Optimización:**
    - Reducción de tamaño temporal
    - Múltiples pasadas del filtro
    - Restauración de dimensiones originales
    """)

with st.sidebar.expander("🎯 Algoritmos Utilizados"):
    st.markdown("""
    **OpenCV Functions:**
    - `cv2.bilateralFilter()`: suavizado preservando bordes
    - `cv2.Laplacian()`: detección de bordes
    - `cv2.medianBlur()`: reducción de ruido
    - `cv2.threshold()`: binarización
    - `cv2.stylization()`: estilización automática
    - `cv2.edgePreservingFilter()`: preservación de bordes
    - `cv2.detailEnhance()`: realce de detalles
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎨 <strong>Efectos de Cartoonización Interactivos</strong> | Capítulo 3 - Filtros Artísticos</p>
        <p><small>Explora diferentes técnicas de procesamiento artístico de imágenes</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)