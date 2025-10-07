"""
Aplicación Streamlit - Transformaciones Afines Interactivas
Aplicación educativa para explorar diferentes tipos de transformaciones afines
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import math

# Configuración de la página
st.set_page_config(
    page_title="Transformaciones Afines Interactivas",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔄 Transformaciones Afines Interactivas")
st.markdown("**Explora diferentes tipos de transformaciones geométricas en imágenes**")

# Funciones auxiliares
@st.cache_data
def load_sample_image():
    """Carga la imagen de ejemplo desde la carpeta images"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'images', 'input.jpg')
    
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            return img, "input.jpg"
    
    # Si no se encuentra la imagen, crear una de ejemplo
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Crear un patrón geométrico para mejor visualización de transformaciones
    cv2.rectangle(img, (50, 50), (550, 350), (200, 200, 200), -1)
    cv2.rectangle(img, (100, 100), (500, 300), (150, 150, 150), 3)
    cv2.rectangle(img, (150, 150), (450, 250), (100, 100, 100), 2)
    
    # Agregar algunos círculos y líneas para mejor referencia
    cv2.circle(img, (300, 200), 50, (50, 50, 50), 2)
    cv2.line(img, (150, 150), (450, 250), (0, 0, 0), 2)
    cv2.line(img, (450, 150), (150, 250), (0, 0, 0), 2)
    
    cv2.putText(img, 'TRANSFORMACIONES', (180, 190), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'AFINES', (230, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img, "imagen_generada.jpg"

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

# Funciones de transformación
def apply_translation(image, tx, ty):
    """Aplica transformación de traslación"""
    rows, cols = image.shape[:2]
    
    # Matriz de transformación
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    
    transformed = cv2.warpAffine(image, M, (cols, rows))
    
    return transformed, M

def apply_rotation(image, angle, center_x=None, center_y=None):
    """Aplica transformación de rotación"""
    rows, cols = image.shape[:2]
    
    # Usar centro de la imagen si no se especifica
    if center_x is None:
        center_x = cols // 2
    if center_y is None:
        center_y = rows // 2
    
    # Matriz de rotación
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    
    transformed = cv2.warpAffine(image, M, (cols, rows))
    
    return transformed, M

def apply_scaling(image, scale_x, scale_y):
    """Aplica transformación de escalado"""
    rows, cols = image.shape[:2]
    
    # Matriz de escalado
    M = np.float32([[scale_x, 0, 0],
                    [0, scale_y, 0]])
    
    transformed = cv2.warpAffine(image, M, (cols, rows))
    
    return transformed, M

def apply_shearing(image, shear_x, shear_y):
    """Aplica transformación de cizallamiento (shearing)"""
    rows, cols = image.shape[:2]
    
    # Matriz de cizallamiento
    M = np.float32([[1, shear_x, 0],
                    [shear_y, 1, 0]])
    
    transformed = cv2.warpAffine(image, M, (cols, rows))
    
    return transformed, M

def apply_custom_affine(image, src_points, dst_points):
    """Aplica transformación afín personalizada"""
    rows, cols = image.shape[:2]
    
    # Matriz de transformación afín
    M = cv2.getAffineTransform(src_points, dst_points)
    
    transformed = cv2.warpAffine(image, M, (cols, rows))
    
    return transformed, M

def apply_perspective_transform(image, perspective_strength):
    """Aplica transformación de perspectiva"""
    rows, cols = image.shape[:2]
    
    # Puntos de origen (esquinas de la imagen)
    src_points = np.float32([[0, 0],
                            [cols-1, 0],
                            [0, rows-1],
                            [cols-1, rows-1]])
    
    # Puntos de destino con efecto perspectiva
    offset = perspective_strength * cols / 4
    dst_points = np.float32([[offset, 0],
                            [cols-1-offset, 0],
                            [0, rows-1],
                            [cols-1, rows-1]])
    
    # Matriz de transformación perspectiva
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    transformed = cv2.warpPerspective(image, M, (cols, rows))
    
    return transformed, M

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

# Menú de transformaciones
st.sidebar.subheader("🔄 Tipo de Transformación")

transformation_type = st.sidebar.selectbox(
    "Selecciona la transformación:",
    options=['traslacion', 'rotacion', 'escalado', 'cizallamiento', 'afin_personalizada', 'perspectiva'],
    format_func=lambda x: {
        'traslacion': '📐 Traslación',
        'rotacion': '🔄 Rotación',
        'escalado': '📏 Escalado',
        'cizallamiento': '📐 Cizallamiento',
        'afin_personalizada': '🎯 Afín Personalizada',
        'perspectiva': '👁️ Perspectiva'
    }[x],
    help="Selecciona el tipo de transformación geométrica"
)

st.sidebar.markdown("---")

# Inicializar variables por defecto para evitar errores
tx, ty = 0, 0
angle = 0
scale_x, scale_y = 1.0, 1.0
shear_x, shear_y = 0.0, 0.0
factor_top, factor_bottom = 0.6, 0.4
perspective_strength = 0.2
explanation_text = ""

# Controles específicos según el tipo de transformación
if transformation_type == 'traslacion':
    st.sidebar.subheader("📐 Parámetros de Traslación")
    tx = st.sidebar.slider("Desplazamiento X:", -200, 200, 50, 5)
    ty = st.sidebar.slider("Desplazamiento Y:", -200, 200, 30, 5)
    
    transformed, matrix = apply_translation(image, tx, ty)
    explanation_text = f"Desplaza la imagen {tx} píxeles en X y {ty} píxeles en Y"
    
elif transformation_type == 'rotacion':
    st.sidebar.subheader("🔄 Parámetros de Rotación")
    angle = st.sidebar.slider("Ángulo (grados):", -180, 180, 45, 1)
    
    use_custom_center = st.sidebar.checkbox("Centro personalizado", False)
    if use_custom_center:
        center_x = st.sidebar.slider("Centro X:", 0, w, w//2, 1)
        center_y = st.sidebar.slider("Centro Y:", 0, h, h//2, 1)
    else:
        center_x, center_y = None, None
    
    transformed, matrix = apply_rotation(image, angle, center_x, center_y)
    explanation_text = f"Rota {angle}° alrededor del centro especificado"
    
elif transformation_type == 'escalado':
    st.sidebar.subheader("📏 Parámetros de Escalado")
    scale_x = st.sidebar.slider("Escala X:", 0.1, 3.0, 1.2, 0.1)
    scale_y = st.sidebar.slider("Escala Y:", 0.1, 3.0, 0.8, 0.1)
    
    transformed, matrix = apply_scaling(image, scale_x, scale_y)
    explanation_text = f"Escala {scale_x}x en X y {scale_y}x en Y"
    
elif transformation_type == 'cizallamiento':
    st.sidebar.subheader("📐 Parámetros de Cizallamiento")
    shear_x = st.sidebar.slider("Cizallamiento X:", -1.0, 1.0, 0.3, 0.1)
    shear_y = st.sidebar.slider("Cizallamiento Y:", -1.0, 1.0, 0.0, 0.1)
    
    transformed, matrix = apply_shearing(image, shear_x, shear_y)
    explanation_text = f"Cizalla {shear_x} en X y {shear_y} en Y"
    
elif transformation_type == 'afin_personalizada':
    st.sidebar.subheader("🎯 Transformación Afín Original")
    st.sidebar.markdown("*Recreación del ejemplo original del libro*")
    
    # Usar los puntos del ejemplo original
    rows, cols = image.shape[:2]
    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.4*(cols-1)),rows-1]])
    
    # Permitir ajustes
    factor_top = st.sidebar.slider("Factor superior:", 0.1, 1.0, 0.6, 0.05)
    factor_bottom = st.sidebar.slider("Factor inferior:", 0.1, 1.0, 0.4, 0.05)
    
    dst_points = np.float32([[0,0], 
                            [int(factor_top*(cols-1)),0], 
                            [int(factor_bottom*(cols-1)),rows-1]])
    
    transformed, matrix = apply_custom_affine(image, src_points, dst_points)
    explanation_text = "Transformación afín que simula perspectiva"
    
elif transformation_type == 'perspectiva':
    st.sidebar.subheader("👁️ Transformación de Perspectiva")
    perspective_strength = st.sidebar.slider("Fuerza de perspectiva:", 0.0, 0.5, 0.2, 0.05)
    
    transformed, matrix = apply_perspective_transform(image, perspective_strength)
    explanation_text = f"Efecto de perspectiva con fuerza {perspective_strength}"

# Opciones de visualización
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Opciones de Vista")

show_comparison = st.sidebar.checkbox("Comparación Lado a Lado", True)
show_matrix = st.sidebar.checkbox("Mostrar Matriz de Transformación", True)
show_grid = st.sidebar.checkbox("Mostrar Cuadrícula de Referencia", False)

# Área principal
transformation_names = {
    'traslacion': '📐 Traslación',
    'rotacion': '🔄 Rotación', 
    'escalado': '📏 Escalado',
    'cizallamiento': '📐 Cizallamiento',
    'afin_personalizada': '🎯 Afín Personalizada',
    'perspectiva': '👁️ Perspectiva'
}

# Función para agregar cuadrícula
def add_grid_overlay(img, grid_size=50):
    """Agrega una cuadrícula de referencia a la imagen"""
    img_with_grid = img.copy()
    h, w = img.shape[:2]
    
    # Líneas verticales
    for x in range(0, w, grid_size):
        cv2.line(img_with_grid, (x, 0), (x, h), (0, 255, 0), 1)
    
    # Líneas horizontales  
    for y in range(0, h, grid_size):
        cv2.line(img_with_grid, (0, y), (w, y), (0, 255, 0), 1)
    
    return img_with_grid

# Aplicar cuadrícula si está seleccionada
display_image = add_grid_overlay(image) if show_grid else image
display_transformed = add_grid_overlay(transformed) if show_grid else transformed

if show_comparison:
    st.subheader(f"{transformation_names[transformation_type]} - Comparación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🖼️ Imagen Original**")
        st.image(cv2_to_pil(display_image), use_container_width=True)
    
    with col2:
        st.write(f"**{transformation_names[transformation_type]}**")
        st.image(cv2_to_pil(display_transformed), use_container_width=True)
else:
    st.subheader(f"Resultado: {transformation_names[transformation_type]}")
    st.image(cv2_to_pil(display_transformed), use_container_width=True)

# Mostrar matriz de transformación
if show_matrix:
    st.subheader("🔢 Matriz de Transformación")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if transformation_type == 'perspectiva':
            st.code(f"""
Matriz de Perspectiva (3x3):
{matrix}
            """)
        else:
            st.code(f"""
Matriz Afín (2x3):
{matrix}
            """)
    
    with col2:
        st.markdown("**💡 Explicación:**")
        st.info(explanation_text)

# Botón de descarga
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    processed_pil = cv2_to_pil(transformed)
    buf = io.BytesIO()
    processed_pil.save(buf, format='PNG')
    
    st.download_button(
        label="📥 Descargar Imagen Transformada",
        data=buf.getvalue(),
        file_name=f"transformed_{transformation_type}.png",
        mime="image/png",
        use_container_width=True
    )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("🔍 Sobre las Transformaciones"):
    st.markdown(f"""
    **📐 Traslación:**
    - Mueve la imagen sin rotarla
    - Matriz: [1,0,tx; 0,1,ty]
    - Preserva forma y orientación
    
    **🔄 Rotación:**
    - Gira alrededor de un punto
    - Usa trigonometría (cos, sin)
    - Preserva distancias
    
    **📏 Escalado:**
    - Cambia el tamaño
    - Puede ser no uniforme
    - Preserva paralelismo
    
    **📐 Cizallamiento:**
    - Inclina la imagen
    - Preserva área
    - Distorsiona ángulos
    
    **🎯 Afín Personalizada:**
    - Combinación de transformaciones
    - 6 grados de libertad
    - Preserva líneas paralelas
    
    **👁️ Perspectiva:**
    - 8 grados de libertad
    - Simula vista en 3D
    - No preserva paralelismo
    """)

with st.sidebar.expander("🧮 Conceptos Matemáticos"):
    st.markdown("""
    **Transformaciones Afines:**
    - Combinan rotación, escalado, cizallamiento y traslación
    - Se representan con matrices 2x3
    - Ecuación: [x', y'] = M × [x, y] + t
    
    **Transformaciones Perspectiva:**
    - Requieren matrices 3x3
    - División por coordenada homogénea
    - Ecuación: [x', y', w'] = M × [x, y, 1]
    
    **Propiedades Preservadas:**
    - Afines: líneas rectas, paralelismo, razones de áreas
    - Perspectiva: líneas rectas, puntos de fuga
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔄 <strong>Transformaciones Afines Interactivas</strong> | Capítulo 1 - Procesamiento de Imágenes</p>
        <p><small>Explora diferentes tipos de transformaciones geométricas de forma interactiva</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)