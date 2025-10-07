"""
Aplicación Streamlit - Detección Facial y Características Interactiva
Aplicación educativa para explorar detección de rostros, ojos, nariz, boca y orejas
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# Configuración de la página
st.set_page_config(
    page_title="Detección Facial Interactiva",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("👤 Detección Facial y Características Interactiva")
st.markdown("**Explora diferentes tipos de detección facial usando Haar Cascades**")

# Funciones auxiliares
@st.cache_resource
def load_cascade_classifiers():
    """Carga todos los clasificadores Haar Cascade disponibles"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_dir = os.path.join(script_dir, 'cascade_files')
    
    classifiers = {}
    cascade_files = {
        'face': 'haarcascade_frontalface_alt.xml',
        'eye': 'haarcascade_eye.xml',
        'nose': 'haarcascade_mcs_nose.xml',
        'mouth': 'haarcascade_mcs_mouth.xml',
        'left_ear': 'haarcascade_mcs_leftear.xml',
        'right_ear': 'haarcascade_mcs_rightear.xml'
    }
    
    for feature, filename in cascade_files.items():
        cascade_path = os.path.join(cascade_dir, filename)
        if os.path.exists(cascade_path):
            classifiers[feature] = cv2.CascadeClassifier(cascade_path)
        else:
            # Fallback a clasificador frontal por defecto
            classifiers[feature] = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    return classifiers

@st.cache_resource
def load_overlay_images():
    """Carga las imágenes de overlay (máscaras, gafas, etc.)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'images')
    
    overlays = {}
    overlay_files = {
        'sunglasses': 'sunglasses.png',
        'moustache': 'moustache.png',
        'mask': 'mask_hannibal.png'
    }
    
    for name, filename in overlay_files.items():
        overlay_path = os.path.join(images_dir, filename)
        if os.path.exists(overlay_path):
            img = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                overlays[name] = img
    
    return overlays

@st.cache_data
def create_sample_face_image():
    """Crea una imagen de ejemplo con caras sintéticas"""
    img = np.ones((600, 800, 3), dtype=np.uint8) * 240
    
    # Simular caras con formas geométricas
    # Cara 1
    cv2.ellipse(img, (200, 200), (80, 100), 0, 0, 360, (220, 200, 180), -1)  # Cara
    cv2.ellipse(img, (180, 180), (15, 20), 0, 0, 360, (50, 50, 50), -1)      # Ojo izq
    cv2.ellipse(img, (220, 180), (15, 20), 0, 0, 360, (50, 50, 50), -1)      # Ojo der
    cv2.ellipse(img, (200, 200), (8, 12), 0, 0, 360, (100, 80, 70), -1)      # Nariz
    cv2.ellipse(img, (200, 230), (20, 10), 0, 0, 360, (150, 100, 100), -1)   # Boca
    
    # Cara 2
    cv2.ellipse(img, (600, 200), (70, 90), 0, 0, 360, (210, 190, 170), -1)   # Cara
    cv2.ellipse(img, (585, 180), (12, 18), 0, 0, 360, (40, 40, 40), -1)      # Ojo izq
    cv2.ellipse(img, (615, 180), (12, 18), 0, 0, 360, (40, 40, 40), -1)      # Ojo der
    cv2.ellipse(img, (600, 200), (6, 10), 0, 0, 360, (90, 70, 60), -1)       # Nariz
    cv2.ellipse(img, (600, 225), (18, 8), 0, 0, 360, (140, 90, 90), -1)      # Boca
    
    # Cara 3 (perfil)
    cv2.ellipse(img, (400, 400), (60, 80), 15, 0, 180, (200, 180, 160), -1)  # Media cara
    cv2.ellipse(img, (420, 380), (10, 15), 0, 0, 360, (30, 30, 30), -1)      # Ojo
    cv2.ellipse(img, (435, 400), (5, 8), 0, 0, 360, (80, 60, 50), -1)        # Nariz
    cv2.ellipse(img, (425, 420), (12, 6), 0, 0, 360, (130, 80, 80), -1)      # Boca
    
    # Agregar texto
    cv2.putText(img, 'DETECCION FACIAL', (250, 500), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 2)
    cv2.putText(img, 'Caras de Ejemplo', (290, 530), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
    
    return img

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def detect_features(image, classifiers, detection_type, scale_factor=1.1, 
                   min_neighbors=3, min_size=(30, 30), max_size=()):
    """
    Detecta características faciales usando Haar Cascades
    
    Args:
        image: Imagen de entrada
        classifiers: Diccionario de clasificadores
        detection_type: Tipo de detección ('face', 'eye', 'nose', etc.)
        scale_factor: Factor de escala para detección
        min_neighbors: Mínimo número de vecinos
        min_size: Tamaño mínimo de detección
        max_size: Tamaño máximo de detección
    
    Returns:
        Lista de rectángulos detectados
    """
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if detection_type in classifiers:
        classifier = classifiers[detection_type]
        
        detections = classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            maxSize=max_size if max_size else ()
        )
        
        return detections
    
    return []

def draw_detections(image, detections, color=(0, 255, 0), thickness=2, 
                   detection_type='face', show_confidence=False):
    """Dibuja rectángulos alrededor de las detecciones"""
    result = image.copy()
    
    colors = {
        'face': (0, 255, 0),      # Verde
        'eye': (255, 0, 0),       # Rojo
        'nose': (0, 0, 255),      # Azul
        'mouth': (255, 0, 255),   # Magenta
        'left_ear': (0, 255, 255), # Cyan
        'right_ear': (255, 255, 0) # Amarillo
    }
    
    detection_color = colors.get(detection_type, color)
    
    for i, (x, y, w, h) in enumerate(detections):
        cv2.rectangle(result, (x, y), (x + w, y + h), detection_color, thickness)
        
        # Agregar etiqueta
        label = f"{detection_type.title()} {i+1}"
        cv2.putText(result, label, (x, y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, detection_color, 1)
    
    return result

def apply_overlay(image, detections, overlay_img, detection_type='face', 
                 scale_factor=1.0, offset_y=0):
    """Aplica overlay (gafas, bigote, etc.) sobre las detecciones"""
    if overlay_img is None or len(detections) == 0:
        return image
    
    result = image.copy()
    
    for (x, y, w, h) in detections:
        # Redimensionar overlay según el tamaño de la detección
        overlay_width = int(w * scale_factor)
        overlay_height = int(overlay_img.shape[0] * (overlay_width / overlay_img.shape[1]))
        
        # Posición del overlay
        overlay_x = x + int((w - overlay_width) / 2)
        overlay_y = y + offset_y
        
        # Redimensionar overlay
        overlay_resized = cv2.resize(overlay_img, (overlay_width, overlay_height))
        
        # Verificar límites
        if (overlay_x >= 0 and overlay_y >= 0 and 
            overlay_x + overlay_width <= result.shape[1] and 
            overlay_y + overlay_height <= result.shape[0]):
            
            # Aplicar overlay con canal alpha si está disponible
            if overlay_resized.shape[2] == 4:  # Con canal alpha
                alpha = overlay_resized[:, :, 3] / 255.0
                for c in range(3):
                    result[overlay_y:overlay_y + overlay_height, 
                          overlay_x:overlay_x + overlay_width, c] = \
                        alpha * overlay_resized[:, :, c] + \
                        (1 - alpha) * result[overlay_y:overlay_y + overlay_height, 
                                           overlay_x:overlay_x + overlay_width, c]
            else:  # Sin canal alpha
                result[overlay_y:overlay_y + overlay_height, 
                      overlay_x:overlay_x + overlay_width] = overlay_resized
    
    return result

def detect_multiple_features(image, classifiers, selected_features, 
                           scale_factor, min_neighbors, min_size):
    """Detecta múltiples características en una sola pasada"""
    result = image.copy()
    all_detections = {}
    
    for feature in selected_features:
        detections = detect_features(image, classifiers, feature, 
                                   scale_factor, min_neighbors, min_size)
        all_detections[feature] = detections
        result = draw_detections(result, detections, detection_type=feature)
    
    return result, all_detections

# Cargar recursos
classifiers = load_cascade_classifiers()
overlays = load_overlay_images()

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
    image = create_sample_face_image()
    image_source = "🖼️ Imagen de ejemplo generada"

# Información de la imagen
h, w = image.shape[:2]
st.sidebar.info(f"**Imagen actual:** {image_source}")
st.sidebar.info(f"**Dimensiones:** {w} x {h} píxeles")

st.sidebar.markdown("---")

# Menú de detección
st.sidebar.subheader("🔍 Tipo de Detección")

detection_mode = st.sidebar.selectbox(
    "Modo de detección:",
    options=['single', 'multiple', 'overlay_mode'],
    format_func=lambda x: {
        'single': '🎯 Detección Individual',
        'multiple': '👥 Detección Múltiple',
        'overlay_mode': '🎭 Modo Overlay'
    }[x],
    help="Selecciona el modo de detección"
)

if detection_mode == 'single':
    detection_type = st.sidebar.selectbox(
        "Característica a detectar:",
        options=['face', 'eye', 'nose', 'mouth', 'left_ear', 'right_ear'],
        format_func=lambda x: {
            'face': '👤 Rostro',
            'eye': '👁️ Ojos',
            'nose': '👃 Nariz',
            'mouth': '👄 Boca',
            'left_ear': '👂 Oreja Izquierda',
            'right_ear': '👂 Oreja Derecha'
        }[x]
    )
elif detection_mode == 'multiple':
    selected_features = st.sidebar.multiselect(
        "Características a detectar:",
        options=['face', 'eye', 'nose', 'mouth', 'left_ear', 'right_ear'],
        default=['face', 'eye'],
        format_func=lambda x: {
            'face': '👤 Rostro',
            'eye': '👁️ Ojos',
            'nose': '👃 Nariz',
            'mouth': '👄 Boca',
            'left_ear': '👂 Oreja Izquierda',
            'right_ear': '👂 Oreja Derecha'
        }[x]
    )
else:  # overlay_mode
    overlay_type = st.sidebar.selectbox(
        "Tipo de overlay:",
        options=['sunglasses', 'moustache', 'mask'],
        format_func=lambda x: {
            'sunglasses': '🕶️ Gafas de Sol',
            'moustache': '🥸 Bigote',
            'mask': '🎭 Máscara'
        }[x]
    )

st.sidebar.markdown("---")

# Parámetros de detección
st.sidebar.subheader("🎛️ Parámetros de Detección")

scale_factor = st.sidebar.slider(
    "Factor de escala:", 
    1.01, 2.0, 1.1, 0.01,
    help="Qué tan rápido se reduce el tamaño de imagen en cada escala"
)

min_neighbors = st.sidebar.slider(
    "Mínimos vecinos:", 
    1, 10, 3, 1,
    help="Cuántos vecinos debe tener cada rectángulo candidato para ser válido"
)

min_size = st.sidebar.slider(
    "Tamaño mínimo:", 
    10, 200, 30, 5,
    help="Tamaño mínimo de la característica a detectar (píxeles)"
)

max_size = st.sidebar.slider(
    "Tamaño máximo:", 
    50, 500, 300, 10,
    help="Tamaño máximo de la característica a detectar (0 = sin límite)"
)

if detection_mode == 'overlay_mode':
    st.sidebar.subheader("🎨 Parámetros de Overlay")
    
    overlay_scale = st.sidebar.slider(
        "Escala del overlay:", 
        0.5, 2.0, 1.0, 0.1,
        help="Escala del elemento overlay respecto a la detección"
    )
    
    overlay_offset_y = st.sidebar.slider(
        "Desplazamiento Y:", 
        -100, 100, 0, 5,
        help="Desplazamiento vertical del overlay"
    )

# Opciones de visualización
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Opciones de Vista")

show_comparison = st.sidebar.checkbox("Comparación Lado a Lado", True)
show_statistics = st.sidebar.checkbox("Mostrar Estadísticas", True)

# Procesamiento según el modo
max_size_tuple = (max_size, max_size) if max_size > 0 else ()

if detection_mode == 'single':
    # Detección individual
    detections = detect_features(image, classifiers, detection_type, 
                               scale_factor, min_neighbors, 
                               (min_size, min_size), max_size_tuple)
    
    processed_image = draw_detections(image, detections, detection_type=detection_type)
    
    detection_title = {
        'face': '👤 Detección de Rostros',
        'eye': '👁️ Detección de Ojos',
        'nose': '👃 Detección de Nariz',
        'mouth': '👄 Detección de Boca',
        'left_ear': '👂 Detección Oreja Izquierda',
        'right_ear': '👂 Detección Oreja Derecha'
    }[detection_type]

elif detection_mode == 'multiple':
    # Detección múltiple
    processed_image, all_detections = detect_multiple_features(
        image, classifiers, selected_features, 
        scale_factor, min_neighbors, (min_size, min_size)
    )
    
    detection_title = "👥 Detección Múltiple de Características"

else:  # overlay_mode
    # Modo overlay
    face_detections = detect_features(image, classifiers, 'face', 
                                    scale_factor, min_neighbors, 
                                    (min_size, min_size), max_size_tuple)
    
    overlay_img = overlays.get(overlay_type)
    processed_image = apply_overlay(image, face_detections, overlay_img, 
                                  'face', overlay_scale, overlay_offset_y)
    
    overlay_names = {'sunglasses': '🕶️ Gafas', 'moustache': '🥸 Bigote', 'mask': '🎭 Máscara'}
    detection_title = f"🎭 Overlay: {overlay_names[overlay_type]}"

# Área principal
if show_comparison:
    st.subheader(f"{detection_title} - Comparación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🖼️ Imagen Original**")
        st.image(cv2_to_pil(image), use_container_width=True)
    
    with col2:
        st.write(f"**{detection_title}**")
        st.image(cv2_to_pil(processed_image), use_container_width=True)
else:
    st.subheader(detection_title)
    st.image(cv2_to_pil(processed_image), use_container_width=True)

# Estadísticas
if show_statistics and detection_mode != 'overlay_mode':
    st.subheader("📊 Estadísticas de Detección")
    
    if detection_mode == 'single':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Detecciones Encontradas", len(detections))
        with col2:
            if len(detections) > 0:
                avg_size = np.mean([w*h for (x,y,w,h) in detections])
                st.metric("Tamaño Promedio", f"{avg_size:.0f} px²")
        with col3:
            confidence_score = len(detections) / max(1, (w*h) // 10000)
            st.metric("Puntuación Confianza", f"{confidence_score:.2f}")
    
    elif detection_mode == 'multiple':
        cols = st.columns(len(selected_features))
        for i, feature in enumerate(selected_features):
            with cols[i]:
                count = len(all_detections.get(feature, []))
                feature_name = {
                    'face': 'Rostros', 'eye': 'Ojos', 'nose': 'Narices',
                    'mouth': 'Bocas', 'left_ear': 'Orejas Izq', 'right_ear': 'Orejas Der'
                }[feature]
                st.metric(feature_name, count)

# Botón de descarga
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    processed_pil = cv2_to_pil(processed_image)
    buf = io.BytesIO()
    processed_pil.save(buf, format='PNG')
    
    st.download_button(
        label="📥 Descargar Imagen Procesada",
        data=buf.getvalue(),
        file_name=f"face_detection_{detection_mode}.png",
        mime="image/png",
        use_container_width=True
    )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("🔍 Sobre Haar Cascades"):
    st.markdown("""
    **¿Qué son los Haar Cascades?**
    - Clasificadores entrenados con características Haar
    - Detectan patrones específicos en imágenes
    - Funcionan bien para detección facial
    
    **Parámetros Clave:**
    - **Scale Factor**: Velocidad de búsqueda multi-escala
    - **Min Neighbors**: Robustez vs sensibilidad
    - **Min/Max Size**: Rango de tamaños a detectar
    
    **Ventajas:**
    - Rápidos y eficientes
    - No requieren GPU
    - Funcionan bien en tiempo real
    
    **Limitaciones:**
    - Sensibles a orientación
    - Menos precisos que deep learning
    - Requieren buenas condiciones de iluminación
    """)

with st.sidebar.expander("🎭 Overlays y Efectos"):
    st.markdown("""
    **Tipos de Overlay:**
    - **🕶️ Gafas de Sol**: Se colocan sobre los ojos
    - **🥸 Bigote**: Se posiciona bajo la nariz
    - **🎭 Máscara**: Cubre parcialmente el rostro
    
    **Técnica:**
    - Detección de rostros primero
    - Redimensionamiento proporcional
    - Aplicación con canal alpha (transparencia)
    - Posicionamiento relativo a la detección
    """)

with st.sidebar.expander("⚙️ Algoritmos Utilizados"):
    st.markdown("""
    **OpenCV Functions:**
    - `cv2.CascadeClassifier()`: Carga clasificador
    - `detectMultiScale()`: Detección multi-escala
    - `cv2.rectangle()`: Dibujo de rectángulos
    - `cv2.resize()`: Redimensionamiento
    - `cv2.cvtColor()`: Conversión de color
    
    **Flujo del Algoritmo:**
    1. Conversión a escala de grises
    2. Búsqueda multi-escala de patrones
    3. Filtrado por vecinos mínimos
    4. Eliminación de duplicados
    5. Dibujo de resultados
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>👤 <strong>Detección Facial Interactiva</strong> | Capítulo 4 - Detección de Características Faciales</p>
        <p><small>Explora diferentes técnicas de detección facial usando Haar Cascades</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)