"""
Aplicación Streamlit - Detección de Características Interactiva
Aplicación educativa para explorar diferentes algoritmos de detección de características
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# Configuración de la página
st.set_page_config(
    page_title="Detección de Características Interactiva",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎯 Detección de Características Interactiva")
st.markdown("**Explora diferentes algoritmos de detección de keypoints y descriptores**")

# Funciones auxiliares
@st.cache_data
def load_sample_image():
    """Carga una imagen de ejemplo desde la carpeta images"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de imágenes disponibles (según el código original)
    sample_images = ['fishing_house.jpg', 'house.jpg', 'box.png', 'tool.png']
    
    for img_name in sample_images:
        img_path = os.path.join(script_dir, 'images', img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_name
    
    # Si no se encuentra ninguna imagen, crear una de ejemplo con características detectables
    img = np.ones((500, 700, 3), dtype=np.uint8) * 240
    
    # Crear patrones con muchas esquinas y características detectables
    # Edificios y estructuras geométricas
    cv2.rectangle(img, (50, 100), (200, 300), (120, 120, 120), -1)
    cv2.rectangle(img, (60, 110), (190, 290), (180, 180, 180), 3)
    cv2.rectangle(img, (70, 120), (180, 280), (80, 80, 80), 2)
    
    # Ventanas (esquinas detectables)
    for i in range(3):
        for j in range(5):
            x = 80 + j * 20
            y = 130 + i * 40
            cv2.rectangle(img, (x, y), (x + 15, y + 25), (50, 50, 50), -1)
            cv2.rectangle(img, (x + 2, y + 2), (x + 13, y + 23), (200, 200, 255), -1)
    
    # Casa con techo triangular
    cv2.rectangle(img, (250, 200), (400, 350), (100, 150, 100), -1)
    points = np.array([[250, 200], [325, 120], [400, 200]], np.int32)
    cv2.fillPoly(img, [points], (150, 100, 100))
    
    # Ventanas de la casa
    cv2.rectangle(img, (270, 230), (300, 280), (255, 255, 200), -1)
    cv2.rectangle(img, (320, 230), (350, 280), (255, 255, 200), -1)
    cv2.rectangle(img, (370, 230), (390, 280), (100, 50, 0), -1)  # Puerta
    
    # Árbol con muchas características
    cv2.circle(img, (500, 180), 60, (50, 120, 50), -1)  # Copa
    cv2.rectangle(img, (490, 240), (510, 320), (101, 67, 33), -1)  # Tronco
    
    # Agregar texturas y patrones
    for i in range(50):
        x = np.random.randint(0, 700)
        y = np.random.randint(0, 500)
        cv2.circle(img, (x, y), 2, (np.random.randint(0, 255), 
                                   np.random.randint(0, 255), 
                                   np.random.randint(0, 255)), -1)
    
    # Texto con esquinas detectables
    cv2.putText(img, 'FEATURE DETECTION', (200, 400), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    cv2.putText(img, 'Corners & Keypoints', (220, 430), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
    
    return img, "imagen_generada.jpg"

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

# Funciones de detección de características
def detect_sift_features(image, n_features=0, n_octave_layers=3, contrast_threshold=0.04, 
                        edge_threshold=10, sigma=1.6):
    """
    Detecta características SIFT (Scale-Invariant Feature Transform)
    """
    try:
        # Para versiones modernas de OpenCV
        sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    except:
        try:
            # Fallback para versiones anteriores
            sift = cv2.xfeatures2d.SIFT_create()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            return keypoints, descriptors
        except:
            return [], None

def detect_surf_features(image, hessian_threshold=400, n_octaves=4, n_octave_layers=3):
    """
    Detecta características SURF (Speeded-Up Robust Features)
    """
    try:
        surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=hessian_threshold,
            nOctaves=n_octaves,
            nOctaveLayers=n_octave_layers
        )
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = surf.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    except:
        # Fallback a SIFT si SURF no está disponible
        return detect_sift_features(image)

def detect_orb_features(image, n_features=500, scale_factor=1.2, n_levels=8, 
                       edge_threshold=31, first_level=0, wta_k=2, patch_size=31):
    """
    Detecta características ORB (Oriented FAST and Rotated BRIEF)
    """
    orb = cv2.ORB_create(
        nfeatures=n_features,
        scaleFactor=scale_factor,
        nlevels=n_levels,
        edgeThreshold=edge_threshold,
        firstLevel=first_level,
        WTA_K=wta_k,
        patchSize=patch_size
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def detect_fast_features(image, threshold=10, nonmax_suppression=True, detector_type=2):
    """
    Detecta esquinas FAST (Features from Accelerated Segment Test)
    """
    fast = cv2.FastFeatureDetector_create(
        threshold=threshold,
        nonmaxSuppression=nonmax_suppression,
        type=detector_type
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints = fast.detect(gray, None)
    
    # FAST no genera descriptores, solo keypoints
    return keypoints, None

def detect_harris_corners(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
    """
    Detecta esquinas Harris
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detectar esquinas Harris
    corners = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # Dilatar para marcar esquinas
    corners = cv2.dilate(corners, None)
    
    # Convertir a keypoints
    keypoints = []
    threshold_value = threshold * corners.max()
    
    corner_coords = np.where(corners > threshold_value)
    for y, x in zip(corner_coords[0], corner_coords[1]):
        keypoint = cv2.KeyPoint(float(x), float(y), 1)
        keypoints.append(keypoint)
    
    return keypoints, None

def detect_goodfeatures_to_track(image, max_corners=100, quality_level=0.01, 
                                min_distance=10, use_harris=False, k=0.04):
    """
    Detecta buenas características para tracking (Shi-Tomasi)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        useHarrisDetector=use_harris,
        k=k
    )
    
    # Convertir a keypoints
    keypoints = []
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            keypoint = cv2.KeyPoint(float(x), float(y), 1)
            keypoints.append(keypoint)
    
    return keypoints, None

def draw_keypoints(image, keypoints, detection_type='sift', color=None):
    """
    Dibuja keypoints en la imagen
    """
    result = image.copy()
    
    if not keypoints:
        return result
    
    # Colores específicos para cada tipo
    colors = {
        'sift': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        'surf': cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        'orb': cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        'fast': cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        'harris': cv2.DRAW_MATCHES_FLAGS_DEFAULT,
        'goodfeatures': cv2.DRAW_MATCHES_FLAGS_DEFAULT
    }
    
    if detection_type in ['sift', 'surf']:
        # SIFT y SURF soportan keypoints ricos
        cv2.drawKeypoints(result, keypoints, result, 
                         flags=colors[detection_type])
    else:
        # Otros detectores usan keypoints simples
        cv2.drawKeypoints(result, keypoints, result, 
                         color=(0, 255, 0) if color is None else color,
                         flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    return result

def compare_detectors(image, detectors_config):
    """
    Compara múltiples detectores en la misma imagen
    """
    results = {}
    
    for detector_name, config in detectors_config.items():
        if detector_name == 'sift':
            keypoints, descriptors = detect_sift_features(image, **config)
        elif detector_name == 'surf':
            keypoints, descriptors = detect_surf_features(image, **config)
        elif detector_name == 'orb':
            keypoints, descriptors = detect_orb_features(image, **config)
        elif detector_name == 'fast':
            keypoints, descriptors = detect_fast_features(image, **config)
        elif detector_name == 'harris':
            keypoints, descriptors = detect_harris_corners(image, **config)
        elif detector_name == 'goodfeatures':
            keypoints, descriptors = detect_goodfeatures_to_track(image, **config)
        
        result_img = draw_keypoints(image, keypoints, detector_name)
        results[detector_name] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'image': result_img,
            'count': len(keypoints)
        }
    
    return results

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

# Menú de detección
st.sidebar.subheader("🔍 Algoritmo de Detección")

detection_algorithm = st.sidebar.selectbox(
    "Selecciona el algoritmo:",
    options=['sift', 'surf', 'orb', 'fast', 'harris', 'goodfeatures', 'comparison'],
    format_func=lambda x: {
        'sift': '🎯 SIFT (Scale-Invariant)',
        'surf': '🌊 SURF (Speeded-Up Robust)',
        'orb': '⭕ ORB (Oriented FAST)',
        'fast': '⚡ FAST (Features Accelerated)',
        'harris': '📐 Harris Corners',
        'goodfeatures': '✨ Good Features (Shi-Tomasi)',
        'comparison': '🆚 Comparación Múltiple'
    }[x],
    help="Selecciona el algoritmo de detección de características"
)

st.sidebar.markdown("---")

# Parámetros específicos según el algoritmo
if detection_algorithm == 'sift':
    st.sidebar.subheader("🎯 Parámetros SIFT")
    
    n_features = st.sidebar.number_input("Máximo características:", 0, 2000, 0, 50)
    n_octave_layers = st.sidebar.slider("Capas por octava:", 1, 10, 3)
    contrast_threshold = st.sidebar.slider("Umbral contraste:", 0.01, 0.2, 0.04, 0.001)
    edge_threshold = st.sidebar.slider("Umbral borde:", 5, 50, 10)
    sigma = st.sidebar.slider("Sigma Gaussiano:", 0.5, 3.0, 1.6, 0.1)
    
    keypoints, descriptors = detect_sift_features(
        image, n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma
    )

elif detection_algorithm == 'surf':
    st.sidebar.subheader("🌊 Parámetros SURF")
    
    hessian_threshold = st.sidebar.slider("Umbral Hessiano:", 100, 1000, 400, 50)
    n_octaves = st.sidebar.slider("Número octavas:", 2, 8, 4)
    n_octave_layers = st.sidebar.slider("Capas por octava:", 1, 6, 3)
    
    keypoints, descriptors = detect_surf_features(
        image, hessian_threshold, n_octaves, n_octave_layers
    )

elif detection_algorithm == 'orb':
    st.sidebar.subheader("⭕ Parámetros ORB")
    
    n_features = st.sidebar.slider("Máximo características:", 100, 2000, 500, 50)
    scale_factor = st.sidebar.slider("Factor escala:", 1.1, 2.0, 1.2, 0.1)
    n_levels = st.sidebar.slider("Niveles pirámide:", 4, 16, 8)
    edge_threshold = st.sidebar.slider("Umbral borde:", 10, 100, 31)
    patch_size = st.sidebar.slider("Tamaño patch:", 15, 63, 31, 2)
    
    keypoints, descriptors = detect_orb_features(
        image, n_features, scale_factor, n_levels, edge_threshold, 0, 2, patch_size
    )

elif detection_algorithm == 'fast':
    st.sidebar.subheader("⚡ Parámetros FAST")
    
    threshold = st.sidebar.slider("Umbral:", 1, 50, 10)
    nonmax_suppression = st.sidebar.checkbox("Supresión no-máximos", True)
    detector_type = st.sidebar.selectbox("Tipo detector:", [0, 1, 2], 
                                       format_func=lambda x: ["TYPE_9_16", "TYPE_7_12", "TYPE_5_8"][x])
    
    keypoints, descriptors = detect_fast_features(
        image, threshold, nonmax_suppression, detector_type
    )

elif detection_algorithm == 'harris':
    st.sidebar.subheader("📐 Parámetros Harris")
    
    block_size = st.sidebar.slider("Tamaño bloque:", 2, 10, 2)
    ksize = st.sidebar.slider("Tamaño kernel:", 3, 31, 3, 2)
    k = st.sidebar.slider("Parámetro k:", 0.01, 0.1, 0.04, 0.01)
    threshold = st.sidebar.slider("Umbral respuesta:", 0.001, 0.1, 0.01, 0.001)
    
    keypoints, descriptors = detect_harris_corners(
        image, block_size, ksize, k, threshold
    )

elif detection_algorithm == 'goodfeatures':
    st.sidebar.subheader("✨ Parámetros Good Features")
    
    max_corners = st.sidebar.slider("Máximo esquinas:", 10, 500, 100, 10)
    quality_level = st.sidebar.slider("Nivel calidad:", 0.001, 0.1, 0.01, 0.001)
    min_distance = st.sidebar.slider("Distancia mínima:", 5, 50, 10)
    use_harris = st.sidebar.checkbox("Usar detector Harris", False)
    if use_harris:
        k = st.sidebar.slider("Parámetro k Harris:", 0.01, 0.1, 0.04, 0.01)
    else:
        k = 0.04
    
    keypoints, descriptors = detect_goodfeatures_to_track(
        image, max_corners, quality_level, min_distance, use_harris, k
    )

elif detection_algorithm == 'comparison':
    st.sidebar.subheader("🆚 Comparación Múltiple")
    
    selected_algorithms = st.sidebar.multiselect(
        "Algoritmos a comparar:",
        options=['sift', 'surf', 'orb', 'fast', 'harris'],
        default=['sift', 'orb', 'fast'],
        format_func=lambda x: {
            'sift': '🎯 SIFT',
            'surf': '🌊 SURF', 
            'orb': '⭕ ORB',
            'fast': '⚡ FAST',
            'harris': '📐 Harris'
        }[x]
    )

# Opciones de visualización
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Opciones de Vista")

show_comparison = st.sidebar.checkbox("Comparación Lado a Lado", True)
show_statistics = st.sidebar.checkbox("Mostrar Estadísticas", True)

# Procesamiento y visualización
algorithm_names = {
    'sift': '🎯 SIFT (Scale-Invariant Feature Transform)',
    'surf': '🌊 SURF (Speeded-Up Robust Features)',
    'orb': '⭕ ORB (Oriented FAST and Rotated BRIEF)',
    'fast': '⚡ FAST (Features from Accelerated Segment Test)',
    'harris': '📐 Harris Corner Detection',
    'goodfeatures': '✨ Good Features to Track (Shi-Tomasi)'
}

if detection_algorithm != 'comparison':
    # Detección individual
    processed_image = draw_keypoints(image, keypoints, detection_algorithm)
    
    if show_comparison:
        st.subheader(f"{algorithm_names[detection_algorithm]} - Comparación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🖼️ Imagen Original**")
            st.image(cv2_to_pil(image), use_container_width=True)
        
        with col2:
            st.write(f"**{algorithm_names[detection_algorithm]}**")
            st.image(cv2_to_pil(processed_image), use_container_width=True)
    else:
        st.subheader(f"Resultado: {algorithm_names[detection_algorithm]}")
        st.image(cv2_to_pil(processed_image), use_container_width=True)
    
    # Estadísticas individuales
    if show_statistics:
        st.subheader("📊 Estadísticas de Detección")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Keypoints Detectados", len(keypoints))
        
        with col2:
            if descriptors is not None:
                st.metric("Dimensión Descriptores", f"{descriptors.shape[1]}D" if len(descriptors) > 0 else "N/A")
            else:
                st.metric("Descriptores", "No disponibles")
        
        with col3:
            if keypoints:
                avg_response = np.mean([kp.response for kp in keypoints])
                st.metric("Respuesta Promedio", f"{avg_response:.3f}")
        
        with col4:
            density = len(keypoints) / (w * h) * 10000
            st.metric("Densidad", f"{density:.2f}/10k px")

else:
    # Comparación múltiple
    if selected_algorithms:
        st.subheader("🆚 Comparación de Algoritmos de Detección")
        
        # Configuraciones por defecto para comparación
        configs = {
            'sift': {},
            'surf': {},
            'orb': {'n_features': 500},
            'fast': {'threshold': 10},
            'harris': {'threshold': 0.01}
        }
        
        selected_configs = {alg: configs[alg] for alg in selected_algorithms}
        results = compare_detectors(image, selected_configs)
        
        # Layout en grid
        cols_per_row = 2
        rows = (len(selected_algorithms) + cols_per_row - 1) // cols_per_row
        
        for i in range(0, len(selected_algorithms), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, alg in enumerate(selected_algorithms[i:i+cols_per_row]):
                with cols[j]:
                    st.write(f"**{algorithm_names[alg]}**")
                    st.image(cv2_to_pil(results[alg]['image']), use_container_width=True)
                    st.caption(f"Keypoints: {results[alg]['count']}")
        
        # Estadísticas comparativas
        if show_statistics:
            st.subheader("📈 Estadísticas Comparativas")
            
            # Crear tabla comparativa
            comparison_data = []
            for alg in selected_algorithms:
                result = results[alg]
                comparison_data.append({
                    'Algoritmo': algorithm_names[alg].split(' (')[0],
                    'Keypoints': result['count'],
                    'Descriptores': 'Sí' if result['descriptors'] is not None else 'No',
                    'Densidad (keypoints/10k px)': f"{result['count'] / (w * h) * 10000:.2f}"
                })
            
            st.table(comparison_data)

# Botón de descarga
if detection_algorithm != 'comparison':
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        processed_pil = cv2_to_pil(processed_image)
        buf = io.BytesIO()
        processed_pil.save(buf, format='PNG')
        
        st.download_button(
            label="📥 Descargar Imagen con Características",
            data=buf.getvalue(),
            file_name=f"features_{detection_algorithm}.png",
            mime="image/png",
            use_container_width=True
        )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("🎯 Sobre los Algoritmos"):
    st.markdown("""
    **🎯 SIFT:**
    - Invariante a escala y rotación
    - Robusto a cambios de iluminación
    - Descriptores de 128 dimensiones
    - Patentado (uso comercial limitado)
    
    **🌊 SURF:**
    - Versión acelerada de SIFT
    - Usa aproximación de Hessiano
    - Más rápido que SIFT
    - También patentado
    
    **⭕ ORB:**
    - Alternativa libre a SIFT/SURF
    - Combina FAST + BRIEF
    - Orientación añadida
    - Muy rápido y eficiente
    
    **⚡ FAST:**
    - Solo detección de esquinas
    - Extremadamente rápido
    - No genera descriptores
    - Ideal para tiempo real
    
    **📐 Harris:**
    - Detector clásico de esquinas
    - Basado en gradientes
    - Robusto y confiable
    - No invariante a escala
    """)

with st.sidebar.expander("🔧 Parámetros Clave"):
    st.markdown("""
    **SIFT/SURF:**
    - **Octavas**: Escalas diferentes
    - **Umbral contraste**: Sensibilidad
    - **Umbral borde**: Filtrado de bordes
    
    **ORB:**
    - **Níveis pirámide**: Escalas múltiples
    - **Factor escala**: Reducción entre niveles
    - **Patch size**: Tamaño descriptor
    
    **FAST:**
    - **Umbral**: Diferencia intensidad
    - **Supresión no-máximos**: Refinamiento
    
    **Harris:**
    - **Bloque**: Ventana análisis
    - **k**: Sensibilidad esquinas vs bordes
    """)

with st.sidebar.expander("💡 Casos de Uso"):
    st.markdown("""
    **Matching de imágenes:**
    - SIFT/SURF para alta calidad
    - ORB para velocidad
    
    **Tracking de objetos:**
    - FAST para tiempo real
    - Good Features para estabilidad
    
    **Reconocimiento:**
    - SIFT para robustez
    - ORB para eficiencia
    
    **Calibración cámaras:**
    - Harris para esquinas precisas
    - Good Features para patrones
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎯 <strong>Detección de Características Interactiva</strong> | Capítulo 5 - Feature Detection</p>
        <p><small>Explora diferentes algoritmos de detección de keypoints y descriptores</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)