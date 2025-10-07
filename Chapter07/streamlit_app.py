"""
Aplicación Streamlit - Segmentación Watershed Interactiva
Aplicación educativa para explorar técnicas de segmentación de imágenes usando Watershed
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# Imports opcionales - usar fallbacks si no están disponibles
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from skimage.segmentation import watershed as watershed_sklearn
    from skimage.feature import peak_local_maxima
    from skimage.measure import label
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Segmentación Watershed Interactiva",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("💧 Segmentación Watershed Interactiva")
st.markdown("**Explora técnicas avanzadas de segmentación de imágenes usando el algoritmo Watershed**")

# Funciones auxiliares
@st.cache_data
def load_sample_image():
    """Carga una imagen de ejemplo desde la carpeta images"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de imágenes disponibles (según el código original)
    sample_images = ['shapes.png', 'random_shapes.png', 'convex_shapes.png', 
                    'boomerang.png', 'road.jpg', 'hand_pen.jpg']
    
    for img_name in sample_images:
        img_path = os.path.join(script_dir, 'images', img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_name
    
    # Si no se encuentra ninguna imagen, crear una de ejemplo con formas separables
    img = np.ones((500, 600, 3), dtype=np.uint8) * 255
    
    # Crear formas geométricas que se puedan segmentar
    # Círculos de diferentes tamaños
    cv2.circle(img, (150, 150), 60, (100, 100, 100), -1)
    cv2.circle(img, (450, 150), 50, (120, 120, 120), -1)
    cv2.circle(img, (300, 300), 70, (80, 80, 80), -1)
    
    # Rectángulos
    cv2.rectangle(img, (50, 350), (150, 450), (90, 90, 90), -1)
    cv2.rectangle(img, (450, 350), (550, 450), (110, 110, 110), -1)
    
    # Elipses
    cv2.ellipse(img, (200, 400), (40, 60), 45, 0, 360, (95, 95, 95), -1)
    cv2.ellipse(img, (400, 250), (50, 30), -30, 0, 360, (105, 105, 105), -1)
    
    # Formas irregulares (polígonos)
    points1 = np.array([[250, 100], [300, 80], [350, 120], [320, 180], [280, 170]], np.int32)
    cv2.fillPoly(img, [points1], (85, 85, 85))
    
    points2 = np.array([[100, 250], [180, 230], [160, 300], [120, 320], [80, 280]], np.int32)
    cv2.fillPoly(img, [points2], (115, 115, 115))
    
    # Agregar ruido sutil
    noise = np.random.randint(-5, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img, "formas_geometricas.png"

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    cv2_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return np.asarray(cv2_image, dtype=np.uint8)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    # Asegurar tipo y formato correcto
    cv2_image = np.asarray(cv2_image, dtype=np.uint8)
    if len(cv2_image.shape) == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2_image
    return Image.fromarray(rgb_image)

# Funciones de segmentación Watershed
def preprocess_image(img, method='otsu', kernel_size=3, iterations_opening=2, iterations_dilation=3):
    """
    Preprocesamiento de imagen para Watershed (basado en código original)
    """
    # Asegurar que la imagen esté en formato correcto
    img = np.asarray(img, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización según el método
    if method == 'otsu':
        # Método original del código
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == 'adaptive':
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
    elif method == 'manual':
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Eliminación de ruido con operaciones morfológicas
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=iterations_opening)
    
    # Área de fondo segura
    sure_bg = cv2.dilate(opening, kernel, iterations=iterations_dilation)
    
    # Asegurar que todos los arrays tengan el tipo correcto
    return gray.astype(np.uint8), thresh.astype(np.uint8), opening.astype(np.uint8), sure_bg.astype(np.uint8)

def find_foreground_markers(opening, fg_method='distance_transform', threshold_factor=0.7):
    """
    Encuentra marcadores de primer plano
    """
    # Asegurar que opening esté en el formato correcto
    opening = np.asarray(opening, dtype=np.uint8)
    
    if fg_method == 'distance_transform':
        # Método original del código
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, threshold_factor * dist_transform.max(), 255, 0)
        return dist_transform, sure_fg.astype(np.uint8)
    
    elif fg_method == 'erosion':
        # Alternativa usando erosión
        kernel = np.ones((3, 3), np.uint8)
        sure_fg = cv2.erode(opening, kernel, iterations=3)
        # Crear transformada de distancia para compatibilidad
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        return dist_transform, sure_fg.astype(np.uint8)
    
    elif fg_method == 'contours':
        # Usando contornos para encontrar centros
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sure_fg = np.zeros_like(opening)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filtrar contornos pequeños
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(sure_fg, (cx, cy), 10, 255, -1)
        # Crear transformada de distancia para compatibilidad
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        return dist_transform, sure_fg.astype(np.uint8)
    else:
        # Fallback por defecto
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, threshold_factor * dist_transform.max(), 255, 0)
        return dist_transform, sure_fg.astype(np.uint8)

def apply_watershed_opencv(img, sure_bg, sure_fg, connectivity=8):
    """
    Aplica algoritmo Watershed usando OpenCV (método original)
    """
    # Asegurar tipos correctos para todos los inputs
    img = np.asarray(img, dtype=np.uint8)
    sure_fg = np.asarray(sure_fg, dtype=np.uint8)
    sure_bg = np.asarray(sure_bg, dtype=np.uint8)
    
    # Encontrar región desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Etiquetado de marcadores
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Agregar 1 a todas las etiquetas para que el fondo seguro no sea 0, sino 1
    markers = markers + 1
    
    # Marcar región desconocida con cero
    markers[unknown == 255] = 0
    
    # Aplicar Watershed
    markers = cv2.watershed(img, markers)
    
    # Crear imagen de resultado
    result = img.copy()
    result[markers == -1] = [0, 0, 255]  # Bordes en rojo
    
    return markers, unknown, result

def apply_watershed_sklearn(img, distance_transform, min_distance=20):
    """
    Aplica Watershed usando scikit-image (método alternativo) o fallback a OpenCV
    """
    if not SKIMAGE_AVAILABLE:
        # Fallback a método OpenCV si sklearn no está disponible
        sure_bg = np.ones_like(img[:,:,0]) * 255
        sure_fg = (distance_transform > 0.5 * distance_transform.max()).astype(np.uint8) * 255
        markers, unknown, result = apply_watershed_opencv(img, sure_bg, sure_fg)
        return markers, result
    
    try:
        # Encontrar picos locales en la transformada de distancia
        local_maxima = peak_local_maxima(distance_transform, min_distance=min_distance, 
                                       threshold_abs=0.3 * distance_transform.max())
        
        # Crear marcadores
        markers = np.zeros_like(distance_transform, dtype=int)
        markers[tuple(local_maxima.T)] = np.arange(1, len(local_maxima) + 1)
        
        # Aplicar Watershed
        labels = watershed_sklearn(-distance_transform, markers, mask=distance_transform > 0)
        
        # Crear imagen de resultado
        result = img.copy()
        
        # Colorear cada segmento
        colors = np.random.randint(0, 255, (len(local_maxima) + 1, 3))
        for i in range(1, len(local_maxima) + 1):
            result[labels == i] = colors[i]
        
        return labels, result
    except Exception:
        # Fallback a método OpenCV si sklearn falla
        sure_bg = np.ones_like(img[:,:,0]) * 255
        sure_fg = (distance_transform > 0.5 * distance_transform.max()).astype(np.uint8) * 255
        markers, unknown, result = apply_watershed_opencv(img, sure_bg, sure_fg)
        return markers, result

def create_colored_segmentation(markers, img):
    """
    Crea una visualización coloreada de la segmentación
    """
    # Asegurar tipos correctos
    markers = np.asarray(markers, dtype=np.int32)
    img = np.asarray(img, dtype=np.uint8)
    
    # Generar colores aleatorios para cada segmento
    unique_markers = np.unique(markers)
    colors = np.random.randint(0, 255, (len(unique_markers), 3), dtype=np.uint8)
    
    result = np.zeros_like(img, dtype=np.uint8)
    
    for i, marker in enumerate(unique_markers):
        if marker > 0:  # Ignorar fondo (0) y bordes (-1)
            mask = markers == marker
            result[mask] = colors[i]
    
    # Mantener bordes en blanco si existen
    if -1 in unique_markers:
        result[markers == -1] = [255, 255, 255]
    
    return result

def analyze_segmentation_quality(markers, img):
    """
    Analiza la calidad de la segmentación
    """
    unique_markers = np.unique(markers)
    
    # Contar segmentos (excluyendo fondo y bordes)
    num_segments = len([m for m in unique_markers if m > 0])
    
    # Calcular tamaño promedio de segmentos
    segment_sizes = []
    for marker in unique_markers:
        if marker > 0:
            segment_sizes.append(np.sum(markers == marker))
    
    avg_segment_size = np.mean(segment_sizes) if segment_sizes else 0
    
    # Calcular cobertura (píxeles segmentados vs total)
    segmented_pixels = np.sum(markers > 0)
    total_pixels = markers.shape[0] * markers.shape[1]
    coverage = (segmented_pixels / total_pixels) * 100
    
    return {
        'num_segments': num_segments,
        'avg_segment_size': avg_segment_size,
        'coverage': coverage,
        'segment_sizes': segment_sizes
    }

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

# Modo de segmentación
st.sidebar.subheader("💧 Algoritmo Watershed")

watershed_method = st.sidebar.selectbox(
    "Método de Watershed:",
    options=['opencv_original', 'opencv_enhanced', 'sklearn_peaks'],
    format_func=lambda x: {
        'opencv_original': '🌊 OpenCV Original',
        'opencv_enhanced': '🌊 OpenCV Mejorado',
        'sklearn_peaks': '⛰️ Scikit-image Picos'
    }[x],
    help="Selecciona el algoritmo de Watershed a usar"
)

st.sidebar.markdown("---")

# Parámetros de preprocesamiento
st.sidebar.subheader("🔧 Preprocesamiento")

threshold_method = st.sidebar.selectbox(
    "Método de umbralización:",
    options=['otsu', 'adaptive', 'manual'],
    format_func=lambda x: {
        'otsu': '🎯 OTSU (Automático)',
        'adaptive': '🔄 Adaptativo',
        'manual': '✋ Manual'
    }[x]
)

kernel_size = st.sidebar.slider("Tamaño kernel morfológico:", 3, 15, 3, 2)
iterations_opening = st.sidebar.slider("Iteraciones apertura:", 1, 10, 2, 1)
iterations_dilation = st.sidebar.slider("Iteraciones dilatación:", 1, 10, 3, 1)

# Parámetros de marcadores de primer plano
st.sidebar.subheader("🎯 Marcadores Primer Plano")

fg_method = st.sidebar.selectbox(
    "Método de detección:",
    options=['distance_transform', 'erosion', 'contours'],
    format_func=lambda x: {
        'distance_transform': '📏 Transformada Distancia',
        'erosion': '🔸 Erosión',
        'contours': '🔲 Contornos'
    }[x]
)

if fg_method == 'distance_transform':
    threshold_factor = st.sidebar.slider("Factor umbral distancia:", 0.1, 1.0, 0.7, 0.1)
else:
    threshold_factor = 0.7

# Parámetros específicos del método
if watershed_method == 'sklearn_peaks':
    min_distance = st.sidebar.slider("Distancia mínima picos:", 5, 50, 20, 5)
else:
    connectivity = st.sidebar.selectbox("Conectividad:", [4, 8], index=1)

# Opciones de visualización
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Opciones de Vista")

show_steps = st.sidebar.checkbox("Mostrar Pasos Intermedios", True)
show_colored = st.sidebar.checkbox("Segmentación Coloreada", True)
show_analysis = st.sidebar.checkbox("Análisis de Calidad", True)

# Procesamiento de la imagen
gray, thresh, opening, sure_bg = preprocess_image(
    image, threshold_method, kernel_size, iterations_opening, iterations_dilation
)

dist_transform, sure_fg = find_foreground_markers(opening, fg_method, threshold_factor)

# Aplicar Watershed según el método seleccionado
if watershed_method == 'opencv_original':
    markers, unknown, result = apply_watershed_opencv(image, sure_bg, sure_fg, connectivity)
    
elif watershed_method == 'opencv_enhanced':
    # Versión mejorada con mejor manejo de bordes
    markers, unknown, result_temp = apply_watershed_opencv(image, sure_bg, sure_fg, connectivity)
    result = create_colored_segmentation(markers, image)
    
elif watershed_method == 'sklearn_peaks':
    if dist_transform is not None:
        labels, result = apply_watershed_sklearn(image, dist_transform, min_distance)
        markers = labels
        unknown = np.zeros_like(sure_fg)
    else:
        # Fallback si no hay transformada de distancia
        markers, unknown, result = apply_watershed_opencv(image, sure_bg, sure_fg, 8)

# Área principal de visualización
method_names = {
    'opencv_original': '🌊 OpenCV Watershed Original',
    'opencv_enhanced': '🌊 OpenCV Watershed Mejorado',
    'sklearn_peaks': '⛰️ Watershed con Detección de Picos'
}

st.subheader(f"{method_names[watershed_method]} - Resultados")

if show_steps:
    # Mostrar pasos intermedios
    st.subheader("🔍 Pasos del Procesamiento")
    
    # Primera fila: Imagen original y umbralización
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📷 Imagen Original**")
        st.image(cv2_to_pil(image), use_container_width=True)
    
    with col2:
        st.write(f"**🎯 Umbralización ({threshold_method.title()})**")
        st.image(cv2_to_pil(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)), use_container_width=True)
    
    with col3:
        st.write("**🔧 Apertura Morfológica**")
        st.image(cv2_to_pil(cv2.cvtColor(opening, cv2.COLOR_GRAY2RGB)), use_container_width=True)
    
    # Segunda fila: Fondo seguro, primer plano, región desconocida
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**✅ Fondo Seguro**")
        st.image(cv2_to_pil(cv2.cvtColor(sure_bg, cv2.COLOR_GRAY2RGB)), use_container_width=True)
    
    with col2:
        st.write(f"**🎯 Primer Plano ({fg_method.replace('_', ' ').title()})**")
        st.image(cv2_to_pil(cv2.cvtColor(sure_fg, cv2.COLOR_GRAY2RGB)), use_container_width=True)
    
    with col3:
        if 'unknown' in locals():
            st.write("**❓ Región Desconocida**")
            st.image(cv2_to_pil(cv2.cvtColor(unknown, cv2.COLOR_GRAY2RGB)), use_container_width=True)
        else:
            st.write("**📏 Transformada Distancia**")
            if dist_transform is not None:
                # Crear array de destino para normalize
                dist_norm = np.zeros_like(dist_transform)
                dist_norm = cv2.normalize(dist_transform, dist_norm, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                st.image(cv2_to_pil(cv2.cvtColor(dist_norm, cv2.COLOR_GRAY2RGB)), use_container_width=True)

# Resultado final
st.subheader("🎨 Resultado de Segmentación")

if show_colored and watershed_method != 'sklearn_peaks':
    # Mostrar versión coloreada
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🌊 Watershed Original**")
        if watershed_method == 'opencv_original':
            st.image(cv2_to_pil(result), use_container_width=True)
        else:
            # Crear resultado con bordes para comparación
            temp_result = image.copy()
            temp_result[markers == -1] = [0, 0, 255]
            st.image(cv2_to_pil(temp_result), use_container_width=True)
    
    with col2:
        st.write("**🎨 Segmentación Coloreada**")
        colored_result = create_colored_segmentation(markers, image)
        st.image(cv2_to_pil(colored_result), use_container_width=True)
else:
    st.image(cv2_to_pil(result), use_container_width=True)

# Análisis de calidad
if show_analysis:
    st.subheader("📊 Análisis de Segmentación")
    
    analysis = analyze_segmentation_quality(markers, image)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Segmentos Detectados", analysis['num_segments'])
    
    with col2:
        st.metric("Tamaño Promedio", f"{analysis['avg_segment_size']:.0f} px")
    
    with col3:
        st.metric("Cobertura", f"{analysis['coverage']:.1f}%")
    
    with col4:
        if analysis['segment_sizes']:
            max_segment = max(analysis['segment_sizes'])
            st.metric("Segmento Mayor", f"{max_segment} px")
    
    # Histograma de tamaños de segmentos
    if analysis['segment_sizes'] and len(analysis['segment_sizes']) > 1:
        st.subheader("📈 Distribución de Tamaños de Segmentos")
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(analysis['segment_sizes'], bins=min(20, len(analysis['segment_sizes'])), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Tamaño del Segmento (píxeles)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Tamaños de Segmentos')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)

# Botón de descarga
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    processed_pil = cv2_to_pil(result)
    buf = io.BytesIO()
    processed_pil.save(buf, format='PNG')
    
    st.download_button(
        label="📥 Descargar Segmentación",
        data=buf.getvalue(),
        file_name=f"watershed_{watershed_method}.png",
        mime="image/png",
        use_container_width=True
    )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("💧 Sobre Watershed"):
    st.markdown("""
    **¿Qué es Watershed?**
    - Algoritmo de segmentación basado en topografía
    - Trata la imagen como una superficie de alturas
    - "Inunda" desde marcadores mínimos locales
    - Se detiene cuando las aguas se encuentran
    
    **Proceso:**
    1. Preprocesamiento y umbralización
    2. Encontrar marcadores seguros (fondo/primer plano)
    3. Identificar regiones desconocidas
    4. Aplicar algoritmo de inundación
    5. Marcar fronteras entre regiones
    
    **Ventajas:**
    - Segmentación precisa de objetos conectados
    - No requiere conocimiento previo del número de objetos
    - Funciona bien con formas irregulares
    """)

with st.sidebar.expander("🔧 Parámetros Clave"):
    st.markdown("""
    **Preprocesamiento:**
    - **Umbralización OTSU**: Automática, basada en histograma
    - **Umbralización Adaptativa**: Local, mejor para iluminación variable
    - **Operaciones Morfológicas**: Limpian ruido y conectan regiones
    
    **Marcadores:**
    - **Transformada de Distancia**: Encuentra centros de objetos
    - **Erosión**: Reduce objetos a sus núcleos
    - **Contornos**: Usa centros geométricos
    
    **Conectividad:**
    - **4-conectividad**: Solo vecinos horizontales/verticales
    - **8-conectividad**: Incluye vecinos diagonales
    """)

with st.sidebar.expander("🎯 Métodos Implementados"):
    st.markdown("""
    **🌊 OpenCV Original:**
    - Implementación del código base
    - Bordes marcados en rojo
    - Método estándar y confiable
    
    **🌊 OpenCV Mejorado:**
    - Segmentación coloreada automática
    - Mejor visualización de regiones
    - Mismo algoritmo, mejor presentación
    
    **⛰️ Scikit-image Picos:**
    - Detección automática de picos
    - Menor supervisión manual
    - Coloreado automático por regiones
    
    **Casos de Uso:**
    - Conteo de objetos separados
    - Segmentación de células
    - Separación de objetos tocándose
    """)

with st.sidebar.expander("💡 Consejos de Uso"):
    st.markdown("""
    **Para mejores resultados:**
    - Usar imágenes con objetos bien contrastados
    - Ajustar umbralización según el contenido
    - Experimentar con tamaños de kernel
    - Probar diferentes métodos de marcadores
    
    **Problemas comunes:**
    - **Sobre-segmentación**: Reducir sensibilidad
    - **Sub-segmentación**: Mejorar marcadores
    - **Ruido**: Aumentar operaciones morfológicas
    - **Objetos perdidos**: Revisar umbralización
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>💧 <strong>Segmentación Watershed Interactiva</strong> | Capítulo 7 - Segmentación de Imágenes</p>
        <p><small>Explora técnicas avanzadas de segmentación usando el algoritmo Watershed</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)