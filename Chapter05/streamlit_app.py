"""
Capítulo 5 - Detección de Características SIFT
Demostración del código sift_detect.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 5 - Detección SIFT", layout="wide")

# Título
st.title("🎯 Capítulo 5: Detección de Características SIFT")
st.markdown("**Demostración del código: `sift_detect.py`**")

def detect_sift_features(img):
    """Función de detección SIFT del código original"""
    # Convertir a escala de grises
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    sift = None
    
    # Intentar diferentes formas de crear SIFT según la versión de OpenCV
    try:
        # OpenCV 4.4+ con opencv-contrib-python
        if hasattr(cv2, 'xfeatures2d'):
            sift = cv2.xfeatures2d.SIFT_create()
        elif hasattr(cv2, 'SIFT_create'):
            # OpenCV 4.5.1+ 
            sift = cv2.SIFT_create()
        else:
            raise AttributeError("SIFT no disponible")
    except (AttributeError, Exception):
        # Usar detector de esquinas como alternativa
        st.warning("⚠️ SIFT no está disponible. Usando detector de esquinas Harris como alternativa.")
        return detect_harris_corners(img, gray_image)
    
    # Detectar keypoints
    keypoints = sift.detect(gray_image, None)
    
    # Dibujar keypoints en la imagen
    result_img = img.copy()
    cv2.drawKeypoints(result_img, keypoints, result_img, 
                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return result_img, keypoints

def detect_harris_corners(img, gray_image):
    """Función alternativa usando detector de esquinas Harris"""
    # Detectar esquinas Harris
    corners = cv2.cornerHarris(gray_image, 2, 3, 0.04)
    
    # Dilatar para marcar las esquinas
    kernel = np.ones((3,3), np.uint8)
    corners = cv2.dilate(corners, kernel)
    
    # Crear resultado con esquinas marcadas
    result_img = img.copy()
    result_img[corners > 0.01 * corners.max()] = [0, 0, 255]  # Marcar en rojo
    
    # Crear keypoints sintéticos para mantener compatibilidad
    keypoints = []
    corner_coords = np.where(corners > 0.01 * corners.max())
    
    for i in range(len(corner_coords[0])):
        y, x = corner_coords[0][i], corner_coords[1][i]
        # Crear keypoint sintético
        kp = cv2.KeyPoint(x=float(x), y=float(y), size=10, angle=0, response=float(corners[y,x]))
        keypoints.append(kp)
    
    return result_img, keypoints

def load_image():
    """Carga una imagen de ejemplo o crea una imagen de prueba"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Buscar imágenes de ejemplo en la carpeta images
    image_files = ['fishing_house.jpg', 'house.jpg', 'box.png', 'tool.png']
    
    for img_file in image_files:
        img_path = os.path.join(script_dir, 'images', img_file)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_file
    
    # Crear imagen de ejemplo con características detectables
    img = np.ones((500, 700, 3), dtype=np.uint8) * 240
    
    # Crear estructuras geométricas con muchas esquinas
    # Edificio principal
    cv2.rectangle(img, (100, 150), (300, 400), (120, 120, 120), -1)
    cv2.rectangle(img, (110, 160), (290, 390), (180, 180, 180), 3)
    
    # Ventanas (esquinas detectables)
    for i in range(3):
        for j in range(4):
            x = 130 + j * 40
            y = 180 + i * 50
            cv2.rectangle(img, (x, y), (x+25, y+35), (80, 80, 80), -1)
            cv2.rectangle(img, (x+3, y+3), (x+22, y+32), (200, 200, 200), 2)
    
    # Casa secundaria
    cv2.rectangle(img, (400, 200), (600, 400), (150, 100, 80), -1)
    cv2.rectangle(img, (410, 210), (590, 390), (200, 150, 100), 3)
    
    # Techo triangular
    pts = np.array([[400, 200], [500, 120], [600, 200]], np.int32)
    cv2.fillPoly(img, [pts], (100, 80, 60))
    cv2.polylines(img, [pts], True, (80, 60, 40), 3)
    
    # Puerta
    cv2.rectangle(img, (480, 320), (520, 390), (60, 40, 20), -1)
    cv2.rectangle(img, (483, 323), (517, 387), (100, 70, 40), 2)
    
    # Ventanas casa 2
    cv2.rectangle(img, (430, 240), (460, 280), (80, 80, 80), -1)
    cv2.rectangle(img, (540, 240), (570, 280), (80, 80, 80), -1)
    
    # Elementos adicionales con características
    # Árbol con ramas (líneas detectables)
    cv2.circle(img, (50, 300), 30, (0, 120, 0), -1)
    cv2.rectangle(img, (45, 330), (55, 400), (101, 67, 33), -1)
    
    # Líneas adicionales para crear más keypoints
    for i in range(5):
        y_pos = 50 + i * 80
        cv2.line(img, (650, y_pos), (680, y_pos + 30), (0, 0, 0), 2)
        cv2.line(img, (680, y_pos + 30), (650, y_pos + 60), (0, 0, 0), 2)
    
    # Texto con esquinas
    cv2.putText(img, 'SIFT DETECTION', (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
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
        # Aplicar detección SIFT
        result_img, keypoints = detect_sift_features(img)
        
        # Mostrar estado de validación
        st.success("✅ **Imagen procesada correctamente con OpenCV SIFT**")
        
        # Información adicional sobre el procesamiento
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Imagen Original", "OK ✅", f"{img.shape[1]}x{img.shape[0]}")
        with col2:
            st.metric("Algoritmo SIFT", "OK ✅", "Detector cargado")
        with col3:
            st.metric("Keypoints Detectados", f"{len(keypoints)} 🎯", "Características encontradas")
        
    except Exception as e:
        st.error(f"❌ **Error al procesar la imagen:** {str(e)}")
        st.info("💡 Intenta con una imagen diferente o instala opencv-contrib-python")
        st.stop()
    
    # Mostrar código original
    st.subheader("📄 Código Original:")
    st.code("""
# Código del archivo sift_detect.py
import cv2 
import numpy as np 
 
input_image = cv2.imread('images/fishing_house.jpg') 
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY) 
 
# For version opencv < 3.0.0, use cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create() 
keypoints = sift.detect(gray_image, None)
 
cv2.drawKeypoints(input_image, keypoints, input_image, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
 
cv2.imshow('SIFT features', input_image) 
cv2.waitKey()
""", language="python")
    
    # Mostrar imagen original
    st.subheader("🖼️ Imagen Original:")
    st.image(cv2_to_pil(img), caption=f"Imagen: {img_name}", width="stretch")
    
    # Mostrar resultados de la detección
    st.subheader("🎯 Resultados de Detección SIFT:")
    
    # Crear dos columnas para comparar
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Sin Procesar**")
        st.image(cv2_to_pil(img), caption="Imagen original", width="stretch")
    
    with col2:
        st.markdown("**Imagen Con Keypoints SIFT**")
        st.image(cv2_to_pil(result_img), caption=f"{len(keypoints)} keypoints detectados", width="stretch")
    
    # Mostrar información de keypoints detectados
    if len(keypoints) > 0:
        st.subheader("📊 Análisis de Keypoints Detectados:")
        
        # Estadísticas generales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Keypoints", len(keypoints))
        
        with col2:
            responses = [kp.response for kp in keypoints]
            avg_response = np.mean(responses) if responses else 0
            st.metric("Respuesta Promedio", f"{avg_response:.2f}")
        
        with col3:
            sizes = [kp.size for kp in keypoints]
            avg_size = np.mean(sizes) if sizes else 0
            st.metric("Tamaño Promedio", f"{avg_size:.1f} px")
        
        with col4:
            angles = [kp.angle for kp in keypoints]
            avg_angle = np.mean(angles) if angles else 0
            st.metric("Ángulo Promedio", f"{avg_angle:.1f}°")
        
        # Mostrar algunos keypoints individuales
        st.markdown("**🔍 Primeros 10 Keypoints:**")
        
        for i, kp in enumerate(keypoints[:10]):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.write(f"**KP {i+1}:**")
            with col2:
                st.write(f"X: {kp.pt[0]:.1f}")
            with col3:
                st.write(f"Y: {kp.pt[1]:.1f}")
            with col4:
                st.write(f"Tamaño: {kp.size:.1f}")
            with col5:
                st.write(f"Respuesta: {kp.response:.3f}")
                
        if len(keypoints) > 10:
            st.info(f"ℹ️ Se muestran solo los primeros 10 keypoints de {len(keypoints)} detectados.")
    else:
        st.info("ℹ️ No se detectaron keypoints SIFT en esta imagen. Prueba con una imagen que tenga más esquinas y características distintivas.")
    
    # Mostrar información técnica del algoritmo
    st.subheader("📊 Información del Algoritmo SIFT:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Parámetros SIFT:**")
        st.code("""
nfeatures = 0           # Máximo número de keypoints
nOctaveLayers = 3       # Capas por octava
contrastThreshold = 0.04 # Umbral de contraste
edgeThreshold = 10      # Umbral de bordes
sigma = 1.6             # Sigma gaussiano
        """)
        
        st.markdown("**Pasos del Proceso:**")
        st.write("1. Conversión a escala de grises")
        st.write("2. Construcción del espacio de escalas")
        st.write("3. Detección de extremos en DoG")
        st.write("4. Localización de keypoints")
        st.write("5. Asignación de orientación")
        st.write("6. Generación de descriptores")
        
    with col2:
        st.markdown("**Propiedades de los Keypoints:**")
        st.write("• **Posición (pt)**: Coordenadas (x, y) del keypoint")
        st.write("• **Tamaño (size)**: Escala del keypoint detectado")
        st.write("• **Ángulo (angle)**: Orientación dominante (-1 a 360°)")
        st.write("• **Respuesta (response)**: Fuerza de la detección")
        st.write("• **Octava (octave)**: Nivel de pirámide donde se detectó")
        st.write("• **Class_id**: Identificador de clase (opcional)")
        
        st.markdown("**Características del Algoritmo:**")
        st.write("🔄 **Invariante a escala y rotación**")
        st.write("💡 **Robusto a cambios de iluminación**")
        st.write("🎯 **Descriptores distintivos de 128 dimensiones**")
        st.write("⚡ **Detección en múltiples escalas**")
    
    # Explicación técnica detallada
    st.subheader("📝 Explicación Técnica SIFT:")
    
    st.markdown("""
    ### 🎯 **Scale-Invariant Feature Transform (SIFT)**
    
    SIFT es uno de los algoritmos más robustos para **detección y descripción de características locales**:
    
    #### **¿Qué hace SIFT?**
    - **Detecta keypoints** invariantes a escala y rotación
    - **Genera descriptores** únicos de 128 dimensiones
    - **Localiza características** estables en diferentes condiciones
    - **Permite matching** entre imágenes similares
    
    #### **Algoritmo Paso a Paso:**
    
    **1. 🔍 Detección de Extremos en el Espacio de Escalas**
    - Construcción de pirámide gaussiana
    - Cálculo de Diferencia de Gaussianas (DoG)
    - Búsqueda de máximos/mínimos locales
    
    **2. 📍 Localización Precisa de Keypoints**
    - Interpolación sub-píxel usando Taylor
    - Eliminación de puntos de bajo contraste
    - Filtrado de respuestas en bordes
    
    **3. 🧭 Asignación de Orientación**
    - Cálculo de gradientes locales
    - Histograma de orientaciones
    - Asignación de orientación dominante
    
    **4. 📊 Generación de Descriptores**
    - Ventana de 16x16 píxeles alrededor del keypoint
    - División en sub-regiones de 4x4
    - Histogramas de gradientes de 8 direcciones
    - Vector resultante de 4×4×8 = 128 dimensiones
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Ventajas de SIFT:**")
        st.write("• **Invarianza completa** a escala y rotación")
        st.write("• **Robustez** a cambios de iluminación")
        st.write("• **Alta distintividad** de descriptores")
        st.write("• **Estabilidad** ante ruido y distorsiones menores")
        st.write("• **Amplia aplicabilidad** en computer vision")
        st.write("• **Matching preciso** entre imágenes")
    
    with col2:
        st.markdown("**⚠️ Limitaciones:**")
        st.write("• **Computacionalmente costoso**")
        st.write("• **Patente hasta 2020** (ahora libre)")
        st.write("• **Menos eficiente** que métodos modernos")
        st.write("• **Sensible a cambios** de perspectiva extremos")
        st.write("• **Memoria intensiva** (128D por descriptor)")
        st.write("• **Menor velocidad** en tiempo real")
    
    # Aplicaciones prácticas
    st.subheader("🚀 Aplicaciones Prácticas:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Reconocimiento y Matching:**")
        st.write("• **Reconocimiento de objetos** en diferentes escalas")
        st.write("• **Stitching panorámico** de múltiples imágenes")
        st.write("• **Tracking de objetos** en video")
        st.write("• **Búsqueda de imágenes** por contenido visual")
        
        st.markdown("**🤖 Robótica y Navegación:**")
        st.write("• **SLAM** (Simultaneous Localization and Mapping)")
        st.write("• **Navegación visual** de robots autónomos")
        st.write("• **Reconocimiento de lugares** visitados")
    
    with col2:
        st.markdown("**🏥 Aplicaciones Médicas:**")
        st.write("• **Registro de imágenes** médicas multimodales")
        st.write("• **Análisis de radiografías** y tomografías")
        st.write("• **Seguimiento de lesiones** en el tiempo")
        
        st.markdown("**📱 Realidad Aumentada:**")
        st.write("• **Detección de marcadores** naturales")
        st.write("• **Superposición de objetos** 3D en tiempo real")
        st.write("• **Aplicaciones móviles** de AR/VR")
    
    # Comparación con otros algoritmos
    st.info("""
    💡 **Alternativas a SIFT:**
    • **SURF**: Más rápido, menos preciso
    • **ORB**: Libre de patentes, más eficiente
    • **AKAZE**: Mejor para texturas
    • **BRISK**: Optimizado para tiempo real
    • **Redes Neuronales**: LIFT, SuperPoint (métodos modernos)
    """)

else:
    st.error("❌ No se pudo cargar ninguna imagen de ejemplo")