"""
Capítulo 4 - Detección Facial
Demostración del código face_detection.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 4 - Detección Facial", layout="wide")

# Título
st.title("👤 Capítulo 4: Detección Facial")
st.markdown("**Demostración del código: `face_detection.py`**")

def detect_faces(img):
    """Función de detección facial del código original"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cascade_path = os.path.join(script_dir, 'cascade_files', 'haarcascade_frontalface_alt.xml')
    
    # Cargar el clasificador Haar Cascade
    if os.path.exists(cascade_path):
        face_cascade = cv2.CascadeClassifier(cascade_path)
    else:
        # Crear un clasificador vacío si no existe el archivo
        face_cascade = cv2.CascadeClassifier()
        if face_cascade.empty():
            st.error("❌ No se pudo cargar el clasificador Haar Cascade")
            return img, []
    
    # Detectar rostros en la imagen
    face_rects = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3)
    
    # Dibujar rectángulos alrededor de los rostros detectados
    result_img = img.copy()
    for (x, y, w, h) in face_rects:
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
    return result_img, face_rects

def load_image():
    """Carga una imagen de ejemplo o crea una imagen de prueba"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Buscar imágenes de ejemplo en la carpeta images
    image_files = ['mask_hannibal.png', 'moustache.png', 'sunglasses.png']
    
    for img_file in image_files:
        img_path = os.path.join(script_dir, 'images', img_file)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_file
    
    # Crear imagen de ejemplo si no existe ninguna
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Crear un rostro simulado para demostrar la detección
    # Cara (óvalo)
    cv2.ellipse(img, (300, 200), (100, 130), 0, 0, 360, (220, 180, 160), -1)
    
    # Ojos
    cv2.circle(img, (270, 170), 15, (0, 0, 0), -1)  # Ojo izquierdo
    cv2.circle(img, (330, 170), 15, (0, 0, 0), -1)  # Ojo derecho
    
    # Nariz
    cv2.ellipse(img, (300, 200), (8, 15), 0, 0, 360, (200, 160, 140), -1)
    
    # Boca
    cv2.ellipse(img, (300, 240), (25, 10), 0, 0, 180, (150, 50, 50), -1)
    
    # Añadir texto
    cv2.putText(img, 'FACE DETECTION', (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
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
        # Aplicar detección facial
        result_img, face_rects = detect_faces(img)
        
        # Mostrar estado de validación
        st.success("✅ **Imagen procesada correctamente con OpenCV**")
        
        # Información adicional sobre el procesamiento
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Imagen Original", "OK ✅", f"{img.shape[1]}x{img.shape[0]}")
        with col2:
            st.metric("Detección Facial", "OK ✅", f"Haar Cascade cargado")
        with col3:
            st.metric("Rostros Detectados", f"{len(face_rects)} 👤", f"Rectángulos dibujados")
        
    except Exception as e:
        st.error(f"❌ **Error al procesar la imagen:** {str(e)}")
        st.info("💡 Intenta con una imagen diferente o verifica el formato")
        st.stop()
    
    # Mostrar código original
    st.subheader("📄 Código Original:")
    st.code("""
# Código del archivo face_detection.py
import cv2 
import numpy as np 
 
face_cascade = cv2.CascadeClassifier('./cascade_files/haarcascade_frontalface_alt.xml') 
 
cap = cv2.VideoCapture(1) 
scaling_factor = 0.5 
 
while True: 
    ret, frame = cap.read() 
    frame = cv2.resize(frame, None, fx=scaling_factor, 
                      fy=scaling_factor, interpolation=cv2.INTER_AREA)
 
    face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3) 
    for (x,y,w,h) in face_rects: 
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3) 
 
    cv2.imshow('Face Detector', frame) 
 
    c = cv2.waitKey(1) 
    if c == 27: 
        break 
 
cap.release() 
cv2.destroyAllWindows()
""", language="python")
    
    # Mostrar imagen original
    st.subheader("🖼️ Imagen Original:")
    st.image(cv2_to_pil(img), caption=f"Imagen: {img_name}", width="stretch")
    
    # Mostrar resultados de la detección
    st.subheader("👤 Resultados de Detección Facial:")
    
    # Crear dos columnas para comparar
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Sin Procesar**")
        st.image(cv2_to_pil(img), caption="Imagen original", width="stretch")
    
    with col2:
        st.markdown("**Imagen Con Detección**")
        st.image(cv2_to_pil(result_img), caption=f"{len(face_rects)} rostro(s) detectado(s)", width="stretch")
    
    # Mostrar información de rostros detectados
    if len(face_rects) > 0:
        st.subheader("📊 Información de Rostros Detectados:")
        
        for i, (x, y, w, h) in enumerate(face_rects):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"Rostro {i+1} - X", f"{x} px")
            with col2:
                st.metric(f"Rostro {i+1} - Y", f"{y} px")
            with col3:
                st.metric(f"Rostro {i+1} - Ancho", f"{w} px")
            with col4:
                st.metric(f"Rostro {i+1} - Alto", f"{h} px")
    else:
        st.info("ℹ️ No se detectaron rostros en esta imagen. Prueba con una imagen que contenga rostros frontales claros.")
    
    # Mostrar información técnica del algoritmo
    st.subheader("📊 Información del Algoritmo:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Parámetros de Detección:**")
        st.code("""
scaleFactor = 1.3       # Factor de escalado
minNeighbors = 3        # Mínimos vecinos
classifier = haarcascade_frontalface_alt.xml
        """)
        
        st.markdown("**Pasos del Proceso:**")
        st.write("1. Cargar clasificador Haar Cascade")
        st.write("2. Aplicar `detectMultiScale()` a la imagen")
        st.write("3. Obtener coordenadas de rostros detectados")
        st.write("4. Dibujar rectángulos con `cv2.rectangle()`")
        
    with col2:
        st.markdown("**Archivos Haar Cascade Disponibles:**")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_dir = os.path.join(script_dir, 'cascade_files')
        
        cascade_files = [
            "haarcascade_frontalface_alt.xml",
            "haarcascade_eye.xml", 
            "haarcascade_mcs_nose.xml",
            "haarcascade_mcs_mouth.xml",
            "haarcascade_mcs_leftear.xml",
            "haarcascade_mcs_rightear.xml"
        ]
        
        for cascade_file in cascade_files:
            cascade_path = os.path.join(cascade_dir, cascade_file)
            status = "✅" if os.path.exists(cascade_path) else "❌"
            st.write(f"{status} {cascade_file}")
    
    # Explicación técnica detallada
    st.subheader("📝 Explicación Técnica:")
    
    st.markdown("""
    ### 👤 **Detección Facial con Haar Cascades**
    
    Los **Haar Cascades** son clasificadores entrenados para detectar objetos específicos en imágenes:
    
    #### **¿Qué son los Haar Cascades?**
    - **Clasificadores en cascada** basados en características de Haar
    - **Pre-entrenados** en miles de imágenes positivas y negativas
    - **Rápidos y eficientes** para detección en tiempo real
    - **Especializados** para diferentes características faciales
    
    #### **Algoritmo de Detección:**
    1. **Carga del clasificador**: `cv2.CascadeClassifier()`
    2. **Detección multi-escala**: `detectMultiScale()`
        - `scaleFactor=1.3`: Reduce imagen en cada escala
        - `minNeighbors=3`: Mínimo de detecciones vecinas para confirmar
    3. **Resultado**: Lista de rectángulos (x, y, width, height)
    4. **Visualización**: Dibujo de rectángulos verdes alrededor de rostros
    
    #### **Parámetros Importantes:**
    - **scaleFactor**: Control de escalas de búsqueda (1.1 - 2.0)
    - **minNeighbors**: Filtro de falsos positivos (3-6 típico)
    - **minSize/maxSize**: Rango de tamaños de rostros a detectar
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Técnicas Utilizadas:**")
        st.write("• **Haar Features**: Patrones rectangulares de luz/sombra")
        st.write("• **Integral Image**: Cálculo rápido de sumas de píxeles")
        st.write("• **AdaBoost**: Algoritmo de aprendizaje por refuerzo")
        st.write("• **Cascade Structure**: Filtros secuenciales rápidos")
        st.write("• **Multi-scale Detection**: Búsqueda en múltiples tamaños")
        st.write("• **Non-maxima Suppression**: Eliminación de duplicados")
    
    with col2:
        st.markdown("**🎯 Aplicaciones Prácticas:**")
        st.write("• **Fotografía digital**: Enfoque automático en rostros")
        st.write("• **Seguridad**: Sistemas de vigilancia y control de acceso")
        st.write("• **Redes sociales**: Etiquetado automático de personas")
        st.write("• **Realidad aumentada**: Filtros y efectos faciales")
        st.write("• **Análisis médico**: Detección en imágenes clínicas")
        st.write("• **Automatización**: Conteo de personas, demografía")
    
    # Información sobre limitaciones
    st.warning("""
    ⚠️ **Limitaciones de Haar Cascades:**
    • Funciona mejor con rostros frontales
    • Sensible a iluminación y ángulos
    • Puede generar falsos positivos/negativos
    • Menos preciso que métodos modernos (DNN, CNN)
    """)
    
    st.info("💡 **Nota**: Para mejor precisión en aplicaciones modernas, considera usar redes neuronales profundas (DNN) como los modelos de OpenCV DNN o bibliotecas especializadas como dlib o face_recognition.")

else:
    st.error("❌ No se pudo cargar ninguna imagen de ejemplo")