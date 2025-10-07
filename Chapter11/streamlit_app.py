"""
Capítulo 11 - Extractor de Características
Demostración del código feature_extractor.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
# Importar las clases necesarias del archivo feature_extractor
# from feature_extractor import FeatureExtractor, DenseDetector, SIFTExtractor, Quantizer

# Definir las clases aquí para compatibilidad con OpenCV 4.x
class DenseDetector(): 
    def __init__(self, step_size=20, feature_scale=20, img_bound=20): 
        self.initXyStep = step_size
        self.initFeatureScale = feature_scale
        self.initImgBound = img_bound
 
    def detect(self, img):
        keypoints = []
        rows, cols = img.shape[:2]
        for x in range(self.initImgBound, rows, self.initFeatureScale):
            for y in range(self.initImgBound, cols, self.initFeatureScale):
                keypoints.append(cv2.KeyPoint(float(x), float(y), self.initXyStep))
        return keypoints 

class SIFTExtractor():
    def __init__(self):
        # Para OpenCV 4.x (versión actual)
        self.extractor = cv2.SIFT_create()

    def compute(self, image, kps): 
        if image is None: 
            print("Not a valid image")
            raise TypeError 
 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = self.extractor.detectAndCompute(gray_image, None)
        return kps, des

class Quantizer(object): 
    def __init__(self, num_clusters=32):
        self.extractor = SIFTExtractor()
        self.num_clusters = num_clusters 
        self.num_retries = 10 
 
    def quantize(self, datapoints): 
        from sklearn.cluster import KMeans
        # Create KMeans object 
        kmeans = KMeans(self.num_clusters, 
                        n_init=max(self.num_retries, 1), 
                        max_iter=10, tol=1.0) 
 
        # Run KMeans on the datapoints 
        res = kmeans.fit(datapoints) 
 
        # Extract the centroids of those clusters 
        centroids = res.cluster_centers_
 
        return kmeans, centroids 
 
    def normalize(self, input_data): 
        sum_input = np.sum(input_data)
        return input_data / sum_input if sum_input > 0 else input_data
 
    # Extract feature vector from the image 
    def get_feature_vector(self, img, kmeans, centroids): 
        kps = DenseDetector().detect(img) 
        kps, fvs = self.extractor.compute(img, kps) 
        labels = kmeans.predict(fvs) 
        fv = np.zeros(self.num_clusters) 
 
        for i, item in enumerate(fvs): 
            fv[labels[i]] += 1 
 
        fv_image = np.reshape(fv, ((1, fv.shape[0]))) 
        return self.normalize(fv_image)

# Configuración de la página
st.set_page_config(page_title="Capítulo 11 - Extractor de Características", layout="wide")

# Título
st.title("🔍 Capítulo 11: Extractor de Características")

def cv2_to_pil(cv2_img):
    """Convierte imagen de OpenCV (BGR) a PIL (RGB)"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    """Convierte imagen de PIL (RGB) a OpenCV (BGR)"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

def load_example_image():
    """Carga la imagen de ejemplo"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'images', 'test.png')
    
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    else:
        # Crear imagen de ejemplo si no existe
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        # Crear algunas formas para extraer características
        cv2.rectangle(img, (50, 50), (150, 150), (100, 150, 200), -1)
        cv2.circle(img, (250, 100), 50, (200, 100, 50), -1)
        cv2.rectangle(img, (100, 180), (300, 250), (50, 200, 100), -1)
        cv2.putText(img, 'FEATURES', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

def visualize_keypoints(img, keypoints):
    """Visualiza los puntos clave en la imagen"""
    img_with_keypoints = img.copy()
    for kp in keypoints[:50]:  # Mostrar solo los primeros 50 para claridad
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(img_with_keypoints, (x, y), 3, (0, 255, 0), -1)
    return img_with_keypoints

def extract_features_demo(img, num_clusters=32):
    """Demuestra la extracción de características usando las clases del código original"""
    try:
        # 1. Detector denso de puntos clave
        dense_detector = DenseDetector(step_size=20, feature_scale=20, img_bound=20)
        keypoints = dense_detector.detect(img)
        
        # 2. Extractor SIFT
        sift_extractor = SIFTExtractor()
        kps, descriptors = sift_extractor.compute(img, keypoints)
        
        # 3. Cuantizador para crear vector de características
        quantizer = Quantizer(num_clusters=num_clusters)
        
        # Preparar datos para cuantización
        if descriptors is not None and len(descriptors) > 0:
            kmeans, centroids = quantizer.quantize(descriptors)
            feature_vector = quantizer.get_feature_vector(img, kmeans, centroids)
            
            return {
                'keypoints': keypoints,
                'sift_keypoints': kps,
                'descriptors': descriptors,
                'feature_vector': feature_vector,
                'num_features': len(descriptors) if descriptors is not None else 0,
                'success': True
            }
        else:
            return {'success': False, 'error': 'No se pudieron extraer características SIFT'}
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

# Sidebar para configuración
st.sidebar.header("🛠️ Configuración")

# Selección de fuente de imagen
image_source = st.sidebar.radio(
    "Elige la fuente de la imagen:",
    ["🖼️ Imagen de ejemplo", "📤 Cargar mi propia imagen"],
    help="Selecciona si quieres usar una imagen de ejemplo del proyecto o cargar tu propia imagen"
)

# Parámetros de extracción
st.sidebar.subheader("📊 Parámetros")
num_clusters = st.sidebar.slider("Número de clusters", min_value=8, max_value=64, value=32, step=8,
                                help="Número de clusters para la cuantización de características")

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
    img = load_example_image()
    img_name = "test.png"
    st.sidebar.success(f"✅ Usando imagen: {img_name}")

# Mostrar información de la imagen si está cargada
if img is not None:
    height, width = img.shape[:2]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Información de la Imagen:**")
    st.sidebar.write(f"• **Nombre:** {img_name}")
    st.sidebar.write(f"• **Dimensiones:** {width} x {height} píxeles")
    st.sidebar.write(f"• **Canales:** {img.shape[2] if len(img.shape) > 2 else 1}")

if img is not None:
    # Procesar la imagen
    try:
        with st.spinner("🔄 Extrayendo características..."):
            results = extract_features_demo(img, num_clusters)
        
        if results['success']:
            st.success("✅ **Características extraídas correctamente con OpenCV**")
            
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Puntos Densos", len(results['keypoints']))
            with col2:
                st.metric("Características SIFT", results['num_features'])
            with col3:
                st.metric("Vector de Características", f"{len(results['feature_vector'][0])}D")
        else:
            st.error(f"❌ **Error al extraer características:** {results['error']}")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ **Error al procesar la imagen:** {str(e)}")
        st.stop()
    
    # Mostrar código original
    st.subheader("📄 Código Original:")
    st.code("""
# Código principal de feature_extractor.py
import cv2
import numpy as np
from sklearn.cluster import KMeans

# 1. Detector denso de características
detector = DenseDetector(step_size=20, feature_scale=20, img_bound=20)
keypoints = detector.detect(img)

# 2. Extractor SIFT
sift_extractor = SIFTExtractor()
kps, descriptors = sift_extractor.compute(img, keypoints)

# 3. Cuantización con K-means
quantizer = Quantizer(num_clusters=32)
kmeans, centroids = quantizer.quantize(descriptors)
feature_vector = quantizer.get_feature_vector(img, kmeans, centroids)
""", language="python")
    
    # Mostrar resultados visuales
    if results['success']:
        st.subheader("🖼️ Resultados Visuales:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Imagen Original**")
            st.image(cv2_to_pil(img), use_container_width=True)
        
        with col2:
            st.markdown("**Puntos Clave Detectados**")
            img_with_kp = visualize_keypoints(img, results['keypoints'])
            st.image(cv2_to_pil(img_with_kp), use_container_width=True)
        
        # Vector de características
        st.subheader("📊 Vector de Características:")
        feature_vec = results['feature_vector'][0]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.line_chart(feature_vec, height=300)
            
        with col2:
            st.markdown("**Estadísticas:**")
            st.write(f"• **Dimensiones:** {len(feature_vec)}")
            st.write(f"• **Valor máximo:** {feature_vec.max():.4f}")
            st.write(f"• **Valor mínimo:** {feature_vec.min():.4f}")
            st.write(f"• **Promedio:** {feature_vec.mean():.4f}")
            st.write(f"• **Desviación estándar:** {feature_vec.std():.4f}")
        
        # Mostrar parte del vector
        st.markdown("**Primeros 10 valores del vector:**")
        vector_str = ", ".join([f"{val:.4f}" for val in feature_vec[:10]])
        st.code(f"[{vector_str}, ...]")
    
    # Explicación técnica
    st.subheader("📚 Explicación:")
    st.markdown("""
    Este código demuestra la **extracción de características** usando OpenCV:
    
    1. **Detector Denso**: Genera puntos clave uniformemente distribuidos por la imagen
    2. **Extractor SIFT**: Calcula descriptores SIFT para cada punto clave
    3. **Cuantización**: Usa K-means para agrupar características similares
    4. **Vector de Características**: Crea un histograma normalizado que representa la imagen
    
    **Aplicaciones:**
    - Clasificación de imágenes
    - Búsqueda por similitud
    - Reconocimiento de objetos
    - Análisis de contenido visual
    """)
    
    st.info("💡 **Nota**: El vector de características resultante puede usarse para entrenar clasificadores o comparar imágenes.")

else:
    st.error("❌ No se pudo cargar la imagen")
    st.info("📁 Asegúrate de que existe una imagen en la carpeta 'images' o sube una imagen propia")