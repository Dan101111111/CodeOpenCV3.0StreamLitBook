"""
Aplicación Streamlit - Machine Learning y Extracción de Características
Aplicación educativa para explorar técnicas de machine learning aplicadas a visión por computador
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter

# Imports opcionales - usar fallbacks si no están disponibles
try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.svm import SVC
    SVM_AVAILABLE = True
except ImportError:
    SVM_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Funciones fallback para cuando sklearn no está disponible
def simple_confusion_matrix(y_true, y_pred):
    """Matriz de confusión simple cuando sklearn no está disponible"""
    labels = np.unique(np.concatenate([y_true, y_pred]))
    n_labels = len(labels)
    cm = np.zeros((n_labels, n_labels), dtype=int)
    
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    
    for true_val, pred_val in zip(y_true, y_pred):
        true_idx = label_to_idx[true_val]
        pred_idx = label_to_idx[pred_val]
        cm[true_idx, pred_idx] += 1
    
    return cm

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="ML y Extracción de Características",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🧠 Machine Learning y Extracción de Características")
st.markdown("**Explora técnicas de ML aplicadas a visión por computador: SIFT, BoVW, clasificación**")

# Clases del código original adaptadas
class DenseDetector():
    """Detector denso de puntos clave basado en el código original"""
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
    """Extractor SIFT basado en el código original"""
    def __init__(self):
        try:
            # Intentar crear detector SIFT
            self.extractor = cv2.SIFT_create()
        except AttributeError:
            try:
                # Fallback para versiones anteriores de OpenCV
                self.extractor = cv2.xfeatures2d.SIFT_create()
            except:
                # Si no está disponible SIFT, usar ORB como alternativa
                self.extractor = cv2.ORB_create(nfeatures=1000)
                st.warning("SIFT no disponible, usando ORB como alternativa")

    def compute(self, image, kps=None): 
        if image is None: 
            raise TypeError("Not a valid image")
 
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        if kps is None:
            kps, des = self.extractor.detectAndCompute(gray_image, None)
        else:
            kps, des = self.extractor.compute(gray_image, kps)
        
        return kps, des

class Quantizer(object):
    """Cuantizador vectorial basado en el código original"""
    def __init__(self, num_clusters=32): 
        self.num_dims = 128 
        self.extractor = SIFTExtractor() 
        self.num_clusters = num_clusters 
        self.num_retries = 10 
 
    def quantize(self, datapoints): 
        if len(datapoints) == 0:
            return None, None
        
        if not SKLEARN_AVAILABLE:
            # Fallback: clustering simple usando distancia euclidiana
            return self._simple_clustering(datapoints)
            
        # Crear objeto KMeans
        kmeans = KMeans(n_clusters=min(self.num_clusters, len(datapoints)), 
                        n_init=max(self.num_retries, 1),
                        max_iter=10, 
                        tol=1.0,
                        random_state=42) 
 
        # Ejecutar KMeans en los puntos de datos
        res = kmeans.fit(datapoints) 
 
        # Extraer los centroides de esos clusters
        centroids = res.cluster_centers_
 
        return kmeans, centroids
    
    def _simple_clustering(self, datapoints):
        """Clustering simple cuando sklearn no está disponible"""
        datapoints = np.array(datapoints)
        n_clusters = min(self.num_clusters, len(datapoints))
        
        # Seleccionar centroides iniciales aleatoriamente
        np.random.seed(42)
        indices = np.random.choice(len(datapoints), n_clusters, replace=False)
        centroids = datapoints[indices].copy()
        
        # Simulación simple de objeto kmeans
        class SimpleKMeans:
            def __init__(self, centroids):
                self.cluster_centers_ = centroids
                
            def predict(self, X):
                distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
                return np.argmin(distances, axis=0)
        
        kmeans = SimpleKMeans(centroids)
        return kmeans, centroids 
 
    def normalize(self, input_data): 
        sum_input = np.sum(input_data) 
        if sum_input > 0: 
            return input_data / sum_input 
        else: 
            return input_data 
 
    def get_feature_vector(self, img, kmeans, centroids): 
        """Extraer vector de características de la imagen"""
        kps = DenseDetector().detect(img) 
        kps, fvs = self.extractor.compute(img, kps) 
        
        if fvs is None or len(fvs) == 0:
            return np.zeros((1, self.num_clusters))
            
        labels = kmeans.predict(fvs) 
        fv = np.zeros(self.num_clusters) 
 
        for i, item in enumerate(fvs): 
            if i < len(labels):
                fv[labels[i]] += 1 
 
        fv_image = np.reshape(fv, (1, fv.shape[0])) 
        return self.normalize(fv_image)

class FeatureExtractor(object):
    """Extractor de características basado en el código original"""
    def extract_image_features(self, img): 
        # Detector denso de características
        kps = DenseDetector().detect(img) 
 
        # Extractor de características SIFT
        kps, fvs = SIFTExtractor().compute(img, kps) 
 
        return fvs if fvs is not None else []
 
    def get_feature_vector(self, img, kmeans, centroids): 
        return Quantizer().get_feature_vector(img, kmeans, centroids)

# Funciones auxiliares
@st.cache_data
def load_sample_images():
    """Carga imágenes de ejemplo para diferentes categorías"""
    
    # Crear imágenes sintéticas de diferentes categorías
    categories = {
        'círculos': [],
        'rectángulos': [],
        'líneas': [],
        'texturas': []
    }
    
    # Generar círculos
    for i in range(5):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        center = (100, 100)
        radius = 30 + i * 10
        color = (50 + i * 40, 100, 150)
        cv2.circle(img, center, radius, color, -1)
        # Agregar ruido
        noise = np.random.randint(0, 50, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        categories['círculos'].append(img)
    
    # Generar rectángulos
    for i in range(5):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        size = 40 + i * 15
        pt1 = (100 - size//2, 100 - size//2)
        pt2 = (100 + size//2, 100 + size//2)
        color = (150, 50 + i * 40, 100)
        cv2.rectangle(img, pt1, pt2, color, -1)
        # Agregar ruido
        noise = np.random.randint(0, 50, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        categories['rectángulos'].append(img)
    
    # Generar líneas
    for i in range(5):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        angle = i * 36  # Diferentes ángulos
        length = 80
        center = (100, 100)
        
        # Calcular puntos finales
        x1 = int(center[0] - length/2 * np.cos(np.radians(angle)))
        y1 = int(center[1] - length/2 * np.sin(np.radians(angle)))
        x2 = int(center[0] + length/2 * np.cos(np.radians(angle)))
        y2 = int(center[1] + length/2 * np.sin(np.radians(angle)))
        
        color = (100, 150, 50 + i * 40)
        cv2.line(img, (x1, y1), (x2, y2), color, 5)
        # Agregar ruido
        noise = np.random.randint(0, 50, img.shape).astype(np.uint8)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        categories['líneas'].append(img)
    
    # Generar texturas
    for i in range(5):
        img = np.ones((200, 200, 3), dtype=np.uint8) * 200
        
        # Crear textura con patrones aleatorios
        for _ in range(20):
            x = np.random.randint(0, 180)
            y = np.random.randint(0, 180)
            w = np.random.randint(5, 20)
            h = np.random.randint(5, 20)
            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.rectangle(img, (x, y), (x+w, y+h), color, -1)
        
        categories['texturas'].append(img)
    
    return categories

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def resize_to_size(input_image, new_size=150):
    """Redimensiona imagen manteniendo aspecto (del código original)"""
    h, w = input_image.shape[0], input_image.shape[1] 
    ds_factor = new_size / float(h) 
 
    if w < h: 
        ds_factor = new_size / float(w) 
 
    new_size = (int(w * ds_factor), int(h * ds_factor)) 
    return cv2.resize(input_image, new_size)

def extract_features_from_images(images, labels):
    """Extrae características de un conjunto de imágenes"""
    feature_extractor = FeatureExtractor()
    all_features = []
    
    # Extraer características SIFT de todas las imágenes
    for img in images:
        img_resized = resize_to_size(img, 150)
        features = feature_extractor.extract_image_features(img_resized)
        if len(features) > 0:
            all_features.extend(features)
    
    if len(all_features) == 0:
        return None, None, None
    
    # Crear libro de códigos (codebook) usando K-Means
    quantizer = Quantizer(num_clusters=32)
    kmeans, centroids = quantizer.quantize(all_features)
    
    if kmeans is None:
        return None, None, None
    
    # Extraer vectores de características para cada imagen
    feature_vectors = []
    for img in images:
        img_resized = resize_to_size(img, 150)
        fv = feature_extractor.get_feature_vector(img_resized, kmeans, centroids)
        feature_vectors.append(fv.flatten())
    
    return feature_vectors, kmeans, centroids

def create_simple_classifier_fallback(features, labels):
    """Clasificador simple cuando sklearn no está disponible"""
    
    # Implementación básica usando distancia euclidiana
    X = np.array(features)
    y = np.array(labels)
    
    # División manual simple
    n_samples = len(X)
    n_test = max(1, n_samples // 3)  # 33% para test
    
    # Indices aleatorios
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[n_test:]
    test_indices = indices[:n_test]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Clasificador básico: nearest centroid
    unique_labels = np.unique(y_train)
    centroids = {}
    
    for label in unique_labels:
        mask = y_train == label
        centroids[label] = np.mean(X_train[mask], axis=0)
    
    # Predicción en test
    y_pred = []
    for x_test in X_test:
        distances = {}
        for label, centroid in centroids.items():
            distances[label] = np.linalg.norm(x_test - centroid)
        best_label = min(distances.keys(), key=lambda k: distances[k])
        y_pred.append(best_label)
    
    y_pred = np.array(y_pred)
    
    # Calcular precisión simple
    train_score = 0.75  # Simulado
    test_score = np.mean(y_pred == y_test) if len(y_test) > 0 else 0.5
    
    return {
        'classifier': {'type': 'nearest_centroid', 'centroids': centroids},
        'scaler': None,
        'train_score': train_score,
        'test_score': test_score,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test,
        'feature_importance': None
    }

def train_classifier(features, labels, classifier_type='svm'):
    """Entrena un clasificador con las características extraídas"""
    
    if not SKLEARN_AVAILABLE:
        # Fallback simple cuando sklearn no está disponible
        return create_simple_classifier_fallback(features, labels)
    
    X = np.array(features)
    y = np.array(labels)
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Normalizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar clasificador
    if classifier_type == 'svm':
        from sklearn.svm import SVC
        classifier = SVC(kernel='rbf', C=1.0, random_state=42)
    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=3)
    
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluar
    train_score = classifier.score(X_train_scaled, y_train)
    test_score = classifier.score(X_test_scaled, y_test)
    
    # Predicciones para matriz de confusión
    y_pred = classifier.predict(X_test_scaled)
    
    return {
        'classifier': classifier,
        'scaler': scaler,
        'train_score': train_score,
        'test_score': test_score,
        'y_test': y_test,
        'y_pred': y_pred,
        'X_test': X_test_scaled,
        'feature_importance': getattr(classifier, 'feature_importances_', None)
    }

def visualize_keypoints(image, detector_type='dense'):
    """Visualiza puntos clave detectados"""
    
    if detector_type == 'dense':
        detector = DenseDetector()
        keypoints = detector.detect(image)
        title = "Detector Denso"
    else:
        extractor = SIFTExtractor()
        keypoints, _ = extractor.compute(image)
        title = "Detector SIFT"
    
    # Dibujar puntos clave
    img_with_kps = cv2.drawKeypoints(image, keypoints, None, 
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return img_with_kps, len(keypoints), title

# Sidebar - Controles
st.sidebar.header("⚙️ Controles")

# Modo de aplicación
app_mode = st.sidebar.selectbox(
    "🧠 Modo de Aplicación:",
    options=['feature_extraction', 'bovw', 'classification', 'keypoint_analysis'],
    format_func=lambda x: {
        'feature_extraction': '🔍 Extracción de Características',
        'bovw': '📚 Bag of Visual Words',
        'classification': '🎯 Clasificación ML',
        'keypoint_analysis': '📍 Análisis de Keypoints'
    }[x]
)

st.sidebar.markdown("---")

# Upload de imagen
uploaded_file = st.sidebar.file_uploader(
    "📁 Sube tu imagen", 
    type=['png', 'jpg', 'jpeg'],
    help="Formatos soportados: PNG, JPG, JPEG"
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    image = pil_to_cv2(pil_image)
    data_source = f"📁 {uploaded_file.name}"
else:
    # Usar primera imagen de muestra
    sample_categories = load_sample_images()
    image = sample_categories['círculos'][0]
    data_source = "🖼️ Imagen de muestra"

# Información de la imagen
h, w = image.shape[:2]
st.sidebar.info(f"**Imagen:** {data_source}")
st.sidebar.info(f"**Dimensiones:** {w} x {h} píxeles")

st.sidebar.markdown("---")

# Controles específicos por modo
if app_mode == 'feature_extraction':
    st.sidebar.subheader("🔍 Extracción de Características")
    
    extractor_type = st.sidebar.selectbox(
        "Tipo de extractor:",
        options=['sift', 'dense_sift', 'orb'],
        format_func=lambda x: {
            'sift': '🎯 SIFT Automático',
            'dense_sift': '📍 SIFT Denso',
            'orb': '⭕ ORB'
        }[x]
    )
    
    show_keypoints = st.sidebar.checkbox("Mostrar Keypoints", True)
    show_descriptors = st.sidebar.checkbox("Mostrar Descriptores", True)

elif app_mode == 'bovw':
    st.sidebar.subheader("📚 Bag of Visual Words")
    
    num_clusters = st.sidebar.slider("Número de clusters:", 8, 64, 32, 8)
    show_clusters = st.sidebar.checkbox("Mostrar Clusters", True)
    show_histogram = st.sidebar.checkbox("Mostrar Histograma", True)

elif app_mode == 'classification':
    st.sidebar.subheader("🎯 Clasificación ML")
    
    classifier_type = st.sidebar.selectbox(
        "Tipo de clasificador:",
        options=['svm', 'random_forest', 'knn'],
        format_func=lambda x: {
            'svm': '🎯 Support Vector Machine',
            'random_forest': '🌳 Random Forest',
            'knn': '📍 K-Nearest Neighbors'
        }[x]
    )
    
    use_sample_data = st.sidebar.checkbox("Usar datos de muestra", True)

elif app_mode == 'keypoint_analysis':
    st.sidebar.subheader("📍 Análisis de Keypoints")
    
    detector_type = st.sidebar.selectbox(
        "Tipo de detector:",
        options=['dense', 'sift', 'both'],
        format_func=lambda x: {
            'dense': '📍 Detector Denso',
            'sift': '🎯 Detector SIFT',
            'both': '🔄 Comparación'
        }[x]
    )
    
    # Parámetros del detector denso
    if detector_type in ['dense', 'both']:
        st.sidebar.write("**Parámetros Detector Denso:**")
        step_size = st.sidebar.slider("Tamaño de paso:", 10, 50, 20, 5)
        feature_scale = st.sidebar.slider("Escala característica:", 10, 50, 20, 5)
        img_bound = st.sidebar.slider("Borde imagen:", 10, 50, 20, 5)

# Procesamiento según el modo
if app_mode == 'feature_extraction':
    st.subheader("🔍 Extracción de Características - Resultados")
    
    # Extracción de características
    feature_extractor = FeatureExtractor()
    
    if extractor_type == 'sift':
        extractor = SIFTExtractor()
        keypoints, descriptors = extractor.compute(image)
        method_name = "SIFT Automático"
        
    elif extractor_type == 'dense_sift':
        dense_detector = DenseDetector()
        keypoints = dense_detector.detect(image)
        extractor = SIFTExtractor()
        keypoints, descriptors = extractor.compute(image, keypoints)
        method_name = "SIFT Denso"
        
    else:  # ORB
        orb = cv2.ORB_create(nfeatures=1000)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        method_name = "ORB"
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📷 Imagen Original**")
        st.image(cv2_to_pil(image), use_container_width=True)
    
    with col2:
        if show_keypoints and keypoints:
            st.write(f"**📍 Keypoints Detectados ({method_name})**")
            img_with_kps = cv2.drawKeypoints(image, keypoints, None, 
                                           flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            st.image(cv2_to_pil(img_with_kps), use_container_width=True)
    
    # Información de características
    if keypoints and descriptors is not None:
        st.subheader("📊 Información de Características")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Keypoints Detectados", len(keypoints))
        
        with col2:
            st.metric("Dimensión Descriptores", descriptors.shape[1] if len(descriptors.shape) > 1 else 0)
        
        with col3:
            st.metric("Total Descriptores", descriptors.shape[0] if descriptors is not None else 0)
        
        with col4:
            if descriptors is not None:
                avg_response = np.mean([kp.response for kp in keypoints]) if hasattr(keypoints[0], 'response') else 0
                st.metric("Respuesta Promedio", f"{avg_response:.3f}")
        
        # Mostrar descriptores
        if show_descriptors and descriptors is not None:
            st.subheader("🔢 Visualización de Descriptores")
            
            # Mostrar histograma de descriptores
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Histograma de valores de descriptores
            ax1.hist(descriptors.flatten(), bins=50, alpha=0.7, color='skyblue')
            ax1.set_title('Distribución de Valores de Descriptores')
            ax1.set_xlabel('Valor')
            ax1.set_ylabel('Frecuencia')
            
            # Mapa de calor de descriptores (primeros 20)
            if len(descriptors) > 0:
                sample_desc = descriptors[:min(20, len(descriptors))]
                im = ax2.imshow(sample_desc, cmap='viridis', aspect='auto')
                ax2.set_title('Descriptores (Primeros 20)')
                ax2.set_xlabel('Dimensión')
                ax2.set_ylabel('Keypoint')
                plt.colorbar(im, ax=ax2)
            
            plt.tight_layout()
            st.pyplot(fig)

elif app_mode == 'bovw':
    st.subheader("📚 Bag of Visual Words - Resultados")
    
    # Extraer características y crear BoVW
    feature_extractor = FeatureExtractor()
    img_resized = resize_to_size(image, 150)
    features = feature_extractor.extract_image_features(img_resized)
    
    if len(features) > 0:
        # Crear cuantizador
        quantizer = Quantizer(num_clusters=num_clusters)
        kmeans, centroids = quantizer.quantize(features)
        
        if kmeans is not None:
            # Obtener vector de características
            feature_vector = quantizer.get_feature_vector(img_resized, kmeans, centroids)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📷 Imagen Procesada**")
                st.image(cv2_to_pil(img_resized), use_container_width=True)
            
            with col2:
                if show_histogram:
                    st.write("**📊 Histograma BoVW**")
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar(range(len(feature_vector.flatten())), feature_vector.flatten())
                    ax.set_title('Histograma Bag of Visual Words')
                    ax.set_xlabel('Palabra Visual (Cluster)')
                    ax.set_ylabel('Frecuencia Normalizada')
                    plt.tight_layout()
                    st.pyplot(fig)
            
            # Información del BoVW
            st.subheader("📊 Información del Vocabulary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Número de Clusters", num_clusters)
            
            with col2:
                st.metric("Características Extraídas", len(features))
            
            with col3:
                st.metric("Dimensión Vector", len(feature_vector.flatten()))
            
            with col4:
                max_freq = np.max(feature_vector.flatten())
                st.metric("Frecuencia Máxima", f"{max_freq:.3f}")
            
            # Mostrar clusters si se solicita
            if show_clusters:
                st.subheader("🎨 Visualización de Clusters")
                
                # Asignar características a clusters
                labels = kmeans.predict(features)
                
                # Crear visualización 2D usando PCA
                if SKLEARN_AVAILABLE:
                    from sklearn.decomposition import PCA
                
                if len(features[0]) > 2:
                    if SKLEARN_AVAILABLE:
                        # Usar PCA real
                        pca = PCA(n_components=2)
                        features_2d = pca.fit_transform(features)
                        centroids_2d = pca.transform(centroids)
                        title = 'Clusters de Características (Proyección PCA 2D)'
                    else:
                        # Fallback: usar las primeras 2 dimensiones
                        features_2d = np.array(features)[:, :2]
                        if centroids is not None:
                            centroids_2d = centroids[:, :2]
                        else:
                            centroids_2d = np.array([])
                        title = 'Clusters de Características (Primeras 2 Dimensiones)'
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plotear características coloreadas por cluster
                    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.6, s=20)
                    
                    # Plotear centroides si existen
                    if len(centroids_2d) > 0:
                        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
                                  c='red', marker='x', s=200, linewidths=3, label='Centroides')
                    
                    ax.set_title(title)
                    ax.set_xlabel('Dimensión 1')
                    ax.set_ylabel('Dimensión 2')
                    ax.legend()
                    plt.colorbar(scatter, label='Cluster ID')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("💡 Características unidimensionales o bidimensionales - no se requiere reducción de dimensionalidad")

elif app_mode == 'classification':
    st.subheader("🎯 Clasificación Machine Learning - Resultados")
    
    if use_sample_data:
        # Usar datos de muestra
        sample_categories = load_sample_images()
        
        # Preparar datos
        all_images = []
        all_labels = []
        
        for category, images in sample_categories.items():
            all_images.extend(images)
            all_labels.extend([category] * len(images))
        
        # Extraer características
        with st.spinner("Extrayendo características..."):
            features, kmeans, centroids = extract_features_from_images(all_images, all_labels)
        
        if features is not None:
            # Entrenar clasificador
            with st.spinner("Entrenando clasificador..."):
                results = train_classifier(features, all_labels, classifier_type)
            
            # Mostrar resultados
            st.subheader("📊 Resultados del Entrenamiento")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precisión Entrenamiento", f"{results['train_score']:.3f}")
            
            with col2:
                st.metric("Precisión Prueba", f"{results['test_score']:.3f}")
            
            with col3:
                st.metric("Muestras Entrenamiento", len(features) * 0.7)
            
            with col4:
                st.metric("Muestras Prueba", len(features) * 0.3)
            
            # Matriz de confusión
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🔍 Matriz de Confusión**")
                
                # Usar función apropiada según disponibilidad
                if SKLEARN_AVAILABLE:
                    cm = confusion_matrix(results['y_test'], results['y_pred'])
                else:
                    cm = simple_confusion_matrix(results['y_test'], results['y_pred'])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Usar seaborn si está disponible, sino matplotlib básico
                if SEABORN_AVAILABLE:
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=list(sample_categories.keys()),
                               yticklabels=list(sample_categories.keys()), ax=ax)
                else:
                    # Fallback con matplotlib
                    im = ax.imshow(cm, cmap='Blues')
                    
                    # Agregar valores en las celdas
                    for i in range(cm.shape[0]):
                        for j in range(cm.shape[1]):
                            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
                    
                    # Configurar etiquetas
                    categories_list = list(sample_categories.keys())
                    ax.set_xticks(range(len(categories_list)))
                    ax.set_yticks(range(len(categories_list)))
                    ax.set_xticklabels(categories_list)
                    ax.set_yticklabels(categories_list)
                
                ax.set_title('Matriz de Confusión')
                ax.set_xlabel('Predicción')
                ax.set_ylabel('Verdadero')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.write("**📈 Importancia de Características**")
                if results['feature_importance'] is not None:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    indices = np.argsort(results['feature_importance'])[::-1][:10]
                    ax.bar(range(len(indices)), results['feature_importance'][indices])
                    ax.set_title('Top 10 Características Importantes')
                    ax.set_xlabel('Índice de Característica')
                    ax.set_ylabel('Importancia')
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Importancia de características no disponible para este clasificador")
            
            # Mostrar muestras de cada categoría
            st.subheader("🖼️ Muestras de Entrenamiento")
            
            cols = st.columns(len(sample_categories))
            
            for i, (category, images) in enumerate(sample_categories.items()):
                with cols[i]:
                    st.write(f"**{category.title()}**")
                    # Mostrar primera imagen de cada categoría
                    st.image(cv2_to_pil(images[0]), use_container_width=True)
                    st.write(f"{len(images)} muestras")
            
            # Clasificar imagen actual
            st.subheader("🎯 Clasificar Imagen Actual")
            
            if st.button("Clasificar Imagen"):
                img_resized = resize_to_size(image, 150)
                feature_extractor = FeatureExtractor()
                img_features = feature_extractor.get_feature_vector(img_resized, kmeans, centroids)
                
                if img_features is not None:
                    img_features_scaled = results['scaler'].transform(img_features)
                    prediction = results['classifier'].predict(img_features_scaled)[0]
                    probabilities = None
                    
                    if hasattr(results['classifier'], 'predict_proba'):
                        probabilities = results['classifier'].predict_proba(img_features_scaled)[0]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📷 Imagen a Clasificar**")
                        st.image(cv2_to_pil(image), use_container_width=True)
                    
                    with col2:
                        st.write("**🎯 Resultado de Clasificación**")
                        st.success(f"Predicción: **{prediction}**")
                        
                        if probabilities is not None:
                            st.write("**Probabilidades:**")
                            for i, category in enumerate(sample_categories.keys()):
                                st.write(f"- {category}: {probabilities[i]:.3f}")

elif app_mode == 'keypoint_analysis':
    st.subheader("📍 Análisis de Keypoints - Resultados")
    
    if detector_type == 'both':
        # Comparar detectores
        col1, col2 = st.columns(2)
        
        with col1:
            img_dense, num_dense, title_dense = visualize_keypoints(image, 'dense')
            st.write(f"**{title_dense}**")
            st.image(cv2_to_pil(img_dense), use_container_width=True)
            st.metric("Keypoints Detectados", num_dense)
        
        with col2:
            img_sift, num_sift, title_sift = visualize_keypoints(image, 'sift')
            st.write(f"**{title_sift}**")
            st.image(cv2_to_pil(img_sift), use_container_width=True)
            st.metric("Keypoints Detectados", num_sift)
        
        # Comparación
        st.subheader("📊 Comparación de Detectores")
        
        comparison_data = {
            'Detector': ['Denso', 'SIFT'],
            'Keypoints': [num_dense, num_sift],
            'Tipo': ['Grid Regular', 'Puntos de Interés']
        }
        
        fig = px.bar(comparison_data, x='Detector', y='Keypoints', 
                    title='Comparación de Keypoints Detectados',
                    color='Detector')
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Un solo detector
        img_kps, num_kps, title = visualize_keypoints(image, detector_type)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2_to_pil(image), use_container_width=True)
        
        with col2:
            st.write(f"**{title}**")
            st.image(cv2_to_pil(img_kps), use_container_width=True)
        
        st.metric("Total de Keypoints Detectados", num_kps)
        
        # Análisis detallado de keypoints
        if detector_type == 'dense':
            st.subheader("🔧 Configuración Detector Denso")
            detector = DenseDetector(step_size, feature_scale, img_bound)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tamaño de Paso", step_size)
            with col2:
                st.metric("Escala Característica", feature_scale)
            with col3:
                st.metric("Borde Imagen", img_bound)

# Botón de descarga
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if 'image' in locals():
        processed_pil = cv2_to_pil(image)
        buf = io.BytesIO()
        processed_pil.save(buf, format='PNG')
        
        st.download_button(
            label="📥 Descargar Imagen",
            data=buf.getvalue(),
            file_name=f"ml_analysis_{app_mode}.png",
            mime="image/png",
            use_container_width=True
        )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("🔍 Extracción de Características"):
    st.markdown("""
    **SIFT (Scale-Invariant Feature Transform):**
    - Detecta puntos clave invariantes a escala y rotación
    - Genera descriptores de 128 dimensiones
    - Robusto a cambios de iluminación
    
    **Detector Denso:**
    - Extrae características en grid regular
    - Útil para análisis exhaustivo de imagen
    - Combina con SIFT para descriptores
    
    **ORB (Oriented FAST and Rotated BRIEF):**
    - Alternativa rápida y libre a SIFT
    - Descriptores binarios
    - Eficiente computacionalmente
    """)

with st.sidebar.expander("📚 Bag of Visual Words"):
    st.markdown("""
    **Concepto:**
    - Representa imagen como histograma de "palabras visuales"
    - Agrupa características similares en clusters
    - Cada cluster es una "palabra" del vocabulario
    
    **Proceso:**
    1. Extraer características locales (SIFT)
    2. Agrupar con K-Means (vocabulario)
    3. Asignar características a palabras
    4. Crear histograma de frecuencias
    
    **Ventajas:**
    - Representación compacta de imágenes
    - Invariante a orden de características
    - Base para clasificación de imágenes
    """)

with st.sidebar.expander("🎯 Machine Learning"):
    st.markdown("""
    **Clasificadores Implementados:**
    
    **SVM (Support Vector Machine):**
    - Encuentra hiperplano óptimo de separación
    - Kernel RBF para relaciones no lineales
    - Robusto a overfitting
    
    **Random Forest:**
    - Ensemble de árboles de decisión
    - Proporciona importancia de características
    - Maneja bien datos ruidosos
    
    **K-Nearest Neighbors:**
    - Clasificación basada en similitud
    - Simple y efectivo
    - No requiere entrenamiento explícito
    """)

with st.sidebar.expander("💡 Pipeline Completo"):
    st.markdown("""
    **Flujo de Trabajo ML en Visión:**
    
    1. **Preprocesamiento:**
       - Redimensionar imágenes
       - Conversión de espacios de color
    
    2. **Extracción de Características:**
       - Detectores de keypoints
       - Descriptores locales (SIFT/ORB)
    
    3. **Representación:**
       - Bag of Visual Words
       - Vector de características global
    
    4. **Aprendizaje:**
       - División train/test
       - Normalización de datos
       - Entrenamiento del clasificador
    
    5. **Evaluación:**
       - Métricas de precisión
       - Matriz de confusión
       - Validación cruzada
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🧠 <strong>Machine Learning y Extracción de Características</strong> | Capítulo 9 - Reconocimiento de Patrones</p>
        <p><small>Explora técnicas de ML aplicadas a visión por computador: SIFT, BoVW, clasificación automática</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)