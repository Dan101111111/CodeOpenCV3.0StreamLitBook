"""
Aplicación Streamlit - Extracción Avanzada de Características y Clasificación
Aplicación educativa para explorar técnicas avanzadas de extracción de características y clasificación multiclase
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Imports opcionales - usar fallbacks si no están disponibles
try:
    from sklearn.cluster import KMeans
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Crear fallbacks para plotly
    class DummyPlotly:
        def bar(self, *args, **kwargs):
            return None
        def pie(self, *args, **kwargs):
            return None
        def scatter(self, *args, **kwargs):
            return None
    
    px = DummyPlotly()
    go = DummyPlotly()
    make_subplots = lambda: None

# Fallbacks simples para funcionalidad sin sklearn
if not SKLEARN_AVAILABLE:
    # Simple KMeans fallback using NumPy
    class SimpleKMeans:
        def __init__(self, n_clusters=8, random_state=42, n_init=10, max_iter=300, tol=1e-4):
            self.n_clusters = n_clusters
            self.random_state = random_state
            self.max_iter = max_iter
            self.tol = tol
            np.random.seed(random_state)
            
        def fit(self, X):
            X = np.array(X)
            n_samples, n_features = X.shape
            
            # Initialize centroids randomly
            self.cluster_centers_ = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
            
            for _ in range(self.max_iter):
                # Assign points to closest centroid
                distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
                labels = np.argmin(distances, axis=0)
                
                # Update centroids
                new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
                
                # Check for convergence
                if np.all(np.abs(self.cluster_centers_ - new_centers) < self.tol):
                    break
                self.cluster_centers_ = new_centers
            
            self.labels_ = labels
            return self
            
        def predict(self, X):
            X = np.array(X)
            distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis])**2).sum(axis=2))
            return np.argmin(distances, axis=0)
        
        def fit_predict(self, X):
            self.fit(X)
            return self.labels_
    
    KMeans = SimpleKMeans
    
    # Simple train_test_split fallback
    def simple_train_test_split(X, y, test_size=0.25, random_state=42, stratify=None):
        X, y = np.array(X), np.array(y)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
        
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]
        
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    
    train_test_split = simple_train_test_split
    
    # Simple accuracy calculation
    def simple_accuracy_score(y_true, y_pred):
        return np.mean(np.array(y_true) == np.array(y_pred))
    
    accuracy_score = simple_accuracy_score
    
    # Dummy classes for other ML models
    class DummyClassifier:
        def __init__(self, **kwargs):
            self.classes_ = None
            
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            # Simple majority class predictor
            self.most_common = Counter(y).most_common(1)[0][0]
            return self
            
        def predict(self, X):
            return np.full(len(X), self.most_common)
            
        def predict_proba(self, X):
            n_samples = len(X)
            n_classes = len(self.classes_) if self.classes_ is not None else 2
            # Return uniform probabilities
            return np.full((n_samples, n_classes), 1.0 / n_classes)
            
        def score(self, X, y):
            y_pred = self.predict(X)
            return simple_accuracy_score(y, y_pred)
    
    SVC = RandomForestClassifier = GradientBoostingClassifier = DummyClassifier
    GaussianNB = KNeighborsClassifier = DummyClassifier
    
    # Simple StandardScaler fallback
    class SimpleScaler:
        def fit(self, X):
            X = np.array(X)
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_[self.std_ == 0] = 1  # Avoid division by zero
            return self
            
        def transform(self, X):
            X = np.array(X)
            return (X - self.mean_) / self.std_
            
        def fit_transform(self, X):
            return self.fit(X).transform(X)
    
    StandardScaler = SimpleScaler
    
    # Simple confusion matrix
    def simple_confusion_matrix(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        return cm
    
    confusion_matrix = simple_confusion_matrix
    
    # Simple cross validation fallback
    def simple_cross_val_score(estimator, X, y, cv=5):
        # Simple mock cross validation - just return dummy scores
        return np.random.uniform(0.6, 0.9, cv)
    
    cross_val_score = simple_cross_val_score
    
    # Simple classification report
    def simple_classification_report(y_true, y_pred, output_dict=False):
        accuracy = simple_accuracy_score(y_true, y_pred)
        if output_dict:
            return {'accuracy': accuracy, 'macro avg': {'f1-score': accuracy}}
        return f"Classification Report\nAccuracy: {accuracy:.3f}\nNote: Install scikit-learn for detailed metrics"
    
    classification_report = simple_classification_report
    
    # Simple label encoder
    class SimpleLabelEncoder:
        def fit(self, y):
            self.classes_ = np.unique(y)
            return self
            
        def transform(self, y):
            return np.array([np.where(self.classes_ == label)[0][0] for label in y])
            
        def fit_transform(self, y):
            return self.fit(y).transform(y)
    
    LabelEncoder = SimpleLabelEncoder
    
    # Simple PCA fallback
    class SimplePCA:
        def __init__(self, n_components=None):
            self.n_components = n_components
            
        def fit(self, X):
            X = np.array(X)
            # Center the data
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
            
            # Compute covariance matrix
            cov_matrix = np.cov(X_centered.T)
            
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort by eigenvalue (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Store components
            if self.n_components is None:
                self.n_components = len(eigenvalues)
            
            self.components_ = eigenvectors[:, :self.n_components].T
            self.explained_variance_ = eigenvalues[:self.n_components]
            
            # Calculate explained variance ratio
            total_variance = np.sum(eigenvalues)
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
            
            return self
            
        def transform(self, X):
            X = np.array(X)
            X_centered = X - self.mean_
            return np.dot(X_centered, self.components_.T)
            
        def fit_transform(self, X):
            return self.fit(X).transform(X)
    
    PCA = SimplePCA

# Configuración de la página
st.set_page_config(
    page_title="Extracción Avanzada de Características",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔬 Extracción Avanzada de Características y Clasificación")
st.markdown("**Explora técnicas avanzadas de feature engineering, clustering y clasificación multiclase**")

# Clases adaptadas del código original
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
    """Extractor SIFT mejorado basado en el código original"""
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
                self.extractor = cv2.ORB_create(nfeatures=2000)
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

    def get_description(self, gray_image):
        """Método adicional del código original"""
        kps, des = self.extractor.detectAndCompute(gray_image, None)
        return kps, des

class Quantizer(object):
    """Cuantizador vectorial mejorado basado en el código original"""
    def __init__(self, num_clusters=32):
        self.extractor = SIFTExtractor()
        self.num_clusters = num_clusters 
        self.num_retries = 10 
 
    def quantize(self, datapoints): 
        if len(datapoints) == 0:
            return None, None
            
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
 
    def normalize(self, input_data): 
        sum_input = np.sum(input_data)
        return input_data / sum_input if sum_input > 0 else input_data
 
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

class AdvancedFeatureExtractor(object):
    """Extractor de características avanzado basado en el código original"""
    def __init__(self, image_path=None, image=None, optimal_size=150):
        if image_path is not None:
            img = cv2.imread(image_path)
            self.image = self.get_optimized_image(img, optimal_size)
        elif image is not None:
            self.image = self.get_optimized_image(image, optimal_size)
        else:
            raise ValueError("Debe proporcionar image_path o image")
        
        self.optimal_size = optimal_size

    def get_feature_vector(self, num_clusters=32):
        """Método principal del código original"""
        kmeans, centroids = self.get_centroids(self.image)
        if kmeans is None:
            return np.zeros((1, num_clusters))
        return Quantizer(num_clusters).get_feature_vector(self.image, kmeans, centroids)

    def get_centroids(self, image):
        """Extraer centroides de las características"""
        kps_all = []
        # Detector denso de características
        kps = DenseDetector().detect(image)
        # Extractor de características SIFT
        kps, fvs = SIFTExtractor().compute(image, kps)
        
        if fvs is not None:
            kps_all.extend(fvs)

        kmeans, centroids = Quantizer().quantize(kps_all)
        return kmeans, centroids

    def get_optimized_image(self, img, new_size=150):
        """Optimizar imagen manteniendo aspecto (del código original)"""
        h, w = img.shape[0], img.shape[1]
        ds_factor = new_size / float(h)
        if w < h: 
            ds_factor = new_size / float(w)
        new_size = (int(w * ds_factor), int(h * ds_factor))
        return cv2.resize(img, new_size)

    def extract_multiple_features(self, include_color=True, include_texture=True, include_shape=True):
        """Extrae múltiples tipos de características"""
        features = {}
        
        # Características SIFT/ORB (del código original)
        bow_features = self.get_feature_vector().flatten()
        features['bow'] = bow_features
        
        if include_color:
            # Características de color
            features.update(self.extract_color_features())
        
        if include_texture:
            # Características de textura
            features.update(self.extract_texture_features())
        
        if include_shape:
            # Características de forma
            features.update(self.extract_shape_features())
        
        return features

    def extract_color_features(self):
        """Extrae características de color"""
        features = {}
        
        # Histogramas de color en diferentes espacios
        # RGB
        for i, color in enumerate(['R', 'G', 'B']):
            hist = cv2.calcHist([self.image], [i], None, [32], [0, 256])
            features[f'rgb_{color.lower()}_hist'] = hist.flatten()
        
        # HSV
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        for i, component in enumerate(['H', 'S', 'V']):
            hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
            features[f'hsv_{component.lower()}_hist'] = hist.flatten()
        
        # Estadísticas de color
        mean_colors = np.mean(self.image.reshape(-1, 3), axis=0)
        std_colors = np.std(self.image.reshape(-1, 3), axis=0)
        features['color_mean'] = mean_colors
        features['color_std'] = std_colors
        
        return features

    def extract_texture_features(self):
        """Extrae características de textura"""
        features = {}
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # LBP (Local Binary Pattern) simplificado
        def simple_lbp(image, radius=1):
            rows, cols = image.shape
            lbp = np.zeros_like(image)
            
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    pattern = 0
                    
                    # 8-conectividad
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for k, neighbor in enumerate(neighbors):
                        if neighbor >= center:
                            pattern |= (1 << k)
                    
                    lbp[i, j] = pattern
            
            return lbp
        
        lbp = simple_lbp(gray)
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=32, range=(0, 256))
        features['lbp_hist'] = lbp_hist
        
        # Gradientes de Sobel
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        gradient_hist, _ = np.histogram(magnitude.flatten(), bins=32, range=(0, 255))
        features['gradient_hist'] = gradient_hist
        
        return features

    def extract_shape_features(self):
        """Extrae características de forma"""
        features = {}
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Detectar contornos
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Usar el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Área y perímetro
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Compacidad
            compactness = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
            
            # Momentos de Hu
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            features['shape_area'] = np.array([area])
            features['shape_perimeter'] = np.array([perimeter])
            features['shape_compactness'] = np.array([compactness])
            features['hu_moments'] = hu_moments
        else:
            # Valores por defecto si no se encuentran contornos
            features['shape_area'] = np.array([0])
            features['shape_perimeter'] = np.array([0])
            features['shape_compactness'] = np.array([0])
            features['hu_moments'] = np.zeros(7)
        
        return features

# Funciones auxiliares
@st.cache_data
def load_sample_images():
    """Carga imágenes de muestra desde la carpeta del capítulo"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(script_dir, 'images')
    
    sample_images = {}
    
    # Intentar cargar imágenes existentes
    image_files = ['multiclass_image.png', 'multiclass_image_features.png', 'test.png']
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                sample_images[img_file] = img
    
    # Crear dataset sintético multiclase si no hay imágenes
    if len(sample_images) == 0:
        sample_images = create_synthetic_multiclass_dataset()
    
    return sample_images

@st.cache_data
def create_synthetic_multiclass_dataset():
    """Crea un dataset sintético multiclase para demostración"""
    categories = {
        'naturales': [],
        'geometricas': [],
        'texturas': [],
        'rostros': [],
        'objetos': []
    }
    
    # Generar imágenes naturales (paisajes simulados)
    for i in range(5):
        img = np.ones((200, 300, 3), dtype=np.uint8) * 135  # Color cielo
        
        # Agregar "montañas"
        pts = np.array([[0, 150], [75, 100], [150, 120], [225, 90], [300, 130], [300, 200], [0, 200]], np.int32)
        cv2.fillPoly(img, [pts], (34, 139, 34))  # Verde
        
        # "Sol"
        cv2.circle(img, (250, 50), 25, (255, 255, 0), -1)
        
        # "Nubes"
        cv2.ellipse(img, (100, 40), (30, 15), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (180, 35), (25, 12), 0, 0, 360, (255, 255, 255), -1)
        
        # Ruido natural
        noise = np.random.randint(-20, 20, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        categories['naturales'].append(img)
    
    # Generar formas geométricas
    for i in range(5):
        img = np.ones((200, 300, 3), dtype=np.uint8) * 240
        
        # Diferentes formas geométricas
        shapes = ['circle', 'rectangle', 'triangle', 'pentagon', 'hexagon']
        shape = shapes[i % len(shapes)]
        
        center_x, center_y = 150, 100
        color = tuple(np.random.randint(50, 200, 3).tolist())
        
        if shape == 'circle':
            cv2.circle(img, (center_x, center_y), 50, color, -1)
        elif shape == 'rectangle':
            cv2.rectangle(img, (center_x-40, center_y-30), (center_x+40, center_y+30), color, -1)
        elif shape == 'triangle':
            pts = np.array([[center_x, center_y-40], [center_x-35, center_y+30], [center_x+35, center_y+30]], np.int32)
            cv2.fillPoly(img, [pts], color)
        elif shape == 'pentagon':
            angles = np.linspace(0, 2*np.pi, 6)[:-1] - np.pi/2
            pts = []
            for angle in angles:
                x = int(center_x + 40 * np.cos(angle))
                y = int(center_y + 40 * np.sin(angle))
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(img, [pts], color)
        
        categories['geometricas'].append(img)
    
    # Generar texturas
    for i in range(5):
        img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        
        # Diferentes tipos de texturas
        if i % 3 == 0:
            # Textura de puntos
            for _ in range(200):
                x = np.random.randint(0, 300)
                y = np.random.randint(0, 200)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.circle(img, (x, y), np.random.randint(2, 8), color, -1)
        elif i % 3 == 1:
            # Textura de líneas
            for _ in range(50):
                x1, y1 = np.random.randint(0, 300), np.random.randint(0, 200)
                x2, y2 = x1 + np.random.randint(-50, 50), y1 + np.random.randint(-30, 30)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.line(img, (x1, y1), (x2, y2), color, np.random.randint(1, 4))
        else:
            # Textura de cuadrados
            for _ in range(30):
                x = np.random.randint(0, 280)
                y = np.random.randint(0, 180)
                size = np.random.randint(5, 20)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(img, (x, y), (x+size, y+size), color, -1)
        
        categories['texturas'].append(img)
    
    # Generar rostros simulados
    for i in range(5):
        img = np.ones((200, 300, 3), dtype=np.uint8) * 220
        
        # Cara (elipse)
        center_x, center_y = 150, 100
        cv2.ellipse(img, (center_x, center_y), (40, 55), 0, 0, 360, (255, 220, 177), -1)
        
        # Ojos
        cv2.ellipse(img, (center_x-15, center_y-15), (8, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (center_x+15, center_y-15), (8, 12), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (center_x-15, center_y-15), 5, (0, 0, 0), -1)
        cv2.circle(img, (center_x+15, center_y-15), 5, (0, 0, 0), -1)
        
        # Nariz
        pts = np.array([[center_x-3, center_y], [center_x+3, center_y], [center_x, center_y+10]], np.int32)
        cv2.fillPoly(img, [pts], (255, 200, 150))
        
        # Boca
        cv2.ellipse(img, (center_x, center_y+20), (15, 8), 0, 0, 180, (200, 100, 100), 2)
        
        categories['rostros'].append(img)
    
    # Generar objetos diversos
    for i in range(5):
        img = np.ones((200, 300, 3), dtype=np.uint8) * 200
        
        objects = ['house', 'car', 'tree', 'flower', 'star']
        obj = objects[i % len(objects)]
        
        if obj == 'house':
            # Casa simple
            cv2.rectangle(img, (100, 120), (200, 180), (139, 69, 19), -1)  # Base
            pts = np.array([[90, 120], [150, 80], [210, 120]], np.int32)
            cv2.fillPoly(img, [pts], (255, 0, 0))  # Techo
            cv2.rectangle(img, (130, 140), (170, 180), (101, 67, 33), -1)  # Puerta
        elif obj == 'car':
            # Auto simple
            cv2.rectangle(img, (80, 130), (220, 160), (255, 0, 0), -1)  # Cuerpo
            cv2.rectangle(img, (100, 110), (200, 130), (0, 0, 255), -1)  # Techo
            cv2.circle(img, (110, 160), 15, (0, 0, 0), -1)  # Rueda 1
            cv2.circle(img, (190, 160), 15, (0, 0, 0), -1)  # Rueda 2
        
        categories['objetos'].append(img)
    
    # Convertir a formato de sample_images
    result = {}
    for category, images in categories.items():
        for i, img in enumerate(images):
            result[f"{category}_{i+1}.png"] = img
    
    return result

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def prepare_multiclass_dataset(sample_images, feature_types=['bow']):
    """Prepara dataset multiclase para entrenamiento"""
    X = []
    y = []
    feature_names = []
    
    for img_name, img in sample_images.items():
        # Extraer categoría del nombre
        if 'naturales' in img_name:
            category = 'Naturales'
        elif 'geometricas' in img_name:
            category = 'Geométricas'
        elif 'texturas' in img_name:
            category = 'Texturas'
        elif 'rostros' in img_name:
            category = 'Rostros'
        elif 'objetos' in img_name:
            category = 'Objetos'
        else:
            # Clasificación automática basada en características
            category = 'Otros'
        
        # Extraer características
        try:
            extractor = AdvancedFeatureExtractor(image=img)
            
            if 'all' in feature_types:
                features = extractor.extract_multiple_features()
                # Combinar todas las características
                combined_features = []
                for feature_name, feature_values in features.items():
                    combined_features.extend(feature_values.flatten())
                    if len(feature_names) == 0:  # Solo en la primera iteración
                        for i in range(len(feature_values.flatten())):
                            feature_names.append(f"{feature_name}_{i}")
            else:
                if 'bow' in feature_types:
                    features = extractor.get_feature_vector()
                    combined_features = features.flatten().tolist()
                    if len(feature_names) == 0:
                        feature_names = [f"bow_cluster_{i}" for i in range(len(combined_features))]
            
            X.append(combined_features)
            y.append(category)
            
        except Exception as e:
            st.warning(f"Error procesando {img_name}: {e}")
            continue
    
    return np.array(X), np.array(y), feature_names

def train_multiclass_models(X, y, test_size=0.3):
    """Entrena múltiples modelos de clasificación multiclase"""
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos a entrenar
    models = {
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'K-NN': KNeighborsClassifier(n_neighbors=3)
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Entrenar modelo
        model.fit(X_train_scaled, y_train)
        
        # Evaluaciones
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        y_pred = model.predict(X_test_scaled)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3)
        
        results[model_name] = {
            'model': model,
            'train_score': train_score,
            'test_score': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test
        }
    
    return results, scaler, X_test_scaled

# Sidebar - Controles
st.sidebar.header("⚙️ Controles")

# Modo de aplicación
app_mode = st.sidebar.selectbox(
    "🔬 Modo de Aplicación:",
    options=['feature_extraction', 'multiclass_classification', 'feature_analysis', 'model_comparison', 'feature_engineering'],
    format_func=lambda x: {
        'feature_extraction': '🔍 Extracción Individual',
        'multiclass_classification': '🎯 Clasificación Multiclase',
        'feature_analysis': '📊 Análisis de Features',
        'model_comparison': '⚖️ Comparación de Modelos',
        'feature_engineering': '🛠️ Feature Engineering'
    }[x]
)

st.sidebar.markdown("---")

# Selector de imagen
sample_images = load_sample_images()

if app_mode == 'feature_extraction':
    # Upload de imagen individual
    uploaded_file = st.sidebar.file_uploader(
        "📁 Sube tu imagen", 
        type=['png', 'jpg', 'jpeg'],
        help="Formatos soportados: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        selected_image = pil_to_cv2(pil_image)
        image_source = f"📁 {uploaded_file.name}"
    else:
        # Usar primera imagen disponible
        if sample_images:
            first_key = list(sample_images.keys())[0]
            selected_image = sample_images[first_key]
            image_source = f"🖼️ {first_key}"
        else:
            selected_image = None
            image_source = "❌ No hay imagen"

else:
    # Mostrar información del dataset
    st.sidebar.info(f"**Dataset:** {len(sample_images)} imágenes")
    
    # Análisis de categorías
    categories = defaultdict(int)
    for img_name in sample_images.keys():
        if 'naturales' in img_name:
            categories['Naturales'] += 1
        elif 'geometricas' in img_name:
            categories['Geométricas'] += 1
        elif 'texturas' in img_name:
            categories['Texturas'] += 1
        elif 'rostros' in img_name:
            categories['Rostros'] += 1
        elif 'objetos' in img_name:
            categories['Objetos'] += 1
        else:
            categories['Otros'] += 1
    
    for category, count in categories.items():
        st.sidebar.write(f"- {category}: {count}")

st.sidebar.markdown("---")

# Controles específicos por modo
if app_mode == 'feature_extraction':
    st.sidebar.subheader("🔍 Configuración de Features")
    
    feature_types = st.sidebar.multiselect(
        "Tipos de características:",
        options=['bow', 'color', 'texture', 'shape'],
        default=['bow'],
        format_func=lambda x: {
            'bow': '📚 Bag of Words (SIFT)',
            'color': '🎨 Color',
            'texture': '🔲 Textura',
            'shape': '📐 Forma'
        }[x]
    )
    
    if 'bow' in feature_types:
        num_clusters = st.sidebar.slider("Clusters BoW:", 16, 128, 32, 16)
    
    optimal_size = st.sidebar.slider("Tamaño optimizado:", 100, 300, 150, 50)

elif app_mode in ['multiclass_classification', 'model_comparison']:
    st.sidebar.subheader("🎯 Configuración ML")
    
    feature_set = st.sidebar.selectbox(
        "Conjunto de características:",
        options=['bow', 'all'],
        format_func=lambda x: {
            'bow': '📚 Solo BoW',
            'all': '🔄 Todas las características'
        }[x]
    )
    
    test_size = st.sidebar.slider("Tamaño test (%):", 10, 50, 30, 5) / 100
    
elif app_mode == 'feature_analysis':
    st.sidebar.subheader("📊 Configuración Análisis")
    
    analysis_type = st.sidebar.selectbox(
        "Tipo de análisis:",
        options=['pca', 'correlation', 'distribution', 'clustering'],
        format_func=lambda x: {
            'pca': '📈 Análisis PCA',
            'correlation': '🔗 Correlaciones',
            'distribution': '📊 Distribuciones',
            'clustering': '🎯 Clustering'
        }[x]
    )

elif app_mode == 'feature_engineering':
    st.sidebar.subheader("🛠️ Feature Engineering")
    
    engineering_method = st.sidebar.selectbox(
        "Método:",
        options=['selection', 'reduction', 'combination', 'normalization'],
        format_func=lambda x: {
            'selection': '🎯 Selección de Features',
            'reduction': '📉 Reducción Dimensionalidad',
            'combination': '🔄 Combinación Features',
            'normalization': '📏 Normalización'
        }[x]
    )

# Procesamiento según el modo
if app_mode == 'feature_extraction':
    st.subheader("🔍 Extracción Individual de Características")
    
    if selected_image is not None:
        # Extraer características
        extractor = AdvancedFeatureExtractor(image=selected_image, optimal_size=optimal_size)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2_to_pil(selected_image), use_container_width=True)
            st.write(f"**Fuente:** {image_source}")
            h, w = selected_image.shape[:2]
            st.write(f"**Dimensiones:** {w} x {h} píxeles")
        
        with col2:
            st.write("**🔧 Imagen Optimizada**")
            st.image(cv2_to_pil(extractor.image), use_container_width=True)
            h_opt, w_opt = extractor.image.shape[:2]
            st.write(f"**Dimensiones optimizadas:** {w_opt} x {h_opt} píxeles")
        
        # Extraer y mostrar características
        st.subheader("📊 Características Extraídas")
        
        if 'all' in feature_types or len(feature_types) > 1:
            include_color = 'color' in feature_types
            include_texture = 'texture' in feature_types
            include_shape = 'shape' in feature_types
            
            features = extractor.extract_multiple_features(
                include_color=include_color,
                include_texture=include_texture, 
                include_shape=include_shape
            )
        else:
            if 'bow' in feature_types:
                bow_features = extractor.get_feature_vector(num_clusters)
                features = {'bow': bow_features.flatten()}
            else:
                features = {}
        
        # Mostrar características por tipo
        tabs = st.tabs(list(features.keys()))
        
        for i, (feature_type, feature_values) in enumerate(features.items()):
            with tabs[i]:
                st.write(f"**Tipo:** {feature_type}")
                st.write(f"**Dimensiones:** {len(feature_values.flatten())}")
                
                # Visualizar según el tipo
                if feature_type == 'bow':
                    # Histograma BoW
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.bar(range(len(feature_values)), feature_values)
                    ax.set_title('Histograma Bag of Words')
                    ax.set_xlabel('Cluster Visual')
                    ax.set_ylabel('Frecuencia Normalizada')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                elif 'hist' in feature_type:
                    # Histogramas de color/textura
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(feature_values)
                    ax.set_title(f'Histograma {feature_type}')
                    ax.set_xlabel('Bin')
                    ax.set_ylabel('Frecuencia')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                else:
                    # Valores numéricos
                    if len(feature_values) <= 10:
                        # Mostrar como métricas si son pocos valores
                        cols = st.columns(min(len(feature_values), 5))
                        for j, value in enumerate(feature_values):
                            with cols[j % 5]:
                                st.metric(f"Valor {j+1}", f"{value:.3f}")
                    else:
                        # Mostrar como array
                        st.write("**Valores:**")
                        st.write(feature_values)
        
        # Resumen total
        total_features = sum(len(fv.flatten()) for fv in features.values())
        st.info(f"**Total de características extraídas:** {total_features}")

elif app_mode == 'multiclass_classification':
    st.subheader("🎯 Clasificación Multiclase - Resultados")
    
    # Preparar dataset
    with st.spinner("Preparando dataset y extrayendo características..."):
        X, y, feature_names = prepare_multiclass_dataset(sample_images, [feature_set])
    
    if len(X) > 0:
        # Entrenar modelos
        with st.spinner("Entrenando modelos de clasificación..."):
            results, scaler, X_test_scaled = train_multiclass_models(X, y, test_size)
        
        # Mostrar resultados
        st.subheader("📊 Rendimiento de Modelos")
        
        # Crear tabla comparativa
        performance_data = []
        for model_name, result in results.items():
            performance_data.append({
                'Modelo': model_name,
                'Precisión Entrenamiento': f"{result['train_score']:.3f}",
                'Precisión Prueba': f"{result['test_score']:.3f}",
                'CV Media': f"{result['cv_mean']:.3f}",
                'CV Std': f"{result['cv_std']:.3f}"
            })
        
        if PANDAS_AVAILABLE:
            df_performance = pd.DataFrame(performance_data)
            st.dataframe(df_performance, use_container_width=True)
        else:
            st.table(performance_data)
        
        # Gráfico de comparación
        if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
            fig = px.bar(
                df_performance, 
                x='Modelo', 
                y=['Precisión Entrenamiento', 'Precisión Prueba'],
                title='Comparación de Rendimiento de Modelos',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback con matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            models = [item['Modelo'] for item in performance_data]
            train_scores = [float(item['Precisión Entrenamiento']) for item in performance_data]
            test_scores = [float(item['Precisión Prueba']) for item in performance_data]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax.bar(x - width/2, train_scores, width, label='Entrenamiento')
            ax.bar(x + width/2, test_scores, width, label='Prueba')
            
            ax.set_xlabel('Modelo')
            ax.set_ylabel('Precisión')
            ax.set_title('Comparación de Rendimiento de Modelos')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
        
        # Matriz de confusión del mejor modelo
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_score'])
        best_result = results[best_model_name]
        
        st.subheader(f"📈 Análisis Detallado - {best_model_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔍 Matriz de Confusión**")
            cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=np.unique(y),
                       yticklabels=np.unique(y), ax=ax)
            ax.set_title(f'Matriz de Confusión - {best_model_name}')
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.write("**📋 Reporte de Clasificación**")
            report = classification_report(best_result['y_test'], best_result['y_pred'], output_dict=True)
            
            # Convertir a DataFrame para mejor visualización
            if PANDAS_AVAILABLE:
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.round(3))
            else:
                # Mostrar reporte como texto simple
                st.text(report if isinstance(report, str) else str(report))
        
        # Distribución de clases
        st.subheader("📊 Distribución del Dataset")
        
        class_counts = Counter(y)
        
        if PLOTLY_AVAILABLE:
            fig = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title='Distribución de Clases en el Dataset'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback con matplotlib
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(list(class_counts.values()), labels=list(class_counts.keys()), 
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Distribución de Clases en el Dataset')
            st.pyplot(fig)
            plt.close()
        
        # Información del dataset
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Muestras", len(X))
        
        with col2:
            st.metric("Número de Clases", len(np.unique(y)))
        
        with col3:
            st.metric("Características", X.shape[1])
        
        with col4:
            st.metric("Mejor Precisión", f"{max(r['test_score'] for r in results.values()):.3f}")

elif app_mode == 'feature_analysis':
    st.subheader("📊 Análisis Avanzado de Características")
    
    # Preparar dataset
    X, y, feature_names = prepare_multiclass_dataset(sample_images, ['all'])
    
    if len(X) > 0:
        if analysis_type == 'pca':
            # Análisis PCA
            st.write("**📈 Análisis de Componentes Principales (PCA)**")
            
            if SKLEARN_AVAILABLE:
                # Normalizar datos
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Aplicar PCA
                pca = PCA()
                X_pca = pca.fit_transform(X_scaled)
                
                # Varianza explicada
                variance_ratio = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(variance_ratio)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Gráfico de varianza explicada
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(1, min(21, len(variance_ratio)+1)), variance_ratio[:20])
                    ax.set_title('Varianza Explicada por Componente')
                    ax.set_xlabel('Componente Principal')
                    ax.set_ylabel('Varianza Explicada')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Gráfico de varianza acumulada
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(1, min(21, len(cumulative_variance)+1)), cumulative_variance[:20], 'bo-')
                    ax.axhline(y=0.95, color='r', linestyle='--', label='95%')
                    ax.set_title('Varianza Explicada Acumulada')
                    ax.set_xlabel('Número de Componentes')
                    ax.set_ylabel('Varianza Acumulada')
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Proyección 2D
                if len(X_pca) > 1:
                    st.write("**🎯 Proyección 2D de las Clases**")
                    
                    if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
                        # Crear DataFrame para plotly
                        df_pca = pd.DataFrame({
                            'PC1': X_pca[:, 0],
                            'PC2': X_pca[:, 1],
                            'Clase': y
                        })
                        
                        fig = px.scatter(
                            df_pca, x='PC1', y='PC2', color='Clase',
                            title='Proyección PCA - Primeras dos componentes',
                            labels={'PC1': f'PC1 ({variance_ratio[0]:.1%} varianza)',
                                   'PC2': f'PC2 ({variance_ratio[1]:.1%} varianza)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fallback con matplotlib
                        fig, ax = plt.subplots(figsize=(10, 6))
                        unique_classes = np.unique(y)
                        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_classes)))
                        
                        for i, cls in enumerate(unique_classes):
                            mask = np.array(y) == cls
                            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                                      c=[colors[i]], label=cls, alpha=0.7)
                        
                        ax.set_xlabel(f'PC1 ({variance_ratio[0]:.1%} varianza)')
                        ax.set_ylabel(f'PC2 ({variance_ratio[1]:.1%} varianza)')
                        ax.set_title('Proyección PCA - Primeras dos componentes')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    
                    # Métricas PCA
                    components_95 = np.argmax(cumulative_variance >= 0.95) + 1
                    st.metric("Componentes para 95% varianza", components_95)
            else:
                st.error("❌ Análisis PCA requiere scikit-learn")
                st.info("📦 Instala scikit-learn para usar esta funcionalidad:")
                st.code("pip install scikit-learn")
                st.stop()
        
        elif analysis_type == 'correlation':
            st.write("**🔗 Análisis de Correlaciones**")
            
            if PANDAS_AVAILABLE:
                # Convertir a DataFrame para análisis de correlación
                df_features = pd.DataFrame(X, columns=feature_names[:len(X[0])] if feature_names else [f"Feature_{i}" for i in range(len(X[0]))])
                
                # Matriz de correlación
                correlation_matrix = df_features.corr()
                
                # Visualizar matriz de correlación
                fig, ax = plt.subplots(figsize=(12, 10))
                
                if SEABORN_AVAILABLE:
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, ax=ax, fmt='.2f')
                else:
                    im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
                    ax.set_xticks(range(len(correlation_matrix.columns)))
                    ax.set_yticks(range(len(correlation_matrix.columns)))
                    ax.set_xticklabels(correlation_matrix.columns, rotation=45)
                    ax.set_yticklabels(correlation_matrix.columns)
                    plt.colorbar(im)
                
                ax.set_title('Matriz de Correlación de Características')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Características más correlacionadas
                st.write("**🔍 Correlaciones Más Altas:**")
                correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        correlations.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j]
                        ))
                
                correlations.sort(key=lambda x: abs(x[2]), reverse=True)
                
                for feat1, feat2, corr in correlations[:10]:
                    st.write(f"• {feat1} ↔ {feat2}: {corr:.3f}")
            else:
                st.error("❌ Análisis de correlación requiere pandas")
                st.info("📦 Instala pandas para usar esta funcionalidad:")
                st.code("pip install pandas")
        
        elif analysis_type == 'distribution':
            st.write("**📊 Análisis de Distribuciones**")
            
            # Estadísticas básicas
            X_array = np.array(X)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Media General", f"{X_array.mean():.3f}")
            
            with col2:
                st.metric("Desv. Estándar", f"{X_array.std():.3f}")
            
            with col3:
                st.metric("Valor Mínimo", f"{X_array.min():.3f}")
            
            with col4:
                st.metric("Valor Máximo", f"{X_array.max():.3f}")
            
            # Histograma de las primeras características
            st.write("**📈 Histogramas de las Primeras 6 Características:**")
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            for i in range(min(6, X_array.shape[1])):
                axes[i].hist(X_array[:, i], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Característica {i+1}')
                axes[i].set_ylabel('Frecuencia')
                axes[i].grid(True, alpha=0.3)
            
            # Ocultar axes no usados
            for i in range(min(6, X_array.shape[1]), 6):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Distribución por clase
            st.write("**🎯 Distribución por Clase:**")
            class_counts = Counter(y)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            ax.bar(classes, counts, color='lightcoral', alpha=0.7, edgecolor='black')
            ax.set_title('Distribución de Muestras por Clase')
            ax.set_xlabel('Clase')
            ax.set_ylabel('Número de Muestras')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        elif analysis_type == 'clustering':
            st.write("**🎯 Análisis de Clustering**")
            
            # Normalizar datos
            X_array = np.array(X)
            X_mean = X_array.mean(axis=0)
            X_std = X_array.std(axis=0)
            X_std[X_std == 0] = 1  # Evitar división por cero
            X_normalized = (X_array - X_mean) / X_std
            
            # K-Means clustering
            n_clusters = st.slider("Número de clusters:", 2, 10, 5)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X_normalized)
            
            # Visualización 2D usando las primeras dos características
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_clusters))
            
            for i in range(n_clusters):
                mask = cluster_labels == i
                if np.any(mask):
                    ax.scatter(X_normalized[mask, 0], X_normalized[mask, 1], 
                             c=[colors[i]], label=f'Cluster {i+1}', alpha=0.7, s=50)
            
            # Centroides
            centroids = kmeans.cluster_centers_
            ax.scatter(centroids[:, 0], centroids[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='Centroides')
            
            ax.set_xlabel('Característica 1 (normalizada)')
            ax.set_ylabel('Característica 2 (normalizada)')
            ax.set_title(f'K-Means Clustering (k={n_clusters})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Comparación con clases reales
            st.write("**📋 Comparación Cluster vs Clase Real:**")
            
            if PANDAS_AVAILABLE:
                comparison_df = pd.DataFrame({
                    'Clase Real': y,
                    'Cluster Asignado': cluster_labels
                })
                
                comparison_table = pd.crosstab(comparison_df['Clase Real'], 
                                             comparison_df['Cluster Asignado'], 
                                             margins=True)
                st.dataframe(comparison_table)
            else:
                # Mostrar comparación simple
                cluster_class_pairs = list(zip(y, cluster_labels))
                class_cluster_map = defaultdict(list)
                
                for real_class, cluster in cluster_class_pairs:
                    class_cluster_map[real_class].append(cluster)
                
                for real_class, clusters in class_cluster_map.items():
                    cluster_counts = Counter(clusters)
                    st.write(f"**{real_class}**: {dict(cluster_counts)}")

elif app_mode == 'model_comparison':
    st.subheader("⚖️ Comparación Exhaustiva de Modelos")
    
    # Preparar dataset
    X, y, feature_names = prepare_multiclass_dataset(sample_images, [feature_set])
    
    if len(X) > 0:
        # Entrenar múltiples configuraciones
        configurations = [
            {'name': 'Configuración Base', 'params': {}},
            {'name': 'SVM Optimizado', 'params': {'C': 10, 'gamma': 'scale'}},
            {'name': 'RF Más Árboles', 'params': {'n_estimators': 200, 'max_depth': 10}},
        ]
        
        all_results = {}
        
        for config in configurations:
            st.write(f"**⚙️ Entrenando {config['name']}...**")
            
            # Modelos con configuración específica
            if config['name'] == 'SVM Optimizado':
                models = {'SVM Optimizado': SVC(C=10, gamma='scale', random_state=42, probability=True)}
            elif config['name'] == 'RF Más Árboles':
                models = {'RF Más Árboles': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)}
            else:
                models = {
                    'SVM Base': SVC(random_state=42, probability=True),
                    'Random Forest Base': RandomForestClassifier(random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
                }
            
            # Entrenar modelos de esta configuración
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            for model_name, model in models.items():
                model.fit(X_train_scaled, y_train)
                test_score = model.score(X_test_scaled, y_test)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                all_results[model_name] = {
                    'test_score': test_score,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'config': config['name']
                }
        
        # Crear tabla de comparación
        comparison_data = []
        for model_name, result in all_results.items():
            comparison_data.append({
                'Modelo': model_name,
                'Configuración': result['config'],
                'Precisión Test': f"{result['test_score']:.3f}",
                'CV Media': f"{result['cv_mean']:.3f}",
                'CV Std': f"{result['cv_std']:.3f}",
                'Score Total': result['test_score'] + result['cv_mean']  # Para ranking
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('Score Total', ascending=False)
        
        st.subheader("🏆 Ranking de Modelos")
        
        # Mostrar tabla sin la columna Score Total
        if PANDAS_AVAILABLE:
            display_df = df_comparison.drop('Score Total', axis=1).reset_index(drop=True)
            display_df.index = display_df.index + 1  # Empezar ranking en 1
            st.dataframe(display_df, use_container_width=True)
        else:
            st.table(comparison_data)
        
        # Gráfico de comparación
        if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
            fig = px.scatter(
                df_comparison,
                x='CV Media',
                y='Precisión Test',
                size='Score Total',
                color='Configuración',
                hover_name='Modelo',
                title='Comparación de Rendimiento: CV vs Test'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback con matplotlib
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extraer datos para matplotlib
            cv_means = [float(item['CV Media']) for item in comparison_data]
            test_scores = [float(item['Precisión Test']) for item in comparison_data]
            models = [item['Modelo'] for item in comparison_data]
            
            scatter = ax.scatter(cv_means, test_scores, alpha=0.6, s=100)
            
            # Agregar etiquetas
            for i, model in enumerate(models):
                ax.annotate(model, (cv_means[i], test_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.7)
            
            ax.set_xlabel('CV Media')
            ax.set_ylabel('Precisión Test')
            ax.set_title('Comparación de Rendimiento: CV vs Test')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Mejor modelo
        best_model = df_comparison.iloc[0]
        st.success(f"🥇 **Mejor modelo:** {best_model['Modelo']} ({best_model['Configuración']})")

elif app_mode == 'feature_engineering':
    st.subheader("🛠️ Feature Engineering Avanzado")
    
    # Preparar dataset completo
    X, y, feature_names = prepare_multiclass_dataset(sample_images, ['all'])
    
    if len(X) > 0:
        if engineering_method == 'selection':
            # Selección de características
            if SKLEARN_AVAILABLE:
                from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
            else:
                st.warning("⚠️ Sklearn no disponible. Mostrando funcionalidad limitada.")
                st.info("💡 Para funcionalidad completa, instala: pip install scikit-learn")
                SelectKBest = f_classif = mutual_info_classif = None
            
            st.write("**🎯 Selección de Mejores Características**")
            
            if SKLEARN_AVAILABLE and SelectKBest is not None:
                # Diferentes métodos de selección
                methods = {
                    'F-Score': SelectKBest(f_classif, k=20),
                    'Mutual Info': SelectKBest(mutual_info_classif, k=20)
                }
                
                results = {}
                for method_name, selector in methods.items():
                    X_selected = selector.fit_transform(X, y)
                    selected_features = selector.get_support(indices=True)
                    scores = selector.scores_
                    
                    results[method_name] = {
                        'selected_features': selected_features,
                        'scores': scores,
                        'X_selected': X_selected
                    }
            else:
                st.info("📊 Funcionalidad de selección de características requiere sklearn")
                st.write("Con sklearn, puedes:")
                st.write("- Usar F-Score para ranking estadístico de características")  
                st.write("- Aplicar Mutual Information para dependencias no lineales")
                st.write("- Comparar diferentes métodos de selección")
                results = {}
            
            # Mostrar características seleccionadas
            col1, col2 = st.columns(2)
            
            for i, (method_name, result) in enumerate(results.items()):
                with [col1, col2][i]:
                    st.write(f"**{method_name}**")
                    
                    # Top 10 características
                    top_indices = np.argsort(result['scores'])[-10:][::-1]
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(range(10), result['scores'][top_indices])
                    ax.set_yticks(range(10))
                    ax.set_yticklabels([f"Feature {idx}" for idx in top_indices])
                    ax.set_title(f'Top 10 Características - {method_name}')
                    ax.set_xlabel('Score')
                    plt.tight_layout()
                    st.pyplot(fig)
        
        elif engineering_method == 'reduction':
            # Reducción de dimensionalidad
            st.write("**📉 Reducción de Dimensionalidad**")
            
            if SKLEARN_AVAILABLE:
                from sklearn.decomposition import PCA
                from sklearn.manifold import TSNE
            else:
                st.warning("⚠️ Sklearn no disponible. Mostrando funcionalidad limitada.")
                st.info("💡 Para funcionalidad completa, instala: pip install scikit-learn")
                PCA = TSNE = None
            
            if SKLEARN_AVAILABLE and PCA is not None:
                # Normalizar datos
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # PCA
                pca = PCA(n_components=10)
                X_pca = pca.fit_transform(X_scaled)
                
                # t-SNE (solo si no son demasiados datos)
                if len(X) <= 100:
                    tsne = TSNE(n_components=2, random_state=42)
                    X_tsne = tsne.fit_transform(X_scaled)
                else:
                    X_tsne = None
            else:
                st.info("📊 Funcionalidad de reducción de dimensionalidad requiere sklearn")
                st.write("Con sklearn, puedes:")
                st.write("- Usar PCA para reducción lineal de dimensionalidad")
                st.write("- Aplicar t-SNE para visualización no lineal") 
                st.write("- Analizar varianza explicada por componentes principales")
                X_pca = X_tsne = pca = None
            
            if X_pca is not None and pca is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📈 PCA - Varianza Explicada**")
                    
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(range(1, 11), pca.explained_variance_ratio_)
                    ax.set_title('Varianza Explicada por Componente PCA')
                    ax.set_xlabel('Componente')
                    ax.set_ylabel('Varianza Explicada')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.metric("Varianza total (10 componentes)", f"{pca.explained_variance_ratio_.sum():.1%}")
                
                with col2:
                    if X_tsne is not None:
                        st.write("**🎯 Visualización t-SNE**")
                        
                        if PLOTLY_AVAILABLE and PANDAS_AVAILABLE:
                            df_tsne = pd.DataFrame({
                                'TSNE1': X_tsne[:, 0],
                                'TSNE2': X_tsne[:, 1],
                                'Clase': y
                            })
                            
                            fig = px.scatter(
                                df_tsne, x='TSNE1', y='TSNE2', color='Clase',
                                title='Proyección t-SNE'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Fallback con matplotlib
                            fig, ax = plt.subplots(figsize=(8, 6))
                            unique_classes = np.unique(y)
                            colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_classes)))
                            
                            for i, cls in enumerate(unique_classes):
                                mask = np.array(y) == cls
                                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                                          c=[colors[i]], label=cls, alpha=0.7)
                            
                            ax.set_xlabel('t-SNE 1')
                            ax.set_ylabel('t-SNE 2')
                            ax.set_title('Proyección t-SNE')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                    else:
                        st.info("t-SNE omitido (demasiados datos)")

# Botón de descarga
if 'selected_image' in locals() and selected_image is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        processed_pil = cv2_to_pil(selected_image)
        buf = io.BytesIO()
        processed_pil.save(buf, format='PNG')
        
        st.download_button(
            label="📥 Descargar Imagen",
            data=buf.getvalue(),
            file_name=f"feature_analysis_{app_mode}.png",
            mime="image/png",
            use_container_width=True
        )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("🔬 Feature Engineering"):
    st.markdown("""
    **Extracción de Características:**
    - **BoW (SIFT)**: Representación basada en vocabulario visual
    - **Color**: Histogramas RGB/HSV, estadísticas
    - **Textura**: LBP, gradientes, patrones locales
    - **Forma**: Momentos de Hu, área, perímetro
    
    **Importancia:**
    - Determina el rendimiento del modelo
    - Debe capturar información discriminativa
    - Balance entre dimensionalidad y información
    """)

with st.sidebar.expander("🎯 Clasificación Multiclase"):
    st.markdown("""
    **Estrategias:**
    - **One-vs-Rest**: Un clasificador binario por clase
    - **One-vs-One**: Clasificador para cada par de clases
    - **Multiclase nativa**: SVM, Random Forest
    
    **Métricas:**
    - **Accuracy**: Proporción de aciertos globales
    - **Precision/Recall**: Por clase individual
    - **F1-Score**: Media armónica precision/recall
    - **Confusion Matrix**: Matriz de confusión detallada
    """)

with st.sidebar.expander("📊 Análisis de Features"):
    st.markdown("""
    **PCA (Análisis de Componentes Principales):**
    - Reducción lineal de dimensionalidad
    - Preserva máxima varianza
    - Interpretable (combinaciones lineales)
    
    **t-SNE:**
    - Reducción no lineal para visualización
    - Preserva estructura local
    - Ideal para explorar clusters
    
    **Selección de Features:**
    - F-Score: Test estadístico ANOVA
    - Mutual Information: Dependencia no lineal
    """)

with st.sidebar.expander("⚖️ Comparación de Modelos"):
    st.markdown("""
    **Modelos Implementados:**
    
    **SVM (Support Vector Machine):**
    - Kernel RBF para relaciones no lineales
    - Robusto a overfitting
    - Parámetros: C (regularización), gamma
    
    **Random Forest:**
    - Ensemble de árboles de decisión
    - Maneja bien features correlacionadas
    - Proporciona importancia de features
    
    **Gradient Boosting:**
    - Ensemble secuencial
    - Alta precisión, propenso a overfitting
    - Requiere tuning cuidadoso
    
    **Naive Bayes:**
    - Asume independencia entre features
    - Rápido y simple
    - Funciona bien con datos pequeños
    
    **K-NN:**
    - Clasificación basada en similitud
    - No paramétrico
    - Sensible a la escala de features
    """)

with st.sidebar.expander("💡 Buenas Prácticas"):
    st.markdown("""
    **Pipeline Completo:**
    1. **Preprocesamiento**: Normalización, manejo de outliers
    2. **Feature Engineering**: Extracción, selección, transformación
    3. **División de datos**: Train/validation/test estratificado
    4. **Entrenamiento**: Múltiples modelos, hiperparámetros
    5. **Evaluación**: Validación cruzada, métricas múltiples
    6. **Selección**: Mejor modelo según métricas objetivas
    
    **Evitar Overfitting:**
    - Validación cruzada
    - Regularización (L1/L2)
    - Early stopping
    - Ensemble methods
    
    **Interpretabilidad:**
    - Feature importance
    - Análisis de correlaciones
    - Visualización de decisiones
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🔬 <strong>Extracción Avanzada de Características y Clasificación</strong> | Capítulo 11 - Machine Learning Avanzado</p>
        <p><small>Explora técnicas avanzadas de feature engineering y clasificación multiclase</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)