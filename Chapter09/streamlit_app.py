"""
Capítulo 9 - Bag of Words
Demostración del código create_features.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 9 - Bag of Words", layout="wide")

# Título
st.title("🧠 Capítulo 9: Bag of Words")

def cv2_to_pil(cv2_img):
    """Convierte imagen de OpenCV (BGR) a PIL (RGB)"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    """Convierte imagen de PIL (RGB) a OpenCV (BGR)"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

class SIFTFeatureExtractor:
    """Extractor de características SIFT simplificado"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create()
    
    def extract_features(self, img):
        """Extraer características SIFT de la imagen"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Dibujar keypoints
        img_with_keypoints = cv2.drawKeypoints(
            img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'img_with_keypoints': img_with_keypoints,
            'num_features': len(keypoints) if keypoints else 0
        }

def create_bovw_features(descriptors_list, k=50):
    """Crear Bag of Visual Words usando K-means"""
    if not descriptors_list or len(descriptors_list) == 0:
        return None, None
    
    try:
        from sklearn.cluster import KMeans
        
        # Concatenar todos los descriptores
        all_descriptors = np.vstack(descriptors_list)
        
        # Ajustar k si es mayor que el número de muestras disponibles
        n_samples = all_descriptors.shape[0]
        original_k = k
        if k > n_samples:
            k = max(1, min(n_samples, 10))  # Usar máximo 10 clusters o el número de muestras
            st.warning(f"⚠️ Ajustando número de clusters de {original_k} original a {k} debido a pocas muestras ({n_samples})")
        
        # Aplicar K-means para crear vocabulario visual
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(all_descriptors)
        
        # Crear histograma para cada imagen
        bovw_features = []
        for descriptors in descriptors_list:
            if descriptors is not None:
                # Predecir clusters para cada descriptor
                labels = kmeans.predict(descriptors)
                # Crear histograma
                hist, _ = np.histogram(labels, bins=k, range=(0, k))
                # Normalizar
                hist = hist.astype(float)
                if np.sum(hist) > 0:
                    hist = hist / np.sum(hist)
                bovw_features.append(hist)
            else:
                bovw_features.append(np.zeros(k))
        
        return kmeans, bovw_features
        
    except ImportError:
        st.error("❌ scikit-learn no está instalado. Usando método alternativo.")
        return None, None

def load_example_images():
    """Carga múltiples imágenes de ejemplo con más características"""
    images = []
    
    # Crear diferentes imágenes de ejemplo con más detalles para generar más características SIFT
    for i in range(5):  # Aumentamos a 5 imágenes
        img = np.ones((400, 500, 3), dtype=np.uint8) * (30 + i * 20)
        
        if i == 0:  # Imagen con múltiples círculos
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            positions = [(80, 80), (300, 120), (150, 250), (400, 300)]
            for pos, color in zip(positions, colors):
                cv2.circle(img, pos, 25, color, -1)
                cv2.circle(img, pos, 35, (255, 255, 255), 2)
                
        elif i == 1:  # Imagen con rectángulos y líneas
            cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 255), -1)
            cv2.rectangle(img, (200, 180), (350, 280), (255, 255, 0), -1)
            cv2.line(img, (0, 200), (500, 200), (255, 0, 255), 3)
            cv2.line(img, (250, 0), (250, 400), (0, 255, 255), 3)
            
        elif i == 2:  # Imagen con formas mixtas
            cv2.circle(img, (150, 150), 60, (255, 0, 255), -1)
            cv2.rectangle(img, (50, 250), (200, 350), (0, 255, 255), -1)
            cv2.ellipse(img, (350, 200), (80, 40), 45, 0, 360, (255, 100, 100), -1)
            
        elif i == 3:  # Imagen con patrón de puntos
            for x in range(50, 450, 60):
                for y in range(50, 350, 60):
                    color = (np.random.randint(100, 255), np.random.randint(100, 255), np.random.randint(100, 255))
                    cv2.circle(img, (x, y), 15, color, -1)
                    
        else:  # Imagen con texto y formas
            cv2.putText(img, "SIFT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
            cv2.putText(img, "TEST", (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.rectangle(img, (300, 50), (480, 150), (255, 0, 0), 3)
        
        images.append(img)
    
    return images

# Sidebar para configuración
st.sidebar.header("🛠️ Configuración")

use_multiple_images = st.sidebar.checkbox("Usar múltiples imágenes", value=True)

if not use_multiple_images:
    image_source = st.sidebar.radio(
        "Selecciona imagen:",
        ["🖼️ Imagen de ejemplo", "📤 Cargar imagen"]
    )

k_clusters = st.sidebar.slider("Número de clusters (K)", 5, 50, 20)

# Cargar imágenes
images = []
image_names = []

if use_multiple_images:
    # Usar múltiples imágenes de ejemplo
    images = load_example_images()
    image_names = ["círculos.jpg", "geometría.jpg", "formas.jpg", "puntos.jpg", "texto.jpg"]
    st.sidebar.success(f"✅ Usando {len(images)} imágenes de ejemplo")
else:
    # Usar una sola imagen
    if image_source == "📤 Cargar imagen":
        uploaded_file = st.sidebar.file_uploader(
            "Sube tu imagen:",
            type=['png', 'jpg', 'jpeg', 'bmp']
        )
        
        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file)
                img = pil_to_cv2(pil_image)
                images = [img]
                image_names = [uploaded_file.name]
                st.sidebar.success(f"✅ Imagen cargada: {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {str(e)}")
        else:
            st.sidebar.info("👆 Sube una imagen")
    else:
        images = load_example_images()[:1]  # Solo la primera
        image_names = ["ejemplo.jpg"]
        st.sidebar.success("✅ Usando imagen de ejemplo")

# Mostrar información
if images:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Info del Dataset:**")
    st.sidebar.write(f"• **Número de imágenes:** {len(images)}")
    st.sidebar.write(f"• **Clusters K-means:** {k_clusters}")

if images:
    # Procesar imágenes
    try:
        with st.spinner("🔄 Extrayendo características SIFT..."):
            extractor = SIFTFeatureExtractor()
            results = []
            descriptors_list = []
            
            for img in images:
                result = extractor.extract_features(img)
                results.append(result)
                if result['descriptors'] is not None:
                    descriptors_list.append(result['descriptors'])
        
        # Verificar que tenemos suficientes características
        if not descriptors_list:
            st.error("❌ No se encontraron características SIFT en las imágenes.")
            st.stop()
        
        total_descriptors = sum(len(desc) for desc in descriptors_list)
        st.info(f"ℹ️ Total de descriptores SIFT encontrados: {total_descriptors}")
        
        with st.spinner("🔄 Creando Bag of Visual Words..."):
            kmeans_model, bovw_features = create_bovw_features(descriptors_list, k_clusters)
        
        st.success("✅ **Bag of Visual Words creado**")
        
        # Métricas
        cols = st.columns(4)
        with cols[0]:
            st.metric("Imágenes", len(images))
        with cols[1]:
            total_features = sum(r['num_features'] for r in results)
            st.metric("Características SIFT", total_features)
        with cols[2]:
            st.metric("Vocabulario Visual", f"{k_clusters} palabras")
        with cols[3]:
            bovw_status = "✅" if bovw_features else "❌"
            st.metric("BoVW", bovw_status)
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    # Mostrar código
    st.subheader("📄 Código Principal:")
    st.code("""
# Bag of Visual Words - Código principal
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Crear vocabulario visual con K-means
from sklearn.cluster import KMeans
all_descriptors = np.vstack(descriptors_list)
kmeans = KMeans(n_clusters=k)
kmeans.fit(all_descriptors)

# Crear histograma BoVW para cada imagen
labels = kmeans.predict(descriptors)
hist, _ = np.histogram(labels, bins=k)
""", language="python")
    
    # Resultados
    st.subheader("🖼️ Características Extraídas:")
    
    # Mostrar imágenes con keypoints
    cols = st.columns(min(len(images), 3))
    for i, (img, result, name) in enumerate(zip(images, results, image_names)):
        with cols[i % 3]:
            st.markdown(f"**{name}**")
            st.image(cv2_to_pil(result['img_with_keypoints']), use_container_width=True)
            st.write(f"Características: {result['num_features']}")
    
    # Mostrar histogramas BoVW si están disponibles
    if bovw_features:
        st.subheader("📊 Histogramas Bag of Visual Words:")
        
        for i, (hist, name) in enumerate(zip(bovw_features, image_names)):
            st.markdown(f"**Histograma BoVW - {name}:**")
            st.bar_chart(hist)
    
    # Explicación
    st.subheader("📚 Explicación:")
    st.markdown("""
    **Bag of Visual Words (BoVW)** es una técnica de representación de imágenes:
    
    1. **Extracción SIFT**: Detecta puntos clave y calcula descriptores locales
    2. **Vocabulario Visual**: Usa K-means para agrupar descriptores similares
    3. **Cuantización**: Asigna cada descriptor al cluster más cercano
    4. **Histograma**: Cuenta frecuencia de cada "palabra visual" por imagen
    5. **Clasificación**: Los histogramas se usan como features para ML
    """)

else:
    st.error("❌ No se pudieron cargar imágenes")