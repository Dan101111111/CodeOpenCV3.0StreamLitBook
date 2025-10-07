"""
Capítulo 7 - Segmentación Watershed
Demostración del código watershed.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 7 - Watershed", layout="wide")

# Título
st.title("💧 Capítulo 7: Segmentación Watershed")

def cv2_to_pil(cv2_img):
    """Convierte imagen de OpenCV (BGR) a PIL (RGB)"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    """Convierte imagen de PIL (RGB) a OpenCV (BGR)"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

def apply_watershed_segmentation(img):
    """Aplicar segmentación watershed"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold binario inverso
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Eliminación de ruido con morfología
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
    
    # Área segura de fondo
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Encontrar área segura de primer plano
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Región desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marcadores para watershed
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Aplicar watershed
    markers = cv2.watershed(img, markers)
    
    # Crear imagen de resultado
    result = img.copy()
    result[markers == -1] = [0, 0, 255]  # Bordes en rojo
    
    return {
        'original': img,
        'gray': gray,
        'thresh': thresh,
        'opening': opening,
        'sure_bg': sure_bg,
        'sure_fg': sure_fg,
        'unknown': unknown,
        'markers': markers,
        'result': result
    }

def load_example_image():
    """Carga imagen de ejemplo"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'images', 'shapes.png')
    
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    else:
        # Crear imagen de ejemplo con formas
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
        cv2.circle(img, (300, 100), 40, (255, 255, 255), -1)
        cv2.circle(img, (100, 200), 40, (255, 255, 255), -1)
        cv2.circle(img, (300, 200), 40, (255, 255, 255), -1)
        cv2.rectangle(img, (150, 150), (250, 250), (255, 255, 255), -1)
        return img

# Sidebar para configuración
st.sidebar.header("🛠️ Configuración")

image_source = st.sidebar.radio(
    "Selecciona imagen:",
    ["🖼️ Imagen de ejemplo", "📤 Cargar imagen"]
)

# Cargar imagen
img = None
img_name = ""

if image_source == "📤 Cargar imagen":
    uploaded_file = st.sidebar.file_uploader(
        "Sube tu imagen:",
        type=['png', 'jpg', 'jpeg', 'bmp']
    )
    
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            img = pil_to_cv2(pil_image)
            img_name = uploaded_file.name
            st.sidebar.success(f"✅ Imagen cargada: {img_name}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    else:
        st.sidebar.info("👆 Sube una imagen")
else:
    img = load_example_image()
    img_name = "shapes.png"
    st.sidebar.success(f"✅ Usando: {img_name}")

# Mostrar información de la imagen
if img is not None:
    height, width = img.shape[:2]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Info de la Imagen:**")
    st.sidebar.write(f"• **Dimensiones:** {width} x {height}")

if img is not None:
    # Procesar imagen
    try:
        with st.spinner("🔄 Procesando watershed..."):
            results = apply_watershed_segmentation(img)
        
        st.success("✅ **Segmentación watershed completada**")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Imagen Original", f"{img.shape[1]}x{img.shape[0]}")
        with col2:
            unique_markers = len(np.unique(results['markers']))
            st.metric("Regiones Detectadas", unique_markers)
        with col3:
            st.metric("Bordes Detectados", "✅")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    # Mostrar código
    st.subheader("📄 Código Principal:")
    st.code("""
# Watershed - Código principal
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
sure_bg = cv2.dilate(opening, kernel, iterations=3)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
ret, markers = cv2.connectedComponents(sure_fg.astype(np.uint8))
markers = cv2.watershed(img, markers)
""", language="python")
    
    # Resultados
    st.subheader("🖼️ Proceso Paso a Paso:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Imagen Original**")
        st.image(cv2_to_pil(results['original']), use_container_width=True)
        
        st.markdown("**2. Escala de Grises**")
        st.image(results['gray'], use_container_width=True, clamp=True)
    
    with col2:
        st.markdown("**3. Threshold Binario**")
        st.image(results['thresh'], use_container_width=True, clamp=True)
        
        st.markdown("**4. Morfología (Opening)**")
        st.image(results['opening'], use_container_width=True, clamp=True)
    
    with col3:
        st.markdown("**5. Fondo Seguro**")
        st.image(results['sure_bg'], use_container_width=True, clamp=True)
        
        st.markdown("**6. Resultado Final**")
        st.image(cv2_to_pil(results['result']), use_container_width=True)
    
    # Explicación
    st.subheader("📚 Explicación:")
    st.markdown("""
    **Watershed** es un algoritmo de segmentación que trata la imagen como topografía:
    
    1. **Threshold**: Convierte a binario para separar objetos del fondo
    2. **Morfología**: Elimina ruido usando operaciones opening/closing
    3. **Distance Transform**: Calcula distancia a los bordes
    4. **Marcadores**: Define semillas para las regiones
    5. **Watershed**: "Inunda" desde los marcadores hasta encontrar bordes
    """)

else:
    st.error("❌ No se pudo cargar la imagen")