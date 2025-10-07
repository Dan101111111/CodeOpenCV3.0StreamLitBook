"""
Capítulo 8 - Tracking por Color
Demostración del código colorspace_tracking.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 8 - Color Tracking", layout="wide")

# Título
st.title("🎯 Capítulo 8: Tracking por Color")

def cv2_to_pil(cv2_img):
    """Convierte imagen de OpenCV (BGR) a PIL (RGB)"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    """Convierte imagen de PIL (RGB) a OpenCV (BGR)"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

def track_color(img, color_ranges):
    """Tracking de color usando espacio HSV"""
    # Convertir a HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    results = {}
    
    for color_name, (lower, upper) in color_ranges.items():
        # Crear máscara para el rango de color
        mask = cv2.inRange(hsv, lower, upper)
        
        # Aplicar operaciones morfológicas para limpiar la máscara
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen resultado
        result = img.copy()
        
        # Dibujar contornos y bounding boxes
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filtrar contornos pequeños
                # Dibujar contorno
                cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                
                # Dibujar bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Añadir etiqueta
                cv2.putText(result, color_name, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        results[color_name] = {
            'mask': mask,
            'result': result,
            'contours': len([c for c in contours if cv2.contourArea(c) > 500])
        }
    
    return hsv, results

def load_example_image():
    """Carga imagen de ejemplo"""
    # Crear imagen de ejemplo con objetos de colores
    img = np.ones((400, 600, 3), dtype=np.uint8) * 50
    
    # Objetos rojos
    cv2.circle(img, (150, 150), 50, (0, 0, 255), -1)
    cv2.rectangle(img, (200, 200), (300, 300), (0, 0, 200), -1)
    
    # Objetos verdes
    cv2.circle(img, (450, 150), 40, (0, 255, 0), -1)
    cv2.rectangle(img, (50, 300), (150, 380), (0, 200, 0), -1)
    
    # Objetos azules
    cv2.circle(img, (300, 100), 30, (255, 0, 0), -1)
    cv2.rectangle(img, (400, 250), (500, 350), (200, 0, 0), -1)
    
    return img

# Definir rangos de color en HSV
COLOR_RANGES = {
    'Rojo': (np.array([0, 50, 50]), np.array([10, 255, 255])),
    'Verde': (np.array([40, 50, 50]), np.array([80, 255, 255])),
    'Azul': (np.array([100, 50, 50]), np.array([130, 255, 255]))
}

# Sidebar para configuración
st.sidebar.header("🛠️ Configuración")

image_source = st.sidebar.radio(
    "Selecciona imagen:",
    ["🖼️ Imagen de ejemplo", "📤 Cargar imagen"]
)

# Selección de colores a trackear
st.sidebar.subheader("🎨 Colores a Trackear")
selected_colors = {}
for color_name in COLOR_RANGES.keys():
    if st.sidebar.checkbox(color_name, value=True):
        selected_colors[color_name] = COLOR_RANGES[color_name]

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
    img_name = "ejemplo_colores.jpg"
    st.sidebar.success(f"✅ Usando: {img_name}")

# Mostrar información de la imagen
if img is not None and selected_colors:
    height, width = img.shape[:2]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Info de la Imagen:**")
    st.sidebar.write(f"• **Dimensiones:** {width} x {height}")
    st.sidebar.write(f"• **Colores seleccionados:** {len(selected_colors)}")

if img is not None and selected_colors:
    # Procesar imagen
    try:
        with st.spinner("🔄 Procesando tracking de color..."):
            hsv_img, results = track_color(img, selected_colors)
        
        st.success("✅ **Color tracking completado**")
        
        # Métricas
        cols = st.columns(len(selected_colors) + 1)
        with cols[0]:
            st.metric("Imagen", f"{img.shape[1]}x{img.shape[0]}")
        
        for i, (color_name, result_data) in enumerate(results.items()):
            with cols[i + 1]:
                st.metric(f"Objetos {color_name}", result_data['contours'])
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    # Mostrar código
    st.subheader("📄 Código Principal:")
    st.code("""
# Color Tracking - Código principal
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
""", language="python")
    
    # Resultados
    st.subheader("🖼️ Resultados por Color:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Original**")
        st.image(cv2_to_pil(img), use_container_width=True)
        
        st.markdown("**Imagen HSV**")
        st.image(cv2_to_pil(hsv_img), use_container_width=True)
    
    with col2:
        for color_name, result_data in results.items():
            st.markdown(f"**Tracking: {color_name}**")
            st.image(cv2_to_pil(result_data['result']), use_container_width=True)
            
            st.markdown(f"**Máscara: {color_name}**")
            st.image(result_data['mask'], use_container_width=True, clamp=True)
    
    # Explicación
    st.subheader("📚 Explicación:")
    st.markdown("""
    **Color Tracking** usando espacio de color HSV:
    
    1. **Conversión HSV**: Separar Hue (tono), Saturation (saturación), Value (valor)
    2. **Rangos de Color**: Definir límites inferior y superior para cada color
    3. **Máscara**: Crear máscara binaria para píxeles en el rango
    4. **Morfología**: Limpiar la máscara con operaciones opening/closing
    5. **Contornos**: Encontrar y dibujar contornos de los objetos detectados
    """)

elif img is not None and not selected_colors:
    st.warning("⚠️ Selecciona al menos un color para trackear")
else:
    st.error("❌ No se pudo cargar la imagen")