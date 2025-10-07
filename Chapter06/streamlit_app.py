"""
Capítulo 6 - Seam Carving
Demostración del código reduce_image_by_seam_carving.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 6 - Seam Carving", layout="wide")

# Título
st.title("✂️ Capítulo 6: Seam Carving")

def cv2_to_pil(cv2_img):
    """Convierte imagen de OpenCV (BGR) a PIL (RGB)"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    """Convierte imagen de PIL (RGB) a OpenCV (BGR)"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

def compute_energy_matrix(img):
    """Calcular matriz de energía usando gradientes Sobel"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

def find_vertical_seam(img, energy):
    """Encontrar seam vertical usando programación dinámica"""
    rows, cols = img.shape[:2]
    seam = np.zeros(img.shape[0])
    
    # Matrices de programación dinámica
    dist_to = np.zeros(img.shape[:2]) + float('inf')
    dist_to[0, :] = np.zeros(img.shape[1])
    edge_to = np.zeros(img.shape[:2])
    
    # Calcular paths óptimos
    for row in range(rows - 1):
        for col in range(cols):
            if col != 0 and dist_to[row + 1, col - 1] > dist_to[row, col] + energy[row + 1, col - 1]:
                dist_to[row + 1, col - 1] = dist_to[row, col] + energy[row + 1, col - 1]
                edge_to[row + 1, col - 1] = 1
            
            if dist_to[row + 1, col] > dist_to[row, col] + energy[row + 1, col]:
                dist_to[row + 1, col] = dist_to[row, col] + energy[row + 1, col]
                edge_to[row + 1, col] = 0
            
            if col != cols - 1 and dist_to[row + 1, col + 1] > dist_to[row, col] + energy[row + 1, col + 1]:
                dist_to[row + 1, col + 1] = dist_to[row, col] + energy[row + 1, col + 1]
                edge_to[row + 1, col + 1] = -1
    
    # Reconstruir path
    seam[rows - 1] = np.argmin(dist_to[rows - 1, :])
    for i in (x for x in reversed(range(rows)) if x > 0):
        seam[i - 1] = seam[i] + edge_to[i, int(seam[i])]
    
    return seam

def remove_vertical_seam(img, seam):
    """Remover seam vertical de la imagen"""
    rows, cols = img.shape[:2]
    for row in range(rows):
        for col in range(int(seam[row]), cols - 1):
            img[row, col] = img[row, col + 1]
    return img[:, 0:cols - 1]

def overlay_seam(img, seam):
    """Dibujar seam en la imagen"""
    img_with_seam = np.copy(img)
    x_coords, y_coords = np.transpose([(i, int(j)) for i, j in enumerate(seam)])
    img_with_seam[x_coords, y_coords] = (0, 255, 0)
    return img_with_seam

def apply_seam_carving(img, num_seams):
    """Aplicar seam carving para reducir ancho"""
    img_copy = np.copy(img)
    
    for i in range(num_seams):
        energy = compute_energy_matrix(img_copy)
        seam = find_vertical_seam(img_copy, energy)
        img_copy = remove_vertical_seam(img_copy, seam)
    
    return img_copy

def load_example_image():
    """Carga imagen de ejemplo"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'images', 'beach.jpg')
    
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    else:
        # Crear imagen de ejemplo
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (350, 250), (100, 150, 200), -1)
        cv2.circle(img, (200, 150), 50, (200, 100, 50), -1)
        return img

# Sidebar para configuración
st.sidebar.header("🛠️ Configuración")

image_source = st.sidebar.radio(
    "Selecciona imagen:",
    ["🖼️ Imagen de ejemplo", "📤 Cargar imagen"]
)

num_seams = st.sidebar.slider("Número de seams a remover", 1, 50, 10)

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
    img_name = "beach.jpg"
    st.sidebar.success(f"✅ Usando: {img_name}")

# Mostrar información de la imagen
if img is not None:
    height, width = img.shape[:2]
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📊 Info de la Imagen:**")
    st.sidebar.write(f"• **Dimensiones:** {width} x {height}")
    st.sidebar.write(f"• **Seams a remover:** {num_seams}")
    st.sidebar.write(f"• **Nuevo ancho:** {width - num_seams}")

if img is not None:
    # Procesar imagen
    try:
        with st.spinner("🔄 Procesando seam carving..."):
            # Calcular energía y primer seam
            energy = compute_energy_matrix(img)
            seam = find_vertical_seam(img, energy)
            img_with_seam = overlay_seam(img, seam)
            
            # Aplicar seam carving
            result = apply_seam_carving(img, num_seams)
        
        st.success("✅ **Seam carving completado**")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Imagen Original", f"{img.shape[1]}x{img.shape[0]}")
        with col2:
            st.metric("Seams Removidos", num_seams)
        with col3:
            st.metric("Imagen Final", f"{result.shape[1]}x{result.shape[0]}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    # Mostrar código
    st.subheader("📄 Código Principal:")
    st.code("""
# Seam Carving - Código principal
energy = compute_energy_matrix(img)
seam = find_vertical_seam(img, energy) 
img_result = remove_vertical_seam(img, seam)
""", language="python")
    
    # Resultados
    st.subheader("🖼️ Resultados:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Original**")
        st.image(cv2_to_pil(img), use_container_width=True)
        
        st.markdown("**Primer Seam Detectado**")
        st.image(cv2_to_pil(img_with_seam), use_container_width=True)
    
    with col2:
        st.markdown("**Matriz de Energía**")
        st.image(energy, use_container_width=True, clamp=True)
        
        st.markdown(f"**Resultado Final (-{num_seams} seams)**")
        st.image(cv2_to_pil(result), use_container_width=True)
    
    # Explicación
    st.subheader("📚 Explicación:")
    st.markdown("""
    **Seam Carving** es un algoritmo para redimensionar imágenes de manera inteligente:
    
    1. **Matriz de Energía**: Calcula gradientes usando filtros Sobel
    2. **Programación Dinámica**: Encuentra el camino de menor energía (seam)
    3. **Remoción**: Elimina píxeles del seam vertical
    4. **Iteración**: Repite el proceso para el número deseado de seams
    """)

else:
    st.error("❌ No se pudo cargar la imagen")