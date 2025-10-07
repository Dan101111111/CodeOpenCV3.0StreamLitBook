"""
Capítulo 1 - Transformaciones Afines
Demostración del código 09_affine_transformation.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 1 - Transformaciones Afines", layout="wide")

# Título
st.title("📐 Capítulo 1: Transformaciones Afines")

# Función principal de transformación afín
def apply_affine_transformation(img):
    """Aplica la transformación afín del código original"""
    rows, cols = img.shape[:2]
    
    # Puntos origen y destino (del código original)
    src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
    dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.4*(cols-1)),rows-1]])
    
    # Crear matriz de transformación afín
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    
    # Aplicar transformación
    img_output = cv2.warpAffine(img, affine_matrix, (cols, rows))
    
    return img_output, affine_matrix, src_points, dst_points

# Cargar imagen
def load_image():
    """Carga la imagen de entrada"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, 'images', 'input.jpg')
    
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    else:
        # Crear imagen de ejemplo si no existe
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (550, 350), (100, 150, 200), -1)
        cv2.rectangle(img, (100, 100), (500, 300), (50, 100, 150), 3)
        cv2.circle(img, (300, 200), 80, (200, 100, 50), -1)
        cv2.putText(img, 'OPENCV', (220, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

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
    img = load_image()
    img_name = "input.jpg"
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
        # Aplicar transformación
        transformed_img, matrix, src_pts, dst_pts = apply_affine_transformation(img)
        
        # Mostrar estado de validación
        st.success("✅ **Imagen procesada correctamente con OpenCV**")
        
        # Información adicional sobre el procesamiento
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Imagen Original", "OK ✅", f"{img.shape[1]}x{img.shape[0]}")
        with col2:
            st.metric("Transformación Afín", "OK ✅", f"{transformed_img.shape[1]}x{transformed_img.shape[0]}")
        
    except Exception as e:
        st.error(f"❌ **Error al procesar la imagen:** {str(e)}")
        st.info("💡 Intenta con una imagen diferente o verifica el formato")
        st.stop()
    
    # Mostrar código original
    st.subheader("📄 Código Original:")
    st.code("""
# Código del archivo 09_affine_transformation.py
import cv2
import numpy as np

img = cv2.imread('images/input.jpg')
rows, cols = img.shape[:2]

src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.4*(cols-1)),rows-1]])

affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))

cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()
""", language="python")
    
    # Mostrar resultados
    st.subheader("�️ Resultados:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Original**")
        st.image(cv2_to_pil(img), width="stretch")
    
    with col2:
        st.markdown("**Imagen Transformada**")
        st.image(cv2_to_pil(transformed_img), width="stretch")
    
    # Mostrar información técnica
    st.subheader("📊 Información Técnica:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Puntos de Origen:**")
        st.write(f"- Esquina superior izquierda: {src_pts[0]}")
        st.write(f"- Esquina superior derecha: {src_pts[1]}")
        st.write(f"- Esquina inferior izquierda: {src_pts[2]}")
        
        st.markdown("**Puntos de Destino:**")
        st.write(f"- Esquina superior izquierda: {dst_pts[0]}")
        st.write(f"- Esquina superior derecha: {dst_pts[1]}")
        st.write(f"- Esquina inferior izquierda: {dst_pts[2]}")
    
    with col2:
        st.markdown("**Matriz de Transformación Afín:**")
        st.write(matrix)
        
        st.markdown("**Dimensiones:**")
        rows, cols = img.shape[:2]
        st.write(f"- Alto: {rows} píxeles")
        st.write(f"- Ancho: {cols} píxeles")
    
    # Explicación
    st.subheader("� Explicación:")
    st.markdown("""
    Esta transformación afín aplica una **deformación de cizallamiento (shear)** a la imagen:
    
    - **Punto superior izquierdo**: Se mantiene fijo en (0,0)
    - **Punto superior derecho**: Se mueve hacia adentro al 60% del ancho
    - **Punto inferior izquierdo**: Se mueve hacia adentro al 40% del ancho
    
    El resultado es una imagen que parece inclinada hacia la derecha, como si estuvieras viendo un paralelogramo en lugar de un rectángulo.
    """)
    
    st.info("💡 **Nota**: Las transformaciones afines preservan las líneas rectas y las proporciones, pero pueden cambiar ángulos y formas.")

else:
    st.error("❌ No se pudo cargar la imagen")