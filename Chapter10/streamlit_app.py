"""
Capítulo 10 - Estimación de Pose
Demostración del código pose_estimation.py
"""

import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image

# Configuración de la página
st.set_page_config(page_title="Capítulo 10 - Pose Estimation", layout="wide")

# Título
st.title("📍 Capítulo 10: Estimación de Pose")

def cv2_to_pil(cv2_img):
    """Convierte imagen de OpenCV (BGR) a PIL (RGB)"""
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def pil_to_cv2(pil_img):
    """Convierte imagen de PIL (RGB) a OpenCV (BGR)"""
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

def estimate_pose_simple(img):
    """Estimación de pose simplificada usando características"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar características usando SIFT
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Dibujar keypoints
    img_with_keypoints = cv2.drawKeypoints(
        img, keypoints, None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    # Simulación de pose estimation con puntos 3D ficticios
    # En una aplicación real, estos serían puntos conocidos del objeto 3D
    object_3d_points = np.array([
        [0, 0, 0],      # Origen
        [1, 0, 0],      # Eje X
        [0, 1, 0],      # Eje Y
        [0, 0, -1],     # Eje Z
        [1, 1, 0],      # Esquina
        [1, 0, -1],     # Esquina
        [0, 1, -1],     # Esquina
        [1, 1, -1]      # Esquina
    ], dtype=np.float32)
    
    # Parámetros de cámara simulados
    # En una aplicación real, estos se obtendrían de la calibración
    camera_matrix = np.array([
        [800, 0, img.shape[1]/2],
        [0, 800, img.shape[0]/2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros((4, 1))
    
    # Si tenemos suficientes keypoints, simular correspondencias
    if len(keypoints) >= 4:
        # Tomar los primeros 8 keypoints como correspondencias simuladas
        image_points = np.array([
            [kp.pt[0], kp.pt[1]] for kp in keypoints[:8]
        ], dtype=np.float32)
        
        # Si tenemos menos de 8 keypoints, usar solo los disponibles
        num_points = min(len(keypoints), 8)
        image_points = image_points[:num_points]
        object_points_used = object_3d_points[:num_points]
        
        try:
            # Resolver PnP para estimar pose
            success, rvec, tvec = cv2.solvePnP(
                object_points_used, image_points, 
                camera_matrix, dist_coeffs
            )
            
            if success:
                # Proyectar ejes 3D a la imagen
                axis_3d = np.array([
                    [0, 0, 0],    # Origen
                    [3, 0, 0],    # Eje X (rojo)
                    [0, 3, 0],    # Eje Y (verde)
                    [0, 0, -3]    # Eje Z (azul)
                ], dtype=np.float32)
                
                axis_2d, _ = cv2.projectPoints(
                    axis_3d, rvec, tvec, camera_matrix, dist_coeffs
                )
                
                # Dibujar ejes en la imagen
                img_with_pose = img.copy()
                origin = tuple(map(int, axis_2d[0].ravel()))
                x_axis = tuple(map(int, axis_2d[1].ravel()))
                y_axis = tuple(map(int, axis_2d[2].ravel()))
                z_axis = tuple(map(int, axis_2d[3].ravel()))
                
                # Dibujar líneas de los ejes
                cv2.arrowedLine(img_with_pose, origin, x_axis, (0, 0, 255), 3)  # X - Rojo
                cv2.arrowedLine(img_with_pose, origin, y_axis, (0, 255, 0), 3)  # Y - Verde  
                cv2.arrowedLine(img_with_pose, origin, z_axis, (255, 0, 0), 3)  # Z - Azul
                
                return {
                    'success': True,
                    'keypoints': keypoints,
                    'img_with_keypoints': img_with_keypoints,
                    'img_with_pose': img_with_pose,
                    'rvec': rvec,
                    'tvec': tvec,
                    'num_features': len(keypoints)
                }
                
        except cv2.error:
            pass
    
    # Si no se puede estimar pose, devolver solo keypoints
    return {
        'success': False,
        'keypoints': keypoints,
        'img_with_keypoints': img_with_keypoints,
        'img_with_pose': img,
        'num_features': len(keypoints)
    }

def load_example_image():
    """Carga imagen de ejemplo"""
    # Crear imagen de ejemplo con patrones geométricos
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Dibujar un patrón similar a un tablero de ajedrez simplificado
    for i in range(0, 400, 50):
        for j in range(0, 600, 50):
            if (i//50 + j//50) % 2 == 0:
                cv2.rectangle(img, (j, i), (j+50, i+50), (0, 0, 0), -1)
    
    # Añadir algunos círculos para más características
    cv2.circle(img, (150, 150), 20, (255, 0, 0), -1)
    cv2.circle(img, (450, 150), 20, (0, 255, 0), -1)
    cv2.circle(img, (300, 300), 20, (0, 0, 255), -1)
    
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
    img_name = "patron_ejemplo.jpg"
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
        with st.spinner("🔄 Procesando estimación de pose..."):
            results = estimate_pose_simple(img)
        
        if results['success']:
            st.success("✅ **Pose estimada correctamente**")
        else:
            st.warning("⚠️ **Pose parcial** - Solo características detectadas")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Imagen Original", f"{img.shape[1]}x{img.shape[0]}")
        with col2:
            st.metric("Características SIFT", results['num_features'])
        with col3:
            pose_status = "✅" if results['success'] else "⚠️"
            st.metric("Estimación Pose", pose_status)
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()
    
    # Mostrar código
    st.subheader("📄 Código Principal:")
    st.code("""
# Pose Estimation - Código principal
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Puntos 3D del objeto (conocidos)
object_points = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,-1]], dtype=np.float32)
# Puntos 2D correspondientes en la imagen
image_points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints[:4]], dtype=np.float32)

# Resolver PnP para obtener pose
success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
""", language="python")
    
    # Resultados
    st.subheader("🖼️ Resultados:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Imagen Original**")
        st.image(cv2_to_pil(img), use_container_width=True)
        
        st.markdown("**Características SIFT**")
        st.image(cv2_to_pil(results['img_with_keypoints']), use_container_width=True)
    
    with col2:
        if results['success']:
            st.markdown("**Pose Estimada (con ejes 3D)**")
            st.image(cv2_to_pil(results['img_with_pose']), use_container_width=True)
            
            # Mostrar vectores de rotación y traslación
            st.markdown("**Parámetros de Pose:**")
            st.write(f"**Vector Rotación (rvec):** {results['rvec'].flatten()}")
            st.write(f"**Vector Traslación (tvec):** {results['tvec'].flatten()}")
        else:
            st.markdown("**Sin Estimación de Pose**")
            st.image(cv2_to_pil(results['img_with_pose']), use_container_width=True)
            st.info("💡 Se necesitan al menos 4 correspondencias confiables para estimar pose")
    
    # Explicación
    st.subheader("📚 Explicación:")
    st.markdown("""
    **Pose Estimation** determina la posición y orientación de un objeto en 3D:
    
    1. **Detección de Características**: Usa SIFT para encontrar puntos clave
    2. **Correspondencias**: Relaciona puntos 2D de la imagen con puntos 3D del objeto
    3. **Problema PnP**: Resuelve Perspective-n-Point usando cv2.solvePnP
    4. **Parámetros de Pose**: Obtiene vectores de rotación (rvec) y traslación (tvec)
    5. **Visualización**: Proyecta ejes 3D a la imagen para mostrar la pose
    """)

else:
    st.error("❌ No se pudo cargar la imagen")