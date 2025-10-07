"""
Aplicación Streamlit - Estimación de Pose y Realidad Aumentada
Aplicación educativa para explorar técnicas de estimación de pose, tracking y realidad aumentada
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
import matplotlib.pyplot as plt
from collections import namedtuple

# Imports opcionales - usar fallbacks si no están disponibles
# Optional dependencies - handle gracefully if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Configuración de la página
st.set_page_config(
    page_title="Estimación de Pose y AR",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎯 Estimación de Pose y Realidad Aumentada")
st.markdown("**Explora técnicas de tracking, estimación de pose y aplicaciones de realidad aumentada**")

# Clases adaptadas del código original
class PoseEstimator(object):
    """Estimador de pose basado en el código original"""
    def __init__(self):
        # Usar algoritmo Locality Sensitive Hashing
        flann_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        
        self.min_matches = 10
        self.cur_target = namedtuple('Current', 'image, rect, keypoints, descriptors, data')
        self.tracked_target = namedtuple('Tracked', 'target, points_prev, points_cur, H, quad')
        
        self.feature_detector = cv2.ORB_create()
        self.feature_detector.setMaxFeatures(1000)
        
        try:
            self.feature_matcher = cv2.FlannBasedMatcher(flann_params, {})
        except:
            # Fallback a BF matcher si FLANN no está disponible
            self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.tracking_targets = []

    def add_target(self, image, rect, data=None):
        """Agregar nuevo objetivo para tracking"""
        x_start, y_start, x_end, y_end = rect
        keypoints, descriptors = [], []
        
        for keypoint, descriptor in zip(*self.detect_features(image)):
            x, y = keypoint.pt
            if x_start <= x <= x_end and y_start <= y <= y_end:
                keypoints.append(keypoint)
                descriptors.append(descriptor)

        if len(descriptors) == 0:
            return False

        descriptors = np.array(descriptors, dtype='uint8')
        
        # Agregar descriptores al matcher
        if hasattr(self.feature_matcher, 'add'):
            self.feature_matcher.add([descriptors])
        
        target = self.cur_target(image=image, rect=rect, keypoints=keypoints, 
                                descriptors=descriptors, data=data)
        self.tracking_targets.append(target)
        return True

    def track_target(self, frame):
        """Detectar y seguir objetivos en el frame actual"""
        self.cur_keypoints, self.cur_descriptors = self.detect_features(frame)

        if len(self.cur_keypoints) < self.min_matches:
            return []

        if len(self.tracking_targets) == 0 or self.cur_descriptors is None:
            return []

        try:
            if hasattr(self.feature_matcher, 'knnMatch'):
                matches = self.feature_matcher.knnMatch(self.cur_descriptors, k=2)
                # Filtro de ratio test de Lowe
                matches = [match[0] for match in matches if len(match) == 2 and 
                          match[0].distance < match[1].distance * 0.75]
            else:
                # BF Matcher fallback
                all_matches = []
                for i, target in enumerate(self.tracking_targets):
                    matches_bf = self.feature_matcher.match(self.cur_descriptors, target.descriptors)
                    for match in matches_bf:
                        match.imgIdx = i
                        all_matches.append(match)
                matches = sorted(all_matches, key=lambda x: x.distance)[:100]
        
        except Exception as e:
            st.warning(f"Error en matching: {e}")
            return []

        if len(matches) < self.min_matches:
            return []

        # Agrupar matches por índice de imagen
        matches_using_index = [[] for _ in range(len(self.tracking_targets))]
        for match in matches:
            matches_using_index[match.imgIdx].append(match)

        tracked = []
        for image_index, matches in enumerate(matches_using_index):
            if len(matches) < self.min_matches:
                continue

            target = self.tracking_targets[image_index]
            points_prev = [target.keypoints[m.trainIdx].pt for m in matches]
            points_cur = [self.cur_keypoints[m.queryIdx].pt for m in matches]
            points_prev, points_cur = np.float32((points_prev, points_cur))
            
            # Encontrar homografía
            H, status = cv2.findHomography(points_prev, points_cur, 
                                         cv2.RANSAC, 3.0)
            
            if H is None:
                continue
                
            status = status.ravel() != 0

            if status.sum() < self.min_matches:
                continue

            points_prev, points_cur = points_prev[status], points_cur[status]

            # Transformar rectángulo de referencia
            x_start, y_start, x_end, y_end = target.rect
            quad = np.float32([[x_start, y_start], [x_end, y_start], 
                              [x_end, y_end], [x_start, y_end]])
            quad = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H).reshape(-1, 2)
            
            track = self.tracked_target(target=target, points_prev=points_prev, 
                                      points_cur=points_cur, H=H, quad=quad)
            tracked.append(track)

        tracked.sort(key=lambda x: len(x.points_prev), reverse=True)
        return tracked

    def detect_features(self, frame):
        """Detectar características en el frame"""
        keypoints, descriptors = self.feature_detector.detectAndCompute(frame, None)
        if descriptors is None:
            descriptors = []
        return keypoints, descriptors

    def clear_targets(self):
        """Limpiar todos los objetivos existentes"""
        if hasattr(self.feature_matcher, 'clear'):
            self.feature_matcher.clear()
        self.tracking_targets = []

# Funciones auxiliares
@st.cache_data
def load_sample_images():
    """Carga imágenes de ejemplo para tracking"""
    
    # Crear imágenes sintéticas con patrones reconocibles
    reference_images = {}
    
    # Imagen 1: Patrón de tablero de ajedrez
    chess_pattern = np.zeros((200, 200, 3), dtype=np.uint8)
    square_size = 25
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                chess_pattern[i*square_size:(i+1)*square_size, 
                            j*square_size:(j+1)*square_size] = [255, 255, 255]
    
    # Agregar borde colorido para mejor tracking
    cv2.rectangle(chess_pattern, (10, 10), (190, 190), (0, 255, 0), 3)
    reference_images['tablero_ajedrez'] = chess_pattern
    
    # Imagen 2: Patrón circular
    circular_pattern = np.ones((200, 200, 3), dtype=np.uint8) * 240
    center = (100, 100)
    for radius in range(20, 90, 15):
        color = (50 + radius, 100, 200 - radius)
        cv2.circle(circular_pattern, center, radius, color, 2)
    
    # Agregar cruz central
    cv2.line(circular_pattern, (80, 100), (120, 100), (255, 0, 0), 3)
    cv2.line(circular_pattern, (100, 80), (100, 120), (255, 0, 0), 3)
    reference_images['patron_circular'] = circular_pattern
    
    # Imagen 3: Texto con características
    text_pattern = np.ones((200, 200, 3), dtype=np.uint8) * 250
    cv2.putText(text_pattern, 'TRACK', (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 
               2, (0, 0, 0), 3)
    cv2.putText(text_pattern, 'ME!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 
               2, (0, 0, 255), 3)
    
    # Agregar esquinas para mejor tracking
    cv2.circle(text_pattern, (20, 20), 8, (0, 255, 0), -1)
    cv2.circle(text_pattern, (180, 20), 8, (0, 255, 0), -1)
    cv2.circle(text_pattern, (20, 180), 8, (0, 255, 0), -1)
    cv2.circle(text_pattern, (180, 180), 8, (0, 255, 0), -1)
    reference_images['patron_texto'] = text_pattern
    
    return reference_images

@st.cache_data
def create_test_scene(reference_img, transform_type='rotation', param=30):
    """Crea una escena de prueba con el objeto de referencia transformado"""
    
    h, w = reference_img.shape[:2]
    
    # Crear escena más grande
    scene = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Agregar ruido de fondo
    noise_points = 50
    for _ in range(noise_points):
        x = np.random.randint(0, 600)
        y = np.random.randint(0, 400)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(scene, (x, y), np.random.randint(2, 5), color, -1)
    
    # Calcular transformación
    if transform_type == 'rotation':
        # Rotación
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, param, 1.0)
        transformed = cv2.warpAffine(reference_img, M, (w, h))
        
    elif transform_type == 'scale':
        # Escalado
        scale_factor = param / 100.0
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        transformed = cv2.resize(reference_img, (new_w, new_h))
        # Rellenar con fondo si es más pequeño
        if scale_factor < 1:
            temp = np.ones((h, w, 3), dtype=np.uint8) * 240
            y_offset = (h - new_h) // 2
            x_offset = (w - new_w) // 2
            temp[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = transformed
            transformed = temp
        else:
            # Recortar si es más grande
            y_offset = (new_h - h) // 2
            x_offset = (new_w - w) // 2
            transformed = transformed[y_offset:y_offset+h, x_offset:x_offset+w]
            
    elif transform_type == 'perspective':
        # Transformación perspectiva
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        offset = param
        pts2 = np.float32([[offset, 0], [w-offset, offset], [w, h], [0, h-offset]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        transformed = cv2.warpPerspective(reference_img, M, (w, h))
    
    else:  # translation
        # Traslación
        M = np.float32([[1, 0, param], [0, 1, param//2]])
        transformed = cv2.warpAffine(reference_img, M, (w, h))
    
    # Colocar objeto transformado en la escena
    y_pos = 100
    x_pos = 200
    
    # Manejar caso donde la imagen transformada podría salirse de los límites
    end_y = min(y_pos + h, 400)
    end_x = min(x_pos + w, 600)
    crop_h = end_y - y_pos
    crop_w = end_x - x_pos
    
    scene[y_pos:end_y, x_pos:end_x] = transformed[:crop_h, :crop_w]
    
    # Información de la posición real para evaluación
    real_rect = (x_pos, y_pos, x_pos + crop_w, y_pos + crop_h)
    
    return scene, real_rect

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def analyze_homography(H):
    """Analiza matriz de homografía para extraer información de pose"""
    if H is None:
        return None
    
    try:
        # Descomponer la homografía
        # Extraer rotación, traslación y escala aproximados
        
        # Normalizando por el elemento (2,2)
        H_norm = H / H[2, 2]
        
        # Extraer componentes de traslación
        tx = H_norm[0, 2]
        ty = H_norm[1, 2]
        
        # Extraer escala y rotación aproximados de la parte superior izquierda 2x2
        a = H_norm[0, 0]
        b = H_norm[0, 1] 
        c = H_norm[1, 0]
        d = H_norm[1, 1]
        
        # Calcular escala
        scale_x = np.sqrt(a*a + c*c)
        scale_y = np.sqrt(b*b + d*d)
        
        # Calcular ángulo de rotación
        angle = np.arctan2(c/scale_x, a/scale_x) * 180 / np.pi
        
        # Calcular sesgo (shear)
        shear = (a*b + c*d) / (scale_x * scale_y)
        
        return {
            'translation_x': tx,
            'translation_y': ty,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'rotation_angle': angle,
            'shear': shear,
            'determinant': np.linalg.det(H_norm[:2, :2])
        }
    except:
        return None

def draw_3d_object(img, quad, H):
    """Dibuja un objeto 3D simple usando la homografía estimada"""
    try:
        # Definir un cubo 3D simple
        # Base del cubo (en el plano de la imagen de referencia)
        h = 50  # Altura del cubo
        
        # Puntos de la base (mismo plano que la imagen de referencia)
        x1, y1 = quad[0]
        x2, y2 = quad[1]  
        x3, y3 = quad[2]
        x4, y4 = quad[3]
        
        # Puntos de la parte superior (elevados)
        # Simular elevación usando perspectiva
        offset_x, offset_y = -20, -20  # Desplazamiento para simular altura
        
        top_quad = quad.copy()
        top_quad[:, 0] += offset_x
        top_quad[:, 1] += offset_y
        
        # Dibujar base
        cv2.polylines(img, [np.int32(quad)], True, (0, 255, 0), 2)
        
        # Dibujar parte superior
        cv2.polylines(img, [np.int32(top_quad)], True, (0, 255, 255), 2)
        
        # Conectar base con parte superior
        for i in range(4):
            pt1 = tuple(np.int32(quad[i]))
            pt2 = tuple(np.int32(top_quad[i]))
            cv2.line(img, pt1, pt2, (255, 0, 0), 2)
            
        return True
    except:
        return False

# Sidebar - Controles
st.sidebar.header("⚙️ Controles")

# Modo de aplicación
app_mode = st.sidebar.selectbox(
    "🎯 Modo de Aplicación:",
    options=['pose_estimation', 'homography_analysis', 'ar_simulation', 'feature_matching'],
    format_func=lambda x: {
        'pose_estimation': '🎯 Estimación de Pose',
        'homography_analysis': '📐 Análisis de Homografía',
        'ar_simulation': '🌟 Simulación AR',
        'feature_matching': '🔍 Matching de Características'
    }[x]
)

st.sidebar.markdown("---")

# Selector de imagen de referencia
reference_images = load_sample_images()
reference_options = list(reference_images.keys()) + ['custom']

selected_reference = st.sidebar.selectbox(
    "🖼️ Imagen de Referencia:",
    options=reference_options,
    format_func=lambda x: {
        'tablero_ajedrez': '♟️ Tablero de Ajedrez',
        'patron_circular': '⭕ Patrón Circular', 
        'patron_texto': '📝 Patrón de Texto',
        'custom': '📁 Imagen Personalizada'
    }.get(x, x)
)

if selected_reference == 'custom':
    uploaded_ref = st.sidebar.file_uploader(
        "📁 Sube imagen de referencia", 
        type=['png', 'jpg', 'jpeg'],
        help="Imagen con características detectables"
    )
    
    if uploaded_ref is not None:
        pil_image = Image.open(uploaded_ref)
        reference_img = pil_to_cv2(pil_image)
        ref_source = f"📁 {uploaded_ref.name}"
    else:
        reference_img = reference_images['tablero_ajedrez']
        ref_source = "♟️ Tablero de Ajedrez (por defecto)"
else:
    reference_img = reference_images[selected_reference]
    ref_source = f"🖼️ {selected_reference}"

st.sidebar.info(f"**Referencia:** {ref_source}")

st.sidebar.markdown("---")

# Controles específicos por modo
if app_mode in ['pose_estimation', 'homography_analysis', 'ar_simulation']:
    st.sidebar.subheader("🔄 Transformación de Prueba")
    
    transform_type = st.sidebar.selectbox(
        "Tipo de transformación:",
        options=['rotation', 'scale', 'perspective', 'translation'],
        format_func=lambda x: {
            'rotation': '🔄 Rotación',
            'scale': '🔍 Escalado',
            'perspective': '📐 Perspectiva',
            'translation': '📍 Traslación'
        }[x]
    )
    
    if transform_type == 'rotation':
        transform_param = st.sidebar.slider("Ángulo de rotación (°):", -180, 180, 30, 5)
    elif transform_type == 'scale':
        transform_param = st.sidebar.slider("Escala (%):", 50, 200, 100, 10)
    elif transform_type == 'perspective':
        transform_param = st.sidebar.slider("Distorsión perspectiva:", 0, 50, 20, 5)
    else:  # translation
        transform_param = st.sidebar.slider("Desplazamiento (px):", -100, 100, 30, 10)

elif app_mode == 'feature_matching':
    st.sidebar.subheader("🔍 Parámetros de Matching")
    
    min_matches = st.sidebar.slider("Mínimo de matches:", 5, 50, 10, 5)
    ratio_threshold = st.sidebar.slider("Umbral ratio test:", 0.5, 1.0, 0.75, 0.05)

# Configuración del detector
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Configuración Detector")

max_features = st.sidebar.slider("Máximo de características:", 100, 2000, 1000, 100)
scale_factor = st.sidebar.slider("Factor de escala:", 1.1, 2.0, 1.2, 0.1)
n_levels = st.sidebar.slider("Niveles de pirámide:", 4, 12, 8, 1)

# Crear escena de prueba
if app_mode in ['pose_estimation', 'homography_analysis', 'ar_simulation']:
    test_scene, real_rect = create_test_scene(reference_img, transform_type, transform_param)
else:
    test_scene, real_rect = create_test_scene(reference_img, 'rotation', 30)

# Inicializar estimador de pose
pose_estimator = PoseEstimator()
pose_estimator.feature_detector.setMaxFeatures(max_features)

# Procesamiento según el modo
if app_mode == 'pose_estimation':
    st.subheader("🎯 Estimación de Pose - Resultados")
    
    # Definir región de referencia (toda la imagen de referencia)
    h_ref, w_ref = reference_img.shape[:2]
    reference_rect = (0, 0, w_ref, h_ref)
    
    # Agregar objetivo al estimador
    success = pose_estimator.add_target(reference_img, reference_rect)
    
    if success:
        # Realizar tracking
        tracked_objects = pose_estimator.track_target(test_scene)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen de Referencia**")
            st.image(cv2_to_pil(reference_img), use_container_width=True)
        
        with col2:
            st.write("**🎯 Detección en Escena**")
            result_img = test_scene.copy()
            
            if tracked_objects:
                for item in tracked_objects:
                    # Dibujar contorno detectado
                    cv2.polylines(result_img, [np.int32(item.quad)], True, (0, 255, 0), 3)
                    
                    # Dibujar puntos matched
                    for (x, y) in np.int32(item.points_cur):
                        cv2.circle(result_img, (x, y), 3, (255, 255, 0), -1)
                
                st.image(cv2_to_pil(result_img), use_container_width=True)
                
                # Información de detección
                item = tracked_objects[0]  # Primer objeto detectado
                st.subheader("📊 Información de Pose")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Matches Encontrados", len(item.points_cur))
                
                with col2:
                    # Calcular área del quad detectado
                    area = cv2.contourArea(item.quad)
                    st.metric("Área Detectada", f"{area:.0f} px²")
                
                with col3:
                    # Calcular centro del objeto
                    center_x = np.mean(item.quad[:, 0])
                    center_y = np.mean(item.quad[:, 1])
                    st.metric("Centro X", f"{center_x:.0f}")
                
                with col4:
                    st.metric("Centro Y", f"{center_y:.0f}")
                
                # Análisis de homografía
                if item.H is not None:
                    pose_info = analyze_homography(item.H)
                    if pose_info:
                        st.subheader("📐 Análisis de Transformación")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Rotación", f"{pose_info['rotation_angle']:.1f}°")
                            st.metric("Escala X", f"{pose_info['scale_x']:.2f}")
                        
                        with col2:
                            st.metric("Traslación X", f"{pose_info['translation_x']:.1f}")
                            st.metric("Escala Y", f"{pose_info['scale_y']:.2f}")
                        
                        with col3:
                            st.metric("Traslación Y", f"{pose_info['translation_y']:.1f}")
                            st.metric("Determinante", f"{pose_info['determinant']:.3f}")
            
            else:
                st.image(cv2_to_pil(result_img), use_container_width=True)
                st.warning("⚠️ No se pudo detectar el objeto en la escena")
                st.info("💡 Intenta ajustar los parámetros o usar una imagen con más características")

elif app_mode == 'homography_analysis':
    st.subheader("📐 Análisis de Homografía - Resultados")
    
    # Realizar estimación de pose
    h_ref, w_ref = reference_img.shape[:2]
    reference_rect = (0, 0, w_ref, h_ref)
    
    pose_estimator.add_target(reference_img, reference_rect)
    tracked_objects = pose_estimator.track_target(test_scene)
    
    if tracked_objects:
        item = tracked_objects[0]
        
        # Visualización de la homografía
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📐 Matriz de Homografía**")
            if item.H is not None:
                # Mostrar matriz
                if PANDAS_AVAILABLE:
                    df_h = pd.DataFrame(item.H, 
                                       columns=['H₀₀', 'H₀₁', 'H₀₂'],
                                       index=['H₁₀', 'H₁₁', 'H₁₂'])
                    st.dataframe(df_h.round(4))
                else:
                    # Fallback: mostrar como tabla simple con numpy
                    H_rounded = np.round(item.H, 4)
                    st.write("Matriz H (3x3):")
                    for i, row in enumerate(H_rounded):
                        cols = st.columns(3)
                        for j, val in enumerate(row):
                            cols[j].metric(f"H[{i},{j}]", f"{val:.4f}")
                
                # Propiedades de la matriz
                det = np.linalg.det(item.H)
                condition_num = np.linalg.cond(item.H)
                
                st.write("**Propiedades:**")
                st.write(f"- Determinante: {det:.4f}")
                st.write(f"- Número de condición: {condition_num:.2f}")
                
                if abs(det) < 1e-6:
                    st.error("⚠️ Matriz singular (no invertible)")
                elif condition_num > 100:
                    st.warning("⚠️ Matriz mal condicionada")
                else:
                    st.success("✅ Matriz bien condicionada")
        
        with col2:
            st.write("**🔄 Descomposición de Transformación**")
            pose_info = analyze_homography(item.H)
            
            if pose_info:
                # Crear gráfico de barras con los componentes
                components = ['Rotación (°)', 'Escala X', 'Escala Y', 
                            'Traslación X', 'Traslación Y']
                values = [pose_info['rotation_angle'], pose_info['scale_x'], 
                         pose_info['scale_y'], pose_info['translation_x']/10, 
                         pose_info['translation_y']/10]
                
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.bar(components, values, color=['red', 'blue', 'blue', 'green', 'green'])
                ax.set_title('Componentes de la Transformación')
                ax.set_ylabel('Valor')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
        
        # Comparación con transformación real
        st.subheader("📊 Comparación con Transformación Real")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🎯 Parámetros Aplicados**")
            st.write(f"Tipo: {transform_type}")
            st.write(f"Valor: {transform_param}")
            if transform_type == 'rotation':
                st.write(f"Ángulo real: {transform_param}°")
            elif transform_type == 'scale':
                st.write(f"Escala real: {transform_param/100:.2f}")
        
        with col2:
            st.write("**🔍 Parámetros Detectados**")
            if pose_info:
                if transform_type == 'rotation':
                    detected = pose_info['rotation_angle']
                    st.write(f"Ángulo detectado: {detected:.1f}°")
                    error = abs(detected - transform_param)
                    st.write(f"Error: {error:.1f}°")
                elif transform_type == 'scale':
                    detected_scale = (pose_info['scale_x'] + pose_info['scale_y']) / 2
                    st.write(f"Escala detectada: {detected_scale:.2f}")
                    real_scale = transform_param / 100
                    error = abs(detected_scale - real_scale)
                    st.write(f"Error: {error:.2f}")
        
        with col3:
            st.write("**📈 Calidad de Detección**")
            st.metric("Matches Usados", len(item.points_cur))
            
            # Calcular error de reproyección
            if len(item.points_prev) > 0:
                # Calcular puntos reproyectados
                points_reproj = cv2.perspectiveTransform(
                    item.points_prev.reshape(-1, 1, 2), item.H
                ).reshape(-1, 2)
                
                # Error RMS
                errors = np.sqrt(np.sum((item.points_cur - points_reproj)**2, axis=1))
                rms_error = np.sqrt(np.mean(errors**2))
                st.metric("Error RMS", f"{rms_error:.2f} px")
        
        # Visualización de la imagen con homografía aplicada
        st.subheader("🖼️ Visualización de Resultados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen de Referencia**")
            st.image(cv2_to_pil(reference_img), caption="Patrón original")
        
        with col2:
            st.write("**🎯 Detección en Escena**")
            result_img = test_scene.copy()
            
            # Dibujar contorno detectado
            cv2.polylines(result_img, [np.int32(item.quad)], True, (0, 255, 0), 3)
            
            # Dibujar puntos matched
            for (x, y) in np.int32(item.points_cur):
                cv2.circle(result_img, (x, y), 3, (255, 255, 0), -1)
            
            # Dibujar centro
            center = np.mean(item.quad, axis=0)
            cv2.circle(result_img, tuple(np.int32(center)), 5, (0, 0, 255), -1)
            
            st.image(cv2_to_pil(result_img), caption="Objeto detectado y pose estimada")
            
    else:
        st.warning("⚠️ No se pudo detectar el objeto en la escena")
        st.info("💡 Intenta ajustar los parámetros del detector o usa una imagen con más características")
        
        # Mostrar imágenes sin detección
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen de Referencia**")
            st.image(cv2_to_pil(reference_img), caption="Patrón de referencia")
        
        with col2:
            st.write("**🎯 Escena de Prueba**")
            st.image(cv2_to_pil(test_scene), caption="Escena sin detección exitosa")

elif app_mode == 'ar_simulation':
    st.subheader("🌟 Simulación de Realidad Aumentada")
    
    # Realizar estimación de pose
    h_ref, w_ref = reference_img.shape[:2]
    reference_rect = (0, 0, w_ref, h_ref)
    
    pose_estimator.add_target(reference_img, reference_rect)
    tracked_objects = pose_estimator.track_target(test_scene)
    
    # Opciones de AR
    ar_object = st.sidebar.selectbox(
        "Objeto AR:",
        options=['cube', 'pyramid', 'axes', 'text'],
        format_func=lambda x: {
            'cube': '📦 Cubo 3D',
            'pyramid': '🔺 Pirámide',
            'axes': '📐 Ejes XYZ',
            'text': '📝 Texto AR'
        }[x]
    )
    
    if tracked_objects:
        item = tracked_objects[0]
        ar_result = test_scene.copy()
        
        # Dibujar objeto detectado
        cv2.polylines(ar_result, [np.int32(item.quad)], True, (0, 255, 0), 2)
        
        # Agregar objeto AR
        if ar_object == 'cube':
            draw_3d_object(ar_result, item.quad, item.H)
            
        elif ar_object == 'pyramid':
            # Dibujar pirámide
            center = np.mean(item.quad, axis=0)
            apex = center + np.array([-30, -40])  # Punto superior
            
            # Conectar apex con cada esquina de la base
            for corner in item.quad:
                cv2.line(ar_result, tuple(np.int32(apex)), 
                        tuple(np.int32(corner)), (255, 0, 255), 2)
            
            # Dibujar base
            cv2.polylines(ar_result, [np.int32(item.quad)], True, (0, 255, 255), 2)
            
        elif ar_object == 'axes':
            # Dibujar ejes XYZ
            center = np.mean(item.quad, axis=0)
            
            # Definir vectores de ejes en coordenadas de la imagen de referencia
            axis_length = 60
            axes_3d = np.float32([
                [0, 0, 0],           # Origen
                [axis_length, 0, 0], # X (rojo)
                [0, axis_length, 0], # Y (verde) 
                [0, 0, -axis_length] # Z (azul)
            ]).reshape(-1, 3)
            
            # Proyectar usando homografía (aproximación 2D)
            origin_2d = center
            x_axis_2d = center + np.array([axis_length, 0])
            y_axis_2d = center + np.array([0, axis_length])
            z_axis_2d = center + np.array([-20, -40])  # Simular profundidad
            
            # Dibujar ejes
            cv2.arrowedLine(ar_result, tuple(np.int32(origin_2d)), 
                           tuple(np.int32(x_axis_2d)), (0, 0, 255), 3)  # X - Rojo
            cv2.arrowedLine(ar_result, tuple(np.int32(origin_2d)), 
                           tuple(np.int32(y_axis_2d)), (0, 255, 0), 3)  # Y - Verde
            cv2.arrowedLine(ar_result, tuple(np.int32(origin_2d)), 
                           tuple(np.int32(z_axis_2d)), (255, 0, 0), 3)  # Z - Azul
            
            # Etiquetas
            cv2.putText(ar_result, 'X', tuple(np.int32(x_axis_2d + [10, 0])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(ar_result, 'Y', tuple(np.int32(y_axis_2d + [0, -10])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(ar_result, 'Z', tuple(np.int32(z_axis_2d + [-10, -10])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        elif ar_object == 'text':
            # Texto AR sobre el objeto
            center = np.mean(item.quad, axis=0)
            text_pos = center + np.array([0, -50])
            
            cv2.putText(ar_result, 'AR OBJECT', tuple(np.int32(text_pos)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            # Información adicional
            info_pos = center + np.array([0, 80])
            cv2.putText(ar_result, f'Matches: {len(item.points_cur)}', 
                       tuple(np.int32(info_pos)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Mostrar resultado
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Escena Original**")
            st.image(cv2_to_pil(test_scene), use_container_width=True)
        
        with col2:
            st.write(f"**🌟 Resultado AR - {ar_object.title()}**")
            st.image(cv2_to_pil(ar_result), use_container_width=True)
        
        # Información AR
        st.subheader("📊 Información de AR")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Objeto AR", ar_object.title())
        
        with col2:
            st.metric("Tracking Activo", "✅ SÍ" if tracked_objects else "❌ NO")
        
        with col3:
            confidence = min(len(item.points_cur) / 50, 1.0) * 100
            st.metric("Confianza", f"{confidence:.0f}%")
        
        with col4:
            fps_sim = 30  # FPS simulado
            st.metric("FPS (Simulado)", fps_sim)
    
    else:
        st.warning("⚠️ No se puede aplicar AR sin detección de objeto")
        st.image(cv2_to_pil(test_scene), use_container_width=True)

elif app_mode == 'feature_matching':
    st.subheader("🔍 Análisis de Matching de Características")
    
    # Detectar características en ambas imágenes
    ref_kps, ref_desc = pose_estimator.detect_features(reference_img)
    scene_kps, scene_desc = pose_estimator.detect_features(test_scene)
    
    if ref_desc is not None and scene_desc is not None and len(ref_desc) > 0 and len(scene_desc) > 0:
        # Crear matcher específico para análisis
        if hasattr(cv2, 'BFMatcher'):
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            matches = matcher.knnMatch(ref_desc, scene_desc, k=2)
            
            # Filtro ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
        else:
            good_matches = []
        
        # Crear imagen de matches
        if len(good_matches) >= min_matches:
            # Seleccionar mejores matches
            good_matches = sorted(good_matches, key=lambda x: x.distance)[:50]
            
            # Crear imagen combinada
            img_matches = cv2.drawMatches(reference_img, ref_kps, test_scene, scene_kps, 
                                        good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**🔍 Matches de Características**")
                st.image(cv2_to_pil(img_matches), use_container_width=True)
            
            with col2:
                st.write("**📊 Estadísticas**")
                st.metric("Características Ref.", len(ref_kps))
                st.metric("Características Escena", len(scene_kps))
                st.metric("Matches Totales", len(good_matches))
                
                if len(good_matches) > 0:
                    avg_distance = np.mean([m.distance for m in good_matches])
                    st.metric("Distancia Promedio", f"{avg_distance:.1f}")
                
                # Calidad del matching
                if len(good_matches) >= min_matches * 2:
                    st.success("✅ Excelente matching")
                elif len(good_matches) >= min_matches:
                    st.warning("⚠️ Matching suficiente") 
                else:
                    st.error("❌ Matching insuficiente")
            
            # Distribución de distancias
            st.subheader("📈 Distribución de Distancias de Matches")
            
            distances = [m.distance for m in good_matches]
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(distances, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Distancia de Match')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Distancias de Matches')
            ax.axvline(np.mean(distances), color='red', linestyle='--', 
                      label=f'Promedio: {np.mean(distances):.1f}')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        
        else:
            st.error("❌ Matches insuficientes para análisis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📷 Características Referencia**")
                ref_with_kps = cv2.drawKeypoints(reference_img, ref_kps, None, 
                                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                st.image(cv2_to_pil(ref_with_kps), use_container_width=True)
                st.write(f"Características detectadas: {len(ref_kps)}")
            
            with col2:
                st.write("**📷 Características Escena**")
                scene_with_kps = cv2.drawKeypoints(test_scene, scene_kps, None, 
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                st.image(cv2_to_pil(scene_with_kps), use_container_width=True)
                st.write(f"Características detectadas: {len(scene_kps)}")
    
    else:
        st.error("❌ No se pudieron detectar características suficientes")

# Botón de descarga
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if 'ar_result' in locals():
        processed_pil = cv2_to_pil(ar_result)
    elif 'img_matches' in locals():
        processed_pil = cv2_to_pil(img_matches)
    else:
        processed_pil = cv2_to_pil(test_scene)
    
    buf = io.BytesIO()
    processed_pil.save(buf, format='PNG')
    
    st.download_button(
        label="📥 Descargar Resultado",
        data=buf.getvalue(),
        file_name=f"pose_estimation_{app_mode}.png",
        mime="image/png",
        use_container_width=True
    )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("🎯 Estimación de Pose"):
    st.markdown("""
    **Concepto:**
    - Determinar posición y orientación de objeto conocido
    - Basado en correspondencias 2D-3D o 2D-2D
    - Usa homografía para objetos planares
    
    **Pipeline:**
    1. Detectar características en referencia y escena
    2. Encontrar correspondencias (matching)
    3. Estimar homografía con RANSAC
    4. Extraer pose de la homografía
    
    **Aplicaciones:**
    - Realidad aumentada
    - Robótica y navegación
    - Calibración de cámaras
    """)

with st.sidebar.expander("📐 Homografía"):
    st.markdown("""
    **Matriz de Homografía (3x3):**
    - Representa transformación perspectiva entre planos
    - 8 grados de libertad (9 elementos, normalización)
    - Preserva líneas rectas
    
    **Componentes:**
    - Rotación y traslación en el plano
    - Escalado anisotrópico
    - Transformación perspectiva
    
    **Estimación:**
    - Mínimo 4 correspondencias (sin ruido)
    - RANSAC para datos con outliers
    - DLT (Direct Linear Transform)
    """)

with st.sidebar.expander("🌟 Realidad Aumentada"):
    st.markdown("""
    **Elementos Clave:**
    - Tracking en tiempo real
    - Registro preciso (alineación)
    - Renderizado de objetos virtuales
    
    **Tipos de AR:**
    - Basada en marcadores (fiducial)
    - Sin marcadores (markerless)
    - Simultaneous Localization and Mapping (SLAM)
    
    **Desafíos:**
    - Robustez ante oclusiones
    - Condiciones de iluminación variables
    - Velocidad de procesamiento
    """)

with st.sidebar.expander("💡 Consejos Prácticos"):
    st.markdown("""
    **Para mejor tracking:**
    - Usar objetos con texturas ricas
    - Evitar superficies especulares
    - Buena iluminación uniforme
    - Mínimo de bordes y esquinas
    
    **Parámetros críticos:**
    - Número mínimo de matches
    - Umbral RANSAC
    - Ratio test para filtering
    - Máximo número de características
    
    **Problemas comunes:**
    - Tracking perdido: Aumentar características
    - Jitter: Filtrado temporal
    - Drift: Re-inicialización periódica
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎯 <strong>Estimación de Pose y Realidad Aumentada</strong> | Capítulo 10 - Visión 3D</p>
        <p><small>Explora técnicas de tracking, estimación de pose y aplicaciones de realidad aumentada</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)