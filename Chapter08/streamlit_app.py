"""
Aplicación Streamlit - Seguimiento de Objetos y Detección de Movimiento
Aplicación educativa para explorar técnicas de seguimiento por color y detección de movimiento
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
import matplotlib.pyplot as plt
from collections import deque
import threading

# Configuración de la página
st.set_page_config(
    page_title="Seguimiento de Objetos Interactive",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎯 Seguimiento de Objetos y Detección de Movimiento")
st.markdown("**Explora técnicas de seguimiento por color, detección de movimiento y análisis temporal**")

# Funciones auxiliares
@st.cache_data
def load_sample_video_frames():
    """Carga frames de ejemplo para simular video"""
    frames = []
    
    # Crear secuencia de frames simulados con objeto en movimiento
    for i in range(20):
        # Frame base
        frame = np.ones((400, 600, 3), dtype=np.uint8) * 50
        
        # Agregar ruido de fondo
        noise = np.random.randint(0, 30, frame.shape).astype(np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Agregar objetos estáticos
        cv2.rectangle(frame, (50, 50), (100, 100), (100, 100, 100), -1)
        cv2.circle(frame, (500, 80), 30, (80, 80, 80), -1)
        
        # Objeto en movimiento (azul) - movimiento circular
        angle = (i / 20.0) * 2 * np.pi
        center_x = int(300 + 100 * np.cos(angle))
        center_y = int(200 + 50 * np.sin(angle))
        
        # Crear objeto azul (HSV: [120, 255, 255])
        cv2.circle(frame, (center_x, center_y), 25, (255, 120, 0), -1)  # BGR para azul
        
        # Objeto rojo en movimiento lineal
        red_x = int(50 + i * 25)
        red_y = 300
        if red_x < 550:
            cv2.circle(frame, (red_x, red_y), 20, (0, 0, 255), -1)  # BGR para rojo
        
        # Objeto verde oscilante
        green_x = 400
        green_y = int(150 + 80 * np.sin(i * 0.3))
        cv2.rectangle(frame, (green_x-15, green_y-15), (green_x+15, green_y+15), (0, 255, 0), -1)
        
        frames.append(frame)
    
    return frames

@st.cache_data
def load_sample_image():
    """Carga una imagen de ejemplo"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Intentar cargar imagen existente
    sample_images = ['blue_carpet.png', 'green_dots.png', 'test_image.jpg']
    
    for img_name in sample_images:
        img_path = os.path.join(script_dir, 'images', img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_name
    
    # Crear imagen de ejemplo con objetos de diferentes colores
    img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    
    # Objetos azules
    cv2.circle(img, (150, 150), 40, (255, 100, 0), -1)  # Azul claro
    cv2.circle(img, (450, 150), 35, (200, 150, 0), -1)  # Azul medio
    cv2.rectangle(img, (100, 250), (200, 350), (150, 80, 0), -1)  # Azul oscuro
    
    # Objetos rojos
    cv2.circle(img, (300, 100), 30, (0, 0, 255), -1)
    cv2.rectangle(img, (350, 250), (450, 350), (0, 0, 200), -1)
    
    # Objetos verdes
    cv2.circle(img, (500, 300), 45, (0, 255, 0), -1)
    cv2.rectangle(img, (50, 50), (120, 120), (0, 200, 0), -1)
    
    # Objetos amarillos
    cv2.circle(img, (300, 300), 25, (0, 255, 255), -1)
    
    return img, "colores_ejemplo.png"

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

# Funciones de seguimiento por color (basado en código original)
def track_color_hsv(frame, lower_hsv, upper_hsv, blur_kernel=5):
    """
    Seguimiento por color en espacio HSV (método original)
    """
    # Convertir a HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Crear máscara para el rango de color
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    
    # Aplicar la máscara
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Aplicar filtro mediano como en el código original
    if blur_kernel > 1:
        result = cv2.medianBlur(result, ksize=blur_kernel)
    
    return result, mask, hsv_frame

def track_color_advanced(frame, lower_hsv, upper_hsv, morphology_ops=True, 
                        kernel_size=5, contour_analysis=True):
    """
    Seguimiento por color avanzado con operaciones morfológicas
    """
    # Seguimiento básico
    result, mask, hsv_frame = track_color_hsv(frame, lower_hsv, upper_hsv)
    
    # Operaciones morfológicas para limpiar la máscara
    if morphology_ops:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apertura para eliminar ruido
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Cierre para llenar huecos
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Volver a aplicar la máscara limpia
        result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Análisis de contornos
    contour_info = None
    if contour_analysis:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Encontrar el contorno más grande
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 100:  # Filtrar contornos pequeños
                # Calcular información del contorno
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calcular bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Calcular área y perímetro
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    
                    contour_info = {
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area,
                        'perimeter': perimeter,
                        'contour': largest_contour
                    }
                    
                    # Dibujar información en el resultado
                    cv2.circle(result, (cx, cy), 5, (255, 255, 255), -1)
                    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    cv2.drawContours(result, [largest_contour], -1, (255, 255, 255), 2)
    
    return result, mask, contour_info

def detect_motion_frame_diff(frame1, frame2, threshold=30, blur_kernel=5):
    """
    Detección de movimiento usando diferencia de frames
    """
    # Convertir a escala de grises
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    if blur_kernel > 1:
        gray1 = cv2.GaussianBlur(gray1, (blur_kernel, blur_kernel), 0)
        gray2 = cv2.GaussianBlur(gray2, (blur_kernel, blur_kernel), 0)
    
    # Calcular diferencia absoluta
    diff = cv2.absdiff(gray1, gray2)
    
    # Umbralización
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos de movimiento
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear imagen de resultado
    motion_result = frame2.copy()
    motion_areas = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filtrar movimientos pequeños
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(motion_result, (x, y), (x+w, y+h), (0, 255, 0), 2)
            motion_areas.append((x, y, w, h, area))
    
    return motion_result, diff, thresh, motion_areas

def background_subtraction_mog(frames, history=200, threshold=16, shadows=True):
    """
    Sustracción de fondo usando MOG (Mixture of Gaussians)
    """
    # Crear el sustractor de fondo
    backSub = cv2.createBackgroundSubtractorMOG2(history=history, 
                                                varThreshold=threshold, 
                                                detectShadows=shadows)
    
    results = []
    for i, frame in enumerate(frames):
        # Aplicar sustracción de fondo
        fg_mask = backSub.apply(frame)
        
        # Limpiar la máscara
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear resultado
        result = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        results.append((result, fg_mask))
    
    return results

def analyze_color_distribution(frame, mask):
    """
    Analiza la distribución de colores en las regiones detectadas
    """
    # Aplicar máscara
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Convertir a HSV
    hsv = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2HSV)
    
    # Calcular histogramas solo en píxeles no negros
    hist_h = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], mask, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], mask, [256], [0, 256])
    
    return hist_h, hist_s, hist_v

# Sidebar - Controles
st.sidebar.header("⚙️ Controles")

# Modo de aplicación
app_mode = st.sidebar.selectbox(
    "🎯 Modo de Aplicación:",
    options=['color_tracking', 'motion_detection', 'background_subtraction', 'color_analysis'],
    format_func=lambda x: {
        'color_tracking': '🎨 Seguimiento por Color',
        'motion_detection': '🏃 Detección de Movimiento',
        'background_subtraction': '🌅 Sustracción de Fondo',
        'color_analysis': '📊 Análisis de Color'
    }[x]
)

st.sidebar.markdown("---")

# Upload de imagen/video
if app_mode in ['motion_detection', 'background_subtraction']:
    st.sidebar.subheader("📹 Datos de Video")
    use_sample_video = st.sidebar.checkbox("Usar secuencia de ejemplo", True)
    
    if not use_sample_video:
        uploaded_files = st.sidebar.file_uploader(
            "📁 Sube secuencia de imágenes", 
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Sube múltiples imágenes para formar una secuencia"
        )
        
        if uploaded_files and len(uploaded_files) >= 2:
            frames = []
            for uploaded_file in uploaded_files:
                pil_image = Image.open(uploaded_file)
                frame = pil_to_cv2(pil_image)
                frames.append(frame)
            data_source = f"📁 {len(uploaded_files)} imágenes subidas"
        else:
            frames = load_sample_video_frames()
            data_source = "🖼️ Secuencia de ejemplo (20 frames)"
    else:
        frames = load_sample_video_frames()
        data_source = "🖼️ Secuencia de ejemplo (20 frames)"
else:
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
        image, sample_name = load_sample_image()
        data_source = f"🖼️ {sample_name}"

# Información de datos
if app_mode in ['motion_detection', 'background_subtraction']:
    st.sidebar.info(f"**Datos:** {data_source}")
    if 'frames' in locals():
        h, w = frames[0].shape[:2]
        st.sidebar.info(f"**Frames:** {len(frames)}")
        st.sidebar.info(f"**Dimensiones:** {w} x {h} píxeles")
else:
    h, w = image.shape[:2]
    st.sidebar.info(f"**Imagen:** {data_source}")
    st.sidebar.info(f"**Dimensiones:** {w} x {h} píxeles")

st.sidebar.markdown("---")

# Controles específicos por modo
if app_mode == 'color_tracking':
    st.sidebar.subheader("🎨 Seguimiento por Color")
    
    # Predefinidos de color
    color_preset = st.sidebar.selectbox(
        "Color predefinido:",
        options=['custom', 'blue', 'red', 'green', 'yellow', 'cyan', 'magenta'],
        format_func=lambda x: {
            'custom': '🎨 Personalizado',
            'blue': '🔵 Azul',
            'red': '🔴 Rojo', 
            'green': '🟢 Verde',
            'yellow': '🟡 Amarillo',
            'cyan': '🔷 Cian',
            'magenta': '🟣 Magenta'
        }[x]
    )
    
    # Rangos HSV predefinidos
    color_ranges = {
        'blue': ([100, 50, 50], [130, 255, 255]),
        'red': ([0, 50, 50], [10, 255, 255]),  # Rojo tiene dos rangos
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 50, 50], [30, 255, 255]),
        'cyan': ([80, 50, 50], [100, 255, 255]),
        'magenta': ([140, 50, 50], [170, 255, 255])
    }
    
    if color_preset == 'custom':
        st.sidebar.write("**Rango HSV Personalizado:**")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.write("Mínimo:")
            h_min = st.slider("H", 0, 179, 100, key="h_min")
            s_min = st.slider("S", 0, 255, 50, key="s_min") 
            v_min = st.slider("V", 0, 255, 50, key="v_min")
        
        with col2:
            st.write("Máximo:")
            h_max = st.slider("H", 0, 179, 130, key="h_max")
            s_max = st.slider("S", 0, 255, 255, key="s_max")
            v_max = st.slider("V", 0, 255, 255, key="v_max")
        
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])
    else:
        lower_hsv = np.array(color_ranges[color_preset][0])
        upper_hsv = np.array(color_ranges[color_preset][1])
    
    # Opciones de procesamiento
    tracking_method = st.sidebar.selectbox(
        "Método:",
        options=['basic', 'advanced'],
        format_func=lambda x: {
            'basic': '🎯 Básico (Original)',
            'advanced': '🔬 Avanzado'
        }[x]
    )
    
    blur_kernel = st.sidebar.slider("Filtro mediano:", 1, 15, 5, 2)
    
    if tracking_method == 'advanced':
        morphology_ops = st.sidebar.checkbox("Operaciones morfológicas", True)
        kernel_size = st.sidebar.slider("Tamaño kernel:", 3, 15, 5, 2)
        contour_analysis = st.sidebar.checkbox("Análisis de contornos", True)

elif app_mode == 'motion_detection':
    st.sidebar.subheader("🏃 Detección de Movimiento")
    
    motion_method = st.sidebar.selectbox(
        "Método:",
        options=['frame_diff', 'accumulated_diff'],
        format_func=lambda x: {
            'frame_diff': '📊 Diferencia de Frames',
            'accumulated_diff': '📈 Diferencia Acumulada'
        }[x]
    )
    
    threshold = st.sidebar.slider("Umbral de movimiento:", 10, 100, 30, 5)
    blur_kernel = st.sidebar.slider("Desenfoque Gaussiano:", 1, 15, 5, 2)
    min_area = st.sidebar.slider("Área mínima:", 100, 2000, 500, 100)

elif app_mode == 'background_subtraction':
    st.sidebar.subheader("🌅 Sustracción de Fondo")
    
    history = st.sidebar.slider("Historia (frames):", 50, 500, 200, 50)
    var_threshold = st.sidebar.slider("Umbral varianza:", 8, 50, 16, 2)
    detect_shadows = st.sidebar.checkbox("Detectar sombras", True)

elif app_mode == 'color_analysis':
    st.sidebar.subheader("📊 Análisis de Color")
    
    analysis_mode = st.sidebar.selectbox(
        "Tipo de análisis:",
        options=['histogram', 'dominant_colors', 'color_space'],
        format_func=lambda x: {
            'histogram': '📈 Histogramas',
            'dominant_colors': '🎨 Colores Dominantes',
            'color_space': '🌈 Espacios de Color'
        }[x]
    )

# Procesamiento según el modo
if app_mode == 'color_tracking':
    st.subheader("🎨 Seguimiento por Color - Resultados")
    
    if tracking_method == 'basic':
        result, mask, hsv_frame = track_color_hsv(image, lower_hsv, upper_hsv, blur_kernel)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2_to_pil(image), use_container_width=True)
        
        with col2:
            st.write("**🎯 Resultado Seguimiento**")
            st.image(cv2_to_pil(result), use_container_width=True)
        
        # Mostrar máscara y HSV
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔲 Máscara de Color**")
            st.image(cv2_to_pil(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)), use_container_width=True)
        
        with col2:
            st.write("**🌈 Imagen HSV**")
            st.image(cv2_to_pil(hsv_frame), use_container_width=True)
    
    else:  # advanced
        result, mask, contour_info = track_color_advanced(
            image, lower_hsv, upper_hsv, morphology_ops, kernel_size, contour_analysis
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📷 Imagen Original**")
            st.image(cv2_to_pil(image), use_container_width=True)
        
        with col2:
            st.write("**🔬 Seguimiento Avanzado**")
            st.image(cv2_to_pil(result), use_container_width=True)
        
        # Información del contorno
        if contour_info:
            st.subheader("📊 Información del Objeto Detectado")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Centro X", contour_info['center'][0])
            
            with col2:
                st.metric("Centro Y", contour_info['center'][1])
            
            with col3:
                st.metric("Área", f"{contour_info['area']:.0f} px²")
            
            with col4:
                st.metric("Perímetro", f"{contour_info['perimeter']:.0f} px")
            
            # Información del bounding box
            x, y, w, h = contour_info['bbox']
            st.info(f"**Bounding Box:** ({x}, {y}) - {w}×{h} píxeles")

elif app_mode == 'motion_detection':
    st.subheader("🏃 Detección de Movimiento - Resultados")
    
    if len(frames) >= 2:
        # Selector de frame
        frame_idx = st.slider("Frame actual:", 1, len(frames)-1, 1)
        
        if motion_method == 'frame_diff':
            motion_result, diff, thresh, motion_areas = detect_motion_frame_diff(
                frames[frame_idx-1], frames[frame_idx], threshold, blur_kernel
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**📷 Frame {frame_idx}**")
                st.image(cv2_to_pil(frames[frame_idx]), use_container_width=True)
            
            with col2:
                st.write("**🏃 Movimiento Detectado**")
                st.image(cv2_to_pil(motion_result), use_container_width=True)
            
            # Mostrar diferencias
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📊 Diferencia de Frames**")
                st.image(cv2_to_pil(cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)), use_container_width=True)
            
            with col2:
                st.write("**🔲 Umbralización**")
                st.image(cv2_to_pil(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)), use_container_width=True)
            
            # Información de movimiento
            if motion_areas:
                st.subheader("📊 Áreas de Movimiento Detectadas")
                
                for i, (x, y, w, h, area) in enumerate(motion_areas):
                    st.write(f"**Área {i+1}:** Posición ({x}, {y}), Tamaño {w}×{h}, Área {area:.0f} px²")

elif app_mode == 'background_subtraction':
    st.subheader("🌅 Sustracción de Fondo - Resultados")
    
    if len(frames) >= 5:
        # Procesar sustracción de fondo
        results = background_subtraction_mog(frames, history, var_threshold, detect_shadows)
        
        # Selector de frame
        frame_idx = st.slider("Frame:", 0, len(results)-1, len(results)//2)
        
        result_frame, fg_mask = results[frame_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**📷 Frame Original {frame_idx+1}**")
            st.image(cv2_to_pil(frames[frame_idx]), use_container_width=True)
        
        with col2:
            st.write("**🌅 Primer Plano Detectado**")
            st.image(cv2_to_pil(result_frame), use_container_width=True)
        
        # Mostrar máscara de primer plano
        st.write("**🔲 Máscara de Primer Plano**")
        st.image(cv2_to_pil(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2RGB)), use_container_width=True)
        
        # Animación automática
        if st.button("▶️ Reproducir Secuencia"):
            placeholder = st.empty()
            
            for i, (result_frame, fg_mask) in enumerate(results):
                with placeholder.container():
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Frame {i+1}**")
                        st.image(cv2_to_pil(frames[i]), use_container_width=True)
                    
                    with col2:
                        st.write(f"**Resultado {i+1}**")
                        st.image(cv2_to_pil(result_frame), use_container_width=True)
                
                time.sleep(0.3)

elif app_mode == 'color_analysis':
    st.subheader("📊 Análisis de Color - Resultados")
    
    if analysis_mode == 'histogram':
        # Análisis de histogramas por canal
        
        # Crear máscara para análisis (toda la imagen)
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        hist_h, hist_s, hist_v = analyze_color_distribution(image, mask)
        
        # Mostrar histogramas
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        ax1.plot(hist_h, color='red')
        ax1.set_title('Histograma Matiz (H)')
        ax1.set_xlabel('Matiz')
        ax1.set_ylabel('Frecuencia')
        
        ax2.plot(hist_s, color='green')
        ax2.set_title('Histograma Saturación (S)')
        ax2.set_xlabel('Saturación')
        ax2.set_ylabel('Frecuencia')
        
        ax3.plot(hist_v, color='blue')
        ax3.set_title('Histograma Valor (V)')
        ax3.set_xlabel('Valor')
        ax3.set_ylabel('Frecuencia')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Mostrar imagen original
        st.write("**📷 Imagen Analizada**")
        st.image(cv2_to_pil(image), use_container_width=True)

# Botones de descarga
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if app_mode == 'color_tracking':
        if 'result' in locals():
            processed_pil = cv2_to_pil(result)
            buf = io.BytesIO()
            processed_pil.save(buf, format='PNG')
            
            st.download_button(
                label="📥 Descargar Seguimiento",
                data=buf.getvalue(),
                file_name=f"color_tracking_{color_preset}.png",
                mime="image/png",
                use_container_width=True
            )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("🎨 Seguimiento por Color"):
    st.markdown("""
    **Principio:**
    - Convierte imagen a espacio HSV
    - Define rango de color objetivo
    - Crea máscara binaria
    - Aplica máscara a imagen original
    
    **Ventajas HSV:**
    - Separación de color y luminancia
    - Mayor robustez a cambios de iluminación
    - Rangos de color más intuitivos
    
    **Aplicaciones:**
    - Seguimiento de objetos por color
    - Detección de uniformes deportivos
    - Control por gestos con objetos coloridos
    """)

with st.sidebar.expander("🏃 Detección de Movimiento"):
    st.markdown("""
    **Diferencia de Frames:**
    - Resta frames consecutivos
    - Umbraliza diferencias significativas
    - Encuentra contornos de cambios
    
    **Sustracción de Fondo:**
    - Modelo estadístico del fondo
    - Actualización adaptativa
    - Detección de sombras opcional
    
    **Aplicaciones:**
    - Sistemas de vigilancia
    - Detección de intrusos
    - Análisis de tráfico vehicular
    """)

with st.sidebar.expander("🔧 Parámetros Clave"):
    st.markdown("""
    **HSV - Matiz, Saturación, Valor:**
    - **H (0-179)**: Tipo de color puro
    - **S (0-255)**: Intensidad del color
    - **V (0-255)**: Brillo de la imagen
    
    **Filtros:**
    - **Mediano**: Elimina ruido impulsivo
    - **Gaussiano**: Suaviza uniformemente
    - **Morfológico**: Conecta/separa regiones
    
    **Umbrales:**
    - Muy bajo: Detecta micro-movimientos
    - Muy alto: Solo movimientos grandes
    - Óptimo: Balance ruido vs sensibilidad
    """)

with st.sidebar.expander("💡 Consejos Prácticos"):
    st.markdown("""
    **Para mejor seguimiento:**
    - Usar colores distintivos del fondo
    - Iluminación uniforme y estable
    - Objetos de tamaño adecuado
    - Minimizar reflejos y sombras
    
    **Problemas comunes:**
    - **Color similar al fondo**: Cambiar umbral HSV
    - **Múltiples detecciones**: Usar morfología
    - **Parpadeo**: Aumentar filtrado temporal
    - **Objetos perdidos**: Revisar rangos HSV
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎯 <strong>Seguimiento de Objetos y Detección de Movimiento</strong> | Capítulo 8 - Análisis Temporal</p>
        <p><small>Explora técnicas de seguimiento por color y detección de movimiento en tiempo real</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)