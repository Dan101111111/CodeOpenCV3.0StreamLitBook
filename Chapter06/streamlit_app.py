"""
Aplicación Streamlit - Seam Carving Interactivo
Aplicación educativa para explorar técnicas de redimensionamiento inteligente de imágenes
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os

# Configuración de la página
st.set_page_config(
    page_title="Seam Carving Interactivo",
    page_icon="✂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("✂️ Seam Carving Interactivo")
st.markdown("**Explora técnicas de redimensionamiento inteligente preservando contenido importante**")

# Funciones auxiliares
@st.cache_data
def load_sample_image():
    """Carga una imagen de ejemplo desde la carpeta images"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Lista de imágenes disponibles (según el código original)
    sample_images = ['beach.jpg', 'ducks.jpg']
    
    for img_name in sample_images:
        img_path = os.path.join(script_dir, 'images', img_name)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                return img, img_name
    
    # Si no se encuentra ninguna imagen, crear una de ejemplo con contenido relevante
    img = np.ones((400, 600, 3), dtype=np.uint8) * 220
    
    # Crear paisaje con elementos importantes y fondo menos importante
    # Cielo (área menos importante)
    sky_gradient = np.linspace(200, 150, 150).astype(np.uint8)
    for i in range(150):
        img[i, :] = [sky_gradient[i], sky_gradient[i] + 30, 255]
    
    # Montañas (contenido semi-importante)
    mountain_points = []
    for x in range(0, 600, 50):
        height = 150 + int(50 * np.sin(x * 0.02)) + np.random.randint(-20, 20)
        mountain_points.append([x, height])
    
    mountain_points = np.array(mountain_points, np.int32)
    for i in range(len(mountain_points) - 1):
        cv2.line(img, tuple(mountain_points[i]), tuple(mountain_points[i+1]), (80, 120, 80), 3)
    
    # Agua (área menos importante, repetitiva)
    water_y = 320
    for y in range(water_y, 400):
        wave = int(5 * np.sin(y * 0.3))
        for x in range(600):
            wave_x = int(3 * np.sin(x * 0.1 + y * 0.05))
            color_var = wave + wave_x
            img[y, x] = [100 + color_var, 150 + color_var, 200 + color_var//2]
    
    # Árboles (contenido importante)
    tree_positions = [100, 200, 450, 520]
    for x_pos in tree_positions:
        # Tronco
        cv2.rectangle(img, (x_pos-5, 200), (x_pos+5, 320), (101, 67, 33), -1)
        # Copa
        cv2.circle(img, (x_pos, 180), 25, (34, 139, 34), -1)
        cv2.circle(img, (x_pos-15, 190), 20, (50, 150, 50), -1)
        cv2.circle(img, (x_pos+15, 190), 20, (50, 150, 50), -1)
    
    # Casa (contenido muy importante)
    house_x, house_y = 300, 250
    # Base de la casa
    cv2.rectangle(img, (house_x-40, house_y), (house_x+40, house_y+60), (150, 130, 100), -1)
    # Techo
    roof_points = np.array([[house_x-50, house_y], [house_x, house_y-40], [house_x+50, house_y]], np.int32)
    cv2.fillPoly(img, [roof_points], (120, 80, 60))
    # Ventanas
    cv2.rectangle(img, (house_x-30, house_y+15), (house_x-15, house_y+35), (255, 255, 150), -1)
    cv2.rectangle(img, (house_x+15, house_y+15), (house_x+30, house_y+35), (255, 255, 150), -1)
    # Puerta
    cv2.rectangle(img, (house_x-8, house_y+25), (house_x+8, house_y+60), (101, 67, 33), -1)
    
    # Agregar ruido sutil para mejor detección de energía
    noise = np.random.randint(-10, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img, "paisaje_generado.jpg"

def pil_to_cv2(pil_image):
    """Convierte imagen PIL a formato OpenCV"""
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convierte imagen OpenCV a formato PIL"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

# Funciones de Seam Carving (basadas en el código original)
def compute_energy_matrix(img, method='gradient'):
    """
    Calcula la matriz de energía usando diferentes métodos
    
    Args:
        img: Imagen de entrada
        method: Método de cálculo ('gradient', 'sobel', 'laplacian', 'scharr')
    
    Returns:
        Matriz de energía
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == 'gradient':
        # Método original del código
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        abs_sobel_x = cv2.convertScaleAbs(sobel_x)
        abs_sobel_y = cv2.convertScaleAbs(sobel_y)
        
        return cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    elif method == 'sobel':
        # Método Sobel mejorado
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
    
    elif method == 'laplacian':
        # Método Laplaciano
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.absolute(laplacian).astype(np.uint8)
    
    elif method == 'scharr':
        # Método Scharr (más preciso que Sobel)
        scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        return np.sqrt(scharr_x**2 + scharr_y**2).astype(np.uint8)

def find_vertical_seam(img, energy):
    """
    Encuentra el seam vertical de menor energía (código original optimizado)
    """
    rows, cols = img.shape[:2]
    
    # Inicializar el vector de seam
    seam = np.zeros(rows, dtype=np.int32)
    
    # Inicializar matrices de distancia y borde
    dist_to = np.full((rows, cols), float('inf'))
    dist_to[0, :] = energy[0, :]
    edge_to = np.zeros((rows, cols), dtype=np.int32)
    
    # Programación dinámica
    for row in range(rows - 1):
        for col in range(cols):
            # Verificar las tres direcciones posibles
            for dc in [-1, 0, 1]:
                new_col = col + dc
                if 0 <= new_col < cols:
                    new_dist = dist_to[row, col] + energy[row + 1, new_col]
                    if new_dist < dist_to[row + 1, new_col]:
                        dist_to[row + 1, new_col] = new_dist
                        edge_to[row + 1, new_col] = dc
    
    # Rastrear el camino de vuelta
    seam[rows - 1] = np.argmin(dist_to[rows - 1, :])
    for i in range(rows - 2, -1, -1):
        next_col = int(seam[i + 1] + edge_to[i + 1, int(seam[i + 1])])
        # Asegurar que el índice esté dentro de los límites
        seam[i] = np.clip(next_col, 0, cols - 1)
    
    return seam

def find_horizontal_seam(img, energy):
    """
    Encuentra el seam horizontal de menor energía
    """
    # Transponer imagen y energía, encontrar seam vertical, luego transponer resultado
    img_t = np.transpose(img, (1, 0, 2))
    energy_t = energy.T
    
    seam = find_vertical_seam(img_t, energy_t)
    return seam

def overlay_vertical_seam(img, seam, color=(0, 255, 0)):
    """
    Dibuja el seam vertical sobre la imagen (código original mejorado)
    """
    img_overlay = np.copy(img)
    
    for i, col in enumerate(seam):
        if 0 <= col < img.shape[1]:
            # Dibujar línea más gruesa para mejor visibilidad
            for offset in range(-1, 2):
                if 0 <= col + offset < img.shape[1]:
                    img_overlay[i, col + offset] = color
    
    return img_overlay

def overlay_horizontal_seam(img, seam, color=(255, 0, 0)):
    """
    Dibuja el seam horizontal sobre la imagen
    """
    img_overlay = np.copy(img)
    
    for j, row in enumerate(seam):
        if 0 <= row < img.shape[0]:
            # Dibujar línea más gruesa para mejor visibilidad
            for offset in range(-1, 2):
                if 0 <= row + offset < img.shape[0]:
                    img_overlay[row + offset, j] = color
    
    return img_overlay

def remove_vertical_seam(img, seam):
    """
    Elimina el seam vertical de la imagen (código original)
    """
    rows, cols = img.shape[:2]
    
    # Crear nueva imagen con una columna menos
    new_img = np.zeros((rows, cols - 1, img.shape[2]), dtype=img.dtype)
    
    for row in range(rows):
        col_to_remove = int(seam[row])
        if 0 <= col_to_remove < cols:
            # Copiar píxeles antes del seam
            new_img[row, :col_to_remove] = img[row, :col_to_remove]
            # Copiar píxeles después del seam
            if col_to_remove < cols - 1:
                new_img[row, col_to_remove:] = img[row, col_to_remove + 1:]
    
    return new_img

def remove_horizontal_seam(img, seam):
    """
    Elimina el seam horizontal de la imagen
    """
    rows, cols = img.shape[:2]
    
    # Crear nueva imagen con una fila menos
    new_img = np.zeros((rows - 1, cols, img.shape[2]), dtype=img.dtype)
    
    for col in range(cols):
        row_to_remove = int(seam[col])
        if 0 <= row_to_remove < rows:
            # Copiar píxeles antes del seam
            new_img[:row_to_remove, col] = img[:row_to_remove, col]
            # Copiar píxeles después del seam
            if row_to_remove < rows - 1:
                new_img[row_to_remove:, col] = img[row_to_remove + 1:, col]
    
    return new_img

def seam_carving_reduce(img, num_vertical=0, num_horizontal=0, energy_method='gradient', 
                      show_progress=False):
    """
    Reduce la imagen usando seam carving
    """
    result = np.copy(img)
    seams_overlay = np.copy(img)
    
    progress_images = []
    
    # Eliminar seams verticales
    for i in range(num_vertical):
        if result.shape[1] <= 1:
            break
            
        energy = compute_energy_matrix(result, energy_method)
        seam = find_vertical_seam(result, energy)
        
        # Guardar seam para visualización
        seams_overlay = overlay_vertical_seam(seams_overlay, seam, (0, 255, 0))
        
        # Eliminar seam
        result = remove_vertical_seam(result, seam)
        
        if show_progress and (i + 1) % max(1, num_vertical // 5) == 0:
            progress_images.append(np.copy(result))
    
    # Eliminar seams horizontales
    for i in range(num_horizontal):
        if result.shape[0] <= 1:
            break
            
        energy = compute_energy_matrix(result, energy_method)
        seam = find_horizontal_seam(result, energy)
        
        # Guardar seam para visualización
        seams_overlay = overlay_horizontal_seam(seams_overlay, seam, (255, 0, 0))
        
        # Eliminar seam
        result = remove_horizontal_seam(result, seam)
        
        if show_progress and (i + 1) % max(1, num_horizontal // 5) == 0:
            progress_images.append(np.copy(result))
    
    return result, seams_overlay, progress_images

def seam_carving_enlarge(img, num_vertical=0, num_horizontal=0, energy_method='gradient'):
    """
    Amplía la imagen duplicando los seams de menor energía
    """
    result = np.copy(img)
    
    # Ampliar verticalmente
    for _ in range(num_vertical):
        energy = compute_energy_matrix(result, energy_method)
        seam = find_vertical_seam(result, energy)
        
        # Duplicar el seam en lugar de eliminarlo
        rows, cols = result.shape[:2]
        new_img = np.zeros((rows, cols + 1, result.shape[2]), dtype=result.dtype)
        
        for row in range(rows):
            col_to_duplicate = int(seam[row])
            if 0 <= col_to_duplicate < cols:
                # Copiar píxeles antes del seam
                new_img[row, :col_to_duplicate] = result[row, :col_to_duplicate]
                # Duplicar el píxel del seam
                new_img[row, col_to_duplicate] = result[row, col_to_duplicate]
                new_img[row, col_to_duplicate + 1] = result[row, col_to_duplicate]
                # Copiar píxeles después del seam
                if col_to_duplicate < cols - 1:
                    new_img[row, col_to_duplicate + 2:] = result[row, col_to_duplicate + 1:]
        
        result = new_img
    
    # Ampliar horizontalmente
    for _ in range(num_horizontal):
        energy = compute_energy_matrix(result, energy_method)
        seam = find_horizontal_seam(result, energy)
        
        # Duplicar el seam horizontal
        rows, cols = result.shape[:2]
        new_img = np.zeros((rows + 1, cols, result.shape[2]), dtype=result.dtype)
        
        for col in range(cols):
            row_to_duplicate = int(seam[col])
            if 0 <= row_to_duplicate < rows:
                # Copiar píxeles antes del seam
                new_img[:row_to_duplicate, col] = result[:row_to_duplicate, col]
                # Duplicar el píxel del seam
                new_img[row_to_duplicate, col] = result[row_to_duplicate, col]
                new_img[row_to_duplicate + 1, col] = result[row_to_duplicate, col]
                # Copiar píxeles después del seam
                if row_to_duplicate < rows - 1:
                    new_img[row_to_duplicate + 2:, col] = result[row_to_duplicate + 1:, col]
        
        result = new_img
    
    return result

# Sidebar - Controles
st.sidebar.header("⚙️ Controles")

# Upload de imagen
uploaded_file = st.sidebar.file_uploader(
    "📁 Sube tu imagen", 
    type=['png', 'jpg', 'jpeg'],
    help="Formatos soportados: PNG, JPG, JPEG"
)

# Cargar imagen
if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    image = pil_to_cv2(pil_image)
    image_source = f"📁 {uploaded_file.name}"
else:
    image, sample_name = load_sample_image()
    image_source = f"🖼️ {sample_name}"

# Información de la imagen
h, w = image.shape[:2]
st.sidebar.info(f"**Imagen actual:** {image_source}")
st.sidebar.info(f"**Dimensiones:** {w} x {h} píxeles")

st.sidebar.markdown("---")

# Modo de operación
st.sidebar.subheader("🎯 Modo de Seam Carving")

operation_mode = st.sidebar.selectbox(
    "Operación:",
    options=['reduce', 'enlarge', 'energy_analysis', 'step_by_step'],
    format_func=lambda x: {
        'reduce': '📉 Reducir Imagen',
        'enlarge': '📈 Ampliar Imagen',
        'energy_analysis': '⚡ Análisis de Energía',
        'step_by_step': '👣 Paso a Paso'
    }[x],
    help="Selecciona el tipo de operación a realizar"
)

st.sidebar.markdown("---")

# Parámetros según el modo
if operation_mode in ['reduce', 'enlarge']:
    st.sidebar.subheader("📐 Parámetros de Redimensionamiento")
    
    num_vertical = st.sidebar.slider(
        "Seams verticales:", 
        0, min(200, w//2), 20, 1,
        help="Número de seams verticales a procesar"
    )
    
    num_horizontal = st.sidebar.slider(
        "Seams horizontales:", 
        0, min(200, h//2), 10, 1,
        help="Número de seams horizontales a procesar"
    )
    
elif operation_mode == 'step_by_step':
    st.sidebar.subheader("👣 Visualización Paso a Paso")
    
    step_direction = st.sidebar.selectbox(
        "Dirección:",
        options=['vertical', 'horizontal'],
        format_func=lambda x: '⬇️ Vertical' if x == 'vertical' else '➡️ Horizontal'
    )
    
    num_steps = st.sidebar.slider(
        "Número de pasos:", 
        1, 10, 3, 1,
        help="Pasos a mostrar en la visualización"
    )

# Método de cálculo de energía
st.sidebar.subheader("⚡ Método de Energía")

energy_method = st.sidebar.selectbox(
    "Algoritmo de energía:",
    options=['gradient', 'sobel', 'laplacian', 'scharr'],
    format_func=lambda x: {
        'gradient': '📈 Gradiente (Original)',
        'sobel': '🔍 Sobel',
        'laplacian': '🌊 Laplaciano',
        'scharr': '⚡ Scharr'
    }[x],
    help="Método para calcular la energía de la imagen"
)

# Opciones de visualización
st.sidebar.markdown("---")
st.sidebar.subheader("👁️ Opciones de Vista")

show_comparison = st.sidebar.checkbox("Comparación Lado a Lado", True)
show_seams = st.sidebar.checkbox("Mostrar Seams", True)
show_energy = st.sidebar.checkbox("Mostrar Mapa de Energía", False)

# Procesamiento según el modo
if operation_mode == 'reduce':
    # Reducción de imagen
    result, seams_overlay, _ = seam_carving_reduce(
        image, num_vertical, num_horizontal, energy_method
    )
    
    operation_title = f"📉 Reducción: -{num_vertical}W x -{num_horizontal}H"
    
elif operation_mode == 'enlarge':
    # Ampliación de imagen
    result = seam_carving_enlarge(
        image, num_vertical, num_horizontal, energy_method
    )
    seams_overlay = image  # No hay seams para mostrar en ampliación
    
    operation_title = f"📈 Ampliación: +{num_vertical}W x +{num_horizontal}H"
    
elif operation_mode == 'energy_analysis':
    # Análisis de energía
    energy_map = compute_energy_matrix(image, energy_method)
    
    # Normalizar para visualización
    energy_normalized = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX)
    energy_colored = cv2.applyColorMap(energy_normalized, cv2.COLORMAP_JET)
    
    result = energy_colored
    seams_overlay = image
    
    operation_title = f"⚡ Análisis de Energía - Método {energy_method.title()}"
    
elif operation_mode == 'step_by_step':
    # Visualización paso a paso
    step_images = [np.copy(image)]
    current_img = np.copy(image)
    
    for step in range(num_steps):
        energy = compute_energy_matrix(current_img, energy_method)
        
        if step_direction == 'vertical':
            seam = find_vertical_seam(current_img, energy)
            step_overlay = overlay_vertical_seam(current_img, seam)
            current_img = remove_vertical_seam(current_img, seam)
        else:
            seam = find_horizontal_seam(current_img, energy)
            step_overlay = overlay_horizontal_seam(current_img, seam)
            current_img = remove_horizontal_seam(current_img, seam)
        
        step_images.append(step_overlay)
        step_images.append(np.copy(current_img))
    
    result = current_img
    seams_overlay = image
    
    operation_title = f"👣 Paso a Paso - {num_steps} seams {step_direction}es"

# Área principal de visualización
if operation_mode == 'step_by_step':
    # Visualización especial para paso a paso
    st.subheader(operation_title)
    
    # Mostrar progresión en grid
    cols_per_row = 3
    for i in range(0, len(step_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(step_images):
                with cols[j]:
                    step_num = (i + j) // 2
                    if (i + j) % 2 == 0:
                        if i + j == 0:
                            st.write("**Original**")
                        else:
                            st.write(f"**Resultado Paso {step_num}**")
                    else:
                        st.write(f"**Seam Paso {step_num + 1}**")
                    
                    st.image(cv2_to_pil(step_images[i + j]), use_container_width=True)

elif operation_mode == 'energy_analysis':
    # Visualización del análisis de energía
    if show_comparison:
        st.subheader(operation_title + " - Comparación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🖼️ Imagen Original**")
            st.image(cv2_to_pil(image), use_container_width=True)
        
        with col2:
            st.write(f"**⚡ Mapa de Energía ({energy_method.title()})**")
            st.image(cv2_to_pil(result), use_container_width=True)
    else:
        st.subheader(f"⚡ Mapa de Energía - {energy_method.title()}")
        st.image(cv2_to_pil(result), use_container_width=True)
    
    # Estadísticas de energía
    st.subheader("📊 Estadísticas de Energía")
    energy_map = compute_energy_matrix(image, energy_method)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Energía Media", f"{np.mean(energy_map):.1f}")
    with col2:
        st.metric("Energía Máxima", f"{np.max(energy_map):.1f}")
    with col3:
        st.metric("Desviación Std", f"{np.std(energy_map):.1f}")
    with col4:
        st.metric("Rango Energía", f"{np.ptp(energy_map):.1f}")

else:
    # Visualización estándar para reducir/ampliar
    if show_comparison:
        st.subheader(operation_title + " - Comparación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🖼️ Imagen Original**")
            st.image(cv2_to_pil(image), use_container_width=True)
        
        with col2:
            st.write(f"**{operation_title}**")
            st.image(cv2_to_pil(result), use_container_width=True)
            
        # Mostrar seams si está habilitado
        if show_seams and operation_mode == 'reduce':
            st.subheader("✂️ Seams Detectados")
            st.image(cv2_to_pil(seams_overlay), use_container_width=True)
            st.caption("Verde: Seams verticales, Rojo: Seams horizontales")
            
    else:
        st.subheader(operation_title)
        
        if show_seams and operation_mode == 'reduce':
            tab1, tab2 = st.tabs(["Resultado", "Seams"])
            
            with tab1:
                st.image(cv2_to_pil(result), use_container_width=True)
            
            with tab2:
                st.image(cv2_to_pil(seams_overlay), use_container_width=True)
                st.caption("Verde: Seams verticales, Rojo: Seams horizontales")
        else:
            st.image(cv2_to_pil(result), use_container_width=True)
    
    # Mostrar mapa de energía si está habilitado
    if show_energy and operation_mode != 'energy_analysis':
        st.subheader("⚡ Mapa de Energía")
        energy_map = compute_energy_matrix(image, energy_method)
        energy_normalized = cv2.normalize(energy_map, None, 0, 255, cv2.NORM_MINMAX)
        energy_colored = cv2.applyColorMap(energy_normalized, cv2.COLORMAP_JET)
        st.image(cv2_to_pil(energy_colored), use_container_width=True)
    
    # Estadísticas de redimensionamiento
    if operation_mode in ['reduce', 'enlarge']:
        st.subheader("📊 Estadísticas de Redimensionamiento")
        
        original_size = image.shape[:2]
        new_size = result.shape[:2]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tamaño Original", f"{original_size[1]}×{original_size[0]}")
        
        with col2:
            st.metric("Tamaño Final", f"{new_size[1]}×{new_size[0]}")
        
        with col3:
            pixels_original = original_size[0] * original_size[1]
            pixels_new = new_size[0] * new_size[1]
            reduction = (1 - pixels_new / pixels_original) * 100
            st.metric("Reducción de Píxeles", f"{reduction:.1f}%")
        
        with col4:
            aspect_original = original_size[1] / original_size[0]
            aspect_new = new_size[1] / new_size[0]
            aspect_change = ((aspect_new - aspect_original) / aspect_original) * 100
            st.metric("Cambio Aspect Ratio", f"{aspect_change:+.1f}%")

# Botón de descarga
if operation_mode != 'step_by_step':
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        processed_pil = cv2_to_pil(result)
        buf = io.BytesIO()
        processed_pil.save(buf, format='PNG')
        
        st.download_button(
            label="📥 Descargar Imagen Procesada",
            data=buf.getvalue(),
            file_name=f"seam_carving_{operation_mode}.png",
            mime="image/png",
            use_container_width=True
        )

# Información educativa
st.sidebar.markdown("---")
st.sidebar.subheader("📚 Información Educativa")

with st.sidebar.expander("✂️ Sobre Seam Carving"):
    st.markdown("""
    **¿Qué es Seam Carving?**
    - Técnica de redimensionamiento content-aware
    - Preserva elementos importantes de la imagen
    - Elimina/duplica caminos de menor energía
    
    **Ventajas:**
    - Mantiene objetos importantes intactos
    - Mejor que escalado simple
    - Preserva características visuales clave
    
    **Aplicaciones:**
    - Redimensionamiento inteligente web
    - Adaptación a diferentes pantallas
    - Remoción de objetos no deseados
    - Retargeting de imágenes
    """)

with st.sidebar.expander("⚡ Métodos de Energía"):
    st.markdown("""
    **🔍 Gradiente (Original):**
    - Combina derivadas X e Y con Sobel
    - Peso igual (0.5) para ambas direcciones
    - Bueno para propósito general
    
    **🔍 Sobel:**
    - Magnitud del gradiente Sobel
    - Más preciso que gradiente simple
    - Mejor detección de bordes
    
    **🌊 Laplaciano:**
    - Detecta cambios de intensidad
    - Sensible a ruido
    - Bueno para texturas finas
    
    **⚡ Scharr:**
    - Versión mejorada de Sobel
    - Mayor precisión en orientaciones
    - Menos sensible a ruido
    """)

with st.sidebar.expander("🎯 Algoritmo Paso a Paso"):
    st.markdown("""
    **1. Cálculo de Energía:**
    - Convertir a escala de grises
    - Aplicar operador de gradiente
    - Obtener mapa de energía
    
    **2. Encontrar Seam Óptimo:**
    - Programación dinámica
    - Buscar camino de menor energía
    - De arriba hacia abajo (vertical)
    
    **3. Eliminar/Duplicar Seam:**
    - Eliminar píxeles del seam (reducir)
    - Duplicar píxeles del seam (ampliar)
    - Mantener conectividad
    
    **4. Repetir:**
    - Recalcular energía
    - Encontrar siguiente seam
    - Continuar hasta objetivo
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>✂️ <strong>Seam Carving Interactivo</strong> | Capítulo 6 - Redimensionamiento Inteligente</p>
        <p><small>Explora técnicas avanzadas de redimensionamiento preservando contenido importante</small></p>
    </div>
    """, 
    unsafe_allow_html=True
)