"""
Aplicación Principal - OpenCV 3.0 Computer Vision with Python Cookbook
Hub central para navegar entre todos los capítulos del libro
"""

import streamlit as st
import sys
import os
from pathlib import Path
import importlib.util
import subprocess

# Configuración de la página
st.set_page_config(
    page_title="OpenCV 3.0 Cookbook - Hub Principal",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar el path para importar módulos
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Información de los capítulos
CHAPTERS = {
    "Capítulo 1": {
        "title": "Operaciones Básicas con Imágenes",
        "description": "Lectura, escritura, conversión y transformaciones afines de imágenes",
        "icon": "🖼️",
        "folder": "Chapter01",
        "features": ["Transformaciones Afines", "Rotación y Escalado", "Traslación", "Perspectiva"],
        "techniques": "Matrices de transformación, interpolación bilineal"
    },
    "Capítulo 2": {
        "title": "Mejoramiento y Filtrado de Imágenes", 
        "description": "Técnicas de sharpening, filtros y mejoramiento de imágenes",
        "icon": "✨",
        "folder": "Chapter02", 
        "features": ["Filtro Unsharp Mask", "Filtros de Realce", "Kernels Personalizados"],
        "techniques": "Convolución, kernels de sharpening, filtros pasa-altos"
    },
    "Capítulo 3": {
        "title": "Procesamiento Avanzado y Efectos Artísticos",
        "description": "Cartoonización, filtros bilaterales y efectos creativos",
        "icon": "🎨", 
        "folder": "Chapter03",
        "features": ["Cartoonización", "Filtro Bilateral", "Efectos Artísticos", "Detección de Bordes"],
        "techniques": "Filtrado bilateral, clustering K-Means, edge detection"
    },
    "Capítulo 4": {
        "title": "Detección de Características Faciales",
        "description": "Detección de rostros, ojos, nariz y aplicación de overlays",
        "icon": "👤",
        "folder": "Chapter04",
        "features": ["Detección Facial", "Detección de Ojos", "Overlays", "Haar Cascades"],
        "techniques": "Haar Cascades, detección multi-escala, ROI processing"
    },
    "Capítulo 5": {
        "title": "Detección de Características Locales",
        "description": "SIFT, SURF, ORB, Harris corners y detección de keypoints",
        "icon": "🔍",
        "folder": "Chapter05", 
        "features": ["Detector SIFT", "Detector SURF", "Detector ORB", "Harris Corners", "Good Features"],
        "techniques": "Descriptores locales, keypoints, feature matching"
    },
    "Capítulo 6": {
        "title": "Seam Carving y Redimensionado Inteligente",
        "description": "Algoritmos de seam carving para redimensionado content-aware",
        "icon": "✂️",
        "folder": "Chapter06",
        "features": ["Seam Carving", "Energy Functions", "Dynamic Programming", "Content-Aware Resize"],
        "techniques": "Programación dinámica, funciones de energía, seam removal"
    },
    "Capítulo 7": {
        "title": "Segmentación Watershed",
        "description": "Algoritmo Watershed para segmentación de imágenes", 
        "icon": "💧",
        "folder": "Chapter07",
        "features": ["Algoritmo Watershed", "Segmentación", "Marcadores", "Morfología"],
        "techniques": "Watershed, operaciones morfológicas, connected components"
    },
    "Capítulo 8": {
        "title": "Seguimiento y Detección de Movimiento",
        "description": "Tracking por color, detección de movimiento y análisis temporal",
        "icon": "🎯",
        "folder": "Chapter08",
        "features": ["Color Tracking", "Motion Detection", "Background Subtraction", "HSV Analysis"],
        "techniques": "Espacio HSV, sustracción de fondo, análisis temporal"
    },
    "Capítulo 9": {
        "title": "Machine Learning y Bag of Words",
        "description": "Extracción de características, BoVW y clasificación automática",
        "icon": "🧠",
        "folder": "Chapter09",
        "features": ["SIFT Features", "Bag of Visual Words", "K-Means Clustering", "SVM Classification"],
        "techniques": "BoVW, cuantización vectorial, machine learning"
    },
    "Capítulo 10": {
        "title": "Estimación de Pose y Realidad Aumentada",
        "description": "Pose estimation, homografías y aplicaciones de AR",
        "icon": "🎯", 
        "folder": "Chapter10",
        "features": ["Pose Estimation", "Homografía", "Realidad Aumentada", "Feature Matching"],
        "techniques": "Homografías, RANSAC, proyección 3D, AR rendering"
    },
    "Capítulo 11": {
        "title": "Machine Learning Avanzado",
        "description": "Feature engineering avanzado y clasificación multiclase",
        "icon": "🔬",
        "folder": "Chapter11", 
        "features": ["Feature Engineering", "Multiclass Classification", "PCA Analysis", "Model Comparison"],
        "techniques": "Feature selection, PCA, múltiples clasificadores, evaluation"
    }
}

def load_chapter_app(chapter_folder):
    """Carga y ejecuta la aplicación de un capítulo específico"""
    try:
        # Path al archivo streamlit_app.py del capítulo
        chapter_path = current_dir / chapter_folder / "streamlit_app.py"
        
        if not chapter_path.exists():
            st.error(f"❌ No se encontró la aplicación en {chapter_path}")
            return False
        
        # Cargar el módulo dinámicamente
        spec = importlib.util.spec_from_file_location("chapter_app", chapter_path)
        chapter_module = importlib.util.module_from_spec(spec)
        
        # Añadir el directorio del capítulo al path para imports relativos
        chapter_dir = str(chapter_path.parent)
        if chapter_dir not in sys.path:
            sys.path.insert(0, chapter_dir)
        
        # Ejecutar el módulo
        spec.loader.exec_module(chapter_module)
        
        return True
        
    except ImportError as e:
        st.error(f"❌ Error de importación en {chapter_folder}: {str(e)}")
        st.error("💡 El capítulo tiene dependencias faltantes, pero debería funcionar con fallbacks")
        
        # Información adicional para debugging
        st.error(f"📍 Archivo: {chapter_path}")
        st.error(f"📍 Error específico: {type(e).__name__}")
        
        return False
        
    except Exception as e:
        st.error(f"❌ Error cargando la aplicación del capítulo: {str(e)}")
        st.error("💡 Asegúrate de que el archivo streamlit_app.py existe en la carpeta del capítulo")
        
        # Información adicional para debugging
        st.error(f"📍 Archivo: {chapter_path}")
        st.error(f"📍 Error específico: {type(e).__name__}")
        
        return False

def show_chapter_overview(chapter_key, chapter_info):
    """Muestra la información general de un capítulo"""
    
    st.header(f"{chapter_info['icon']} {chapter_key}: {chapter_info['title']}")
    
    # Descripción
    st.markdown(f"**📝 Descripción:** {chapter_info['description']}")
    
    # Columnas para información
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Características Principales")
        for feature in chapter_info['features']:
            st.write(f"• {feature}")
    
    with col2:
        st.subheader("🔧 Técnicas Implementadas") 
        st.write(chapter_info['techniques'])
    
    # Botón para ejecutar el capítulo
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"🚀 Ejecutar {chapter_key}", use_container_width=True, type="primary"):
            st.session_state.selected_chapter = chapter_key
            st.rerun()
    
    # Información adicional
    st.markdown("---")
    st.info(f"📁 **Ubicación:** `{chapter_info['folder']}/streamlit_app.py`")

def show_main_menu():
    """Muestra el menú principal con todos los capítulos"""
    
    # Header principal
    st.title("📚 OpenCV 3.0 Computer Vision with Python Cookbook")
    st.markdown("### 🎓 Hub Interactivo de Aprendizaje")
    
    st.markdown("""
    Bienvenido al hub principal del **OpenCV 3.0 Computer Vision Cookbook**. 
    Esta aplicación interactiva te permite explorar todos los capítulos del libro 
    a través de aplicaciones Streamlit dedicadas.
    """)
    
    # Estadísticas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📖 Capítulos", len(CHAPTERS))
    
    with col2:
        total_features = sum(len(info['features']) for info in CHAPTERS.values())
        st.metric("🎯 Características", total_features)
    
    with col3:
        st.metric("🔬 Técnicas CV", "50+")
    
    with col4:
        st.metric("📊 Aplicaciones", "11")
    
    st.markdown("---")
    
    # Selector de capítulo en sidebar
    st.sidebar.header("🧭 Navegación")
    
    # Opciones del selector
    chapter_options = ["📋 Menú Principal"] + list(CHAPTERS.keys())
    
    selected_option = st.sidebar.selectbox(
        "Selecciona un capítulo:",
        options=chapter_options,
        index=0
    )
    
    # Información rápida en sidebar
    if selected_option != "📋 Menú Principal":
        chapter_info = CHAPTERS[selected_option]
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**{chapter_info['icon']} {selected_option}**")
        st.sidebar.markdown(f"*{chapter_info['description']}*")
        
        if st.sidebar.button("🚀 Ejecutar Capítulo", use_container_width=True):
            st.session_state.selected_chapter = selected_option
            st.rerun()
    
    # Mostrar capítulos en grid
    st.subheader("📚 Capítulos Disponibles")
    
    # Organizar en filas de 2 columnas
    chapters_list = list(CHAPTERS.items())
    
    for i in range(0, len(chapters_list), 2):
        col1, col2 = st.columns(2)
        
        # Primer capítulo de la fila
        with col1:
            if i < len(chapters_list):
                chapter_key, chapter_info = chapters_list[i]
                
                with st.container():
                    st.markdown(f"""
                    <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; margin: 10px 0;">
                        <h3>{chapter_info['icon']} {chapter_key}</h3>
                        <h4>{chapter_info['title']}</h4>
                        <p>{chapter_info['description']}</p>
                        <p><strong>Técnicas:</strong> {chapter_info['techniques']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Abrir {chapter_key}", key=f"btn_{i}", use_container_width=True):
                        st.session_state.selected_chapter = chapter_key
                        st.rerun()
        
        # Segundo capítulo de la fila
        with col2:
            if i + 1 < len(chapters_list):
                chapter_key, chapter_info = chapters_list[i + 1]
                
                with st.container():
                    st.markdown(f"""
                    <div style="border: 2px solid #e0e0e0; border-radius: 10px; padding: 20px; margin: 10px 0;">
                        <h3>{chapter_info['icon']} {chapter_key}</h3>
                        <h4>{chapter_info['title']}</h4>
                        <p>{chapter_info['description']}</p>
                        <p><strong>Técnicas:</strong> {chapter_info['techniques']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Abrir {chapter_key}", key=f"btn_{i+1}", use_container_width=True):
                        st.session_state.selected_chapter = chapter_key
                        st.rerun()
    
    # Información adicional
    st.markdown("---")
    st.subheader("ℹ️ Información del Proyecto")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📖 Sobre el Libro:**
        - OpenCV 3.0 Computer Vision with Python Cookbook
        - 11 capítulos de técnicas de visión por computador
        - Desde conceptos básicos hasta ML avanzado
        - Implementaciones prácticas con OpenCV y Python
        """)
    
    with col2:
        st.markdown("""
        **🛠️ Tecnologías Utilizadas:**
        - **OpenCV**: Biblioteca principal de visión por computador
        - **Streamlit**: Framework para aplicaciones web interactivas
        - **NumPy**: Computación numérica y manejo de arrays
        - **Matplotlib/Plotly**: Visualización de datos y resultados
        - **scikit-learn**: Algoritmos de machine learning
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>📚 <strong>OpenCV 3.0 Computer Vision Cookbook</strong> - Hub Interactivo</p>
            <p><small>Explora técnicas de visión por computador de forma interactiva</small></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def main():
    """Función principal de la aplicación"""
    
    # Inicializar estado de sesión
    if 'selected_chapter' not in st.session_state:
        st.session_state.selected_chapter = None
    
    # Verificar si se debe mostrar un capítulo específico
    if st.session_state.selected_chapter and st.session_state.selected_chapter in CHAPTERS:
        
        chapter_key = st.session_state.selected_chapter
        chapter_info = CHAPTERS[chapter_key]
        
        # Header del capítulo
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {chapter_info['icon']} {chapter_key}")
        st.sidebar.markdown(f"*{chapter_info['title']}*")
        
        # Botón para volver al menú principal
        if st.sidebar.button("🏠 Volver al Menú Principal", use_container_width=True):
            st.session_state.selected_chapter = None
            st.rerun()
        
        st.sidebar.markdown("---")
        
        # Cargar y ejecutar la aplicación del capítulo
        success = load_chapter_app(chapter_info['folder'])
        
        if not success:
            st.error("❌ No se pudo cargar la aplicación del capítulo")
            if st.button("🏠 Volver al Menú Principal"):
                st.session_state.selected_chapter = None
                st.rerun()
    
    else:
        # Mostrar menú principal
        show_main_menu()

if __name__ == "__main__":
    main()