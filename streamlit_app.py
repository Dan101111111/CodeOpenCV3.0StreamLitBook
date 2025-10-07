"""
Aplicación Principal - OpenCV 3.0 Computer Vision Cookbook
Menú principal para navegar entre capítulos
"""

import streamlit as st
import sys
from pathlib import Path
import importlib.util

# Configuración de la página
st.set_page_config(
    page_title="OpenCV 3.0 Cookbook - Menú Principal",
    page_icon="📚",
    layout="centered"
)

# Configurar el path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Información básica de los capítulos
CHAPTERS = {
    "Capítulo 1": {"title": "Transformaciones Afines", "icon": "�", "folder": "Chapter01"},
    "Capítulo 2": {"title": "Filtros y Sharpening", "icon": "✨", "folder": "Chapter02"},
    "Capítulo 3": {"title": "Efectos Artísticos", "icon": "🎨", "folder": "Chapter03"},
    "Capítulo 4": {"title": "Detección Facial", "icon": "👤", "folder": "Chapter04"},
    "Capítulo 5": {"title": "Detección SIFT", "icon": "🔍", "folder": "Chapter05"},
    "Capítulo 6": {"title": "Seam Carving", "icon": "✂️", "folder": "Chapter06"},
    "Capítulo 7": {"title": "Segmentación Watershed", "icon": "💧", "folder": "Chapter07"},
    "Capítulo 8": {"title": "Tracking por Color", "icon": "🎯", "folder": "Chapter08"},
    "Capítulo 9": {"title": "Bag of Words", "icon": "🧠", "folder": "Chapter09"},
    "Capítulo 10": {"title": "Estimación de Pose", "icon": "📍", "folder": "Chapter10"},
    "Capítulo 11": {"title": "Extracción de Características", "icon": "🔬", "folder": "Chapter11"}
}

def load_chapter_app(chapter_folder):
    """Carga y ejecuta la aplicación de un capítulo específico"""
    try:
        chapter_path = current_dir / chapter_folder / "streamlit_app.py"
        
        if not chapter_path.exists():
            st.error(f"❌ No se encontró: {chapter_path}")
            return False
        
        spec = importlib.util.spec_from_file_location("chapter_app", chapter_path)
        if spec and spec.loader:
            chapter_module = importlib.util.module_from_spec(spec)
            
            # Añadir directorio al path
            chapter_dir = str(chapter_path.parent)
            if chapter_dir not in sys.path:
                sys.path.insert(0, chapter_dir)
            
            spec.loader.exec_module(chapter_module)
            return True
        
        return False
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return False

def show_main_menu():
    """Menú principal simplificado"""
    st.title("📚 OpenCV 3.0 Computer Vision Cookbook")
    st.markdown("**Selecciona un capítulo para explorar:**")
    
    # Grid de capítulos en 2 columnas
    chapters_list = list(CHAPTERS.items())
    
    for i in range(0, len(chapters_list), 2):
        col1, col2 = st.columns(2)
        
        # Primer capítulo
        with col1:
            if i < len(chapters_list):
                chapter_key, chapter_info = chapters_list[i]
                st.markdown(f"### {chapter_info['icon']} {chapter_key}")
                st.markdown(f"**{chapter_info['title']}**")
                if st.button(f"Abrir {chapter_key}", key=f"btn_{i}", use_container_width=True):
                    st.session_state.selected_chapter = chapter_key
                    st.rerun()
        
        # Segundo capítulo
        with col2:
            if i + 1 < len(chapters_list):
                chapter_key, chapter_info = chapters_list[i + 1]
                st.markdown(f"### {chapter_info['icon']} {chapter_key}")
                st.markdown(f"**{chapter_info['title']}**")
                if st.button(f"Abrir {chapter_key}", key=f"btn_{i+1}", use_container_width=True):
                    st.session_state.selected_chapter = chapter_key
                    st.rerun()

def main():
    """Función principal"""
    if 'selected_chapter' not in st.session_state:
        st.session_state.selected_chapter = None
    
    # Si hay un capítulo seleccionado
    if st.session_state.selected_chapter and st.session_state.selected_chapter in CHAPTERS:
        chapter_key = st.session_state.selected_chapter
        chapter_info = CHAPTERS[chapter_key]
        
        # Sidebar con información del capítulo
        st.sidebar.markdown(f"### {chapter_info['icon']} {chapter_key}")
        st.sidebar.markdown(f"*{chapter_info['title']}*")
        
        if st.sidebar.button("🏠 Volver al Menú", use_container_width=True):
            st.session_state.selected_chapter = None
            st.rerun()
        
        # Cargar aplicación del capítulo
        success = load_chapter_app(chapter_info['folder'])
        
        if not success:
            st.error("❌ Error cargando capítulo")
            if st.button("🏠 Volver al Menú"):
                st.session_state.selected_chapter = None
                st.rerun()
    else:
        # Mostrar menú principal
        show_main_menu()

if __name__ == "__main__":
    main()