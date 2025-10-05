"""
Filtros de Sharpening - Capítulo 2
Aplicación de diferentes kernels para realzar detalles en imágenes
"""

import cv2 
import numpy as np 

def main():
    import os
    
    # Obtener la ruta del directorio del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construir la ruta completa a la imagen
    image_path = os.path.join(script_dir, 'images', 'house_input.png')
    
    # Cargar imagen de entrada
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen desde: {image_path}")
        print("Verificar que la imagen existe en la ruta especificada.")
        return
    
    # Mostrar imagen original
    cv2.imshow('Original', img) 
    
    # Definir los kernels de sharpening
    # Kernel 1: Sharpening básico
    kernel_sharpen_1 = np.array([[-1,-1,-1], 
                                 [-1, 9,-1], 
                                 [-1,-1,-1]]) 
    
    # Kernel 2: Sharpening intenso
    kernel_sharpen_2 = np.array([[1, 1, 1], 
                                 [1,-7, 1], 
                                 [1, 1, 1]]) 
    
    # Kernel 3: Realce de bordes (Edge Enhancement)
    kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1], 
                                 [-1, 2, 2, 2,-1], 
                                 [-1, 2, 8, 2,-1], 
                                 [-1, 2, 2, 2,-1], 
                                 [-1,-1,-1,-1,-1]]) / 8.0 
    
    # Aplicar los diferentes kernels a la imagen
    output_1 = cv2.filter2D(img, -1, kernel_sharpen_1) 
    output_2 = cv2.filter2D(img, -1, kernel_sharpen_2) 
    output_3 = cv2.filter2D(img, -1, kernel_sharpen_3) 
    
    # Mostrar resultados
    cv2.imshow('Sharpening Básico', output_1) 
    cv2.imshow('Sharpening Intenso', output_2) 
    cv2.imshow('Realce de Bordes', output_3) 
    
    print("Presiona cualquier tecla para cerrar las ventanas...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 