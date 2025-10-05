# Filtros de Sharpening - Procesamiento de Imágenes

Este proyecto demuestra la aplicación de diferentes tipos de filtros de sharpening (realce de nitidez) en imágenes usando OpenCV y Python.

## 📋 Descripción

El código implementa tres tipos diferentes de kernels de convolución para realzar detalles en imágenes:

1. **Sharpening Básico**: Realza detalles de forma suave
2. **Sharpening Intenso**: Efecto más dramático, puede introducir artefactos
3. **Realce de Bordes**: Enfatiza contornos y transiciones

## 🚀 Instalación

1. Clona este repositorio:

```bash
git clone <url-del-repositorio>
cd <nombre-del-repositorio>
```

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## 💻 Uso

Ejecuta el script principal:

```bash
cd Chapter02
python 03_sharpening.py
```

El programa:

- Carga una imagen de entrada desde la carpeta `images/`
- Aplica los tres tipos de filtros de sharpening
- Muestra las imágenes originales y procesadas en ventanas separadas
- Presiona cualquier tecla para cerrar las ventanas

## 📁 Estructura del Proyecto

```
Capitulo 2/
├── requirements.txt
├── SHARPENING_README.md
└── Chapter02/
    ├── 03_sharpening.py
    └── images/
        ├── house_input.png
        ├── geometrics_input.png
        └── ... (otras imágenes de prueba)
```

## 🔧 Dependencias

- `opencv-python`: Para procesamiento de imágenes
- `numpy`: Para operaciones con arrays y matrices

## 📖 Conceptos Implementados

### Kernels de Convolución

**Sharpening Básico:**

```
[[-1, -1, -1],
 [-1,  9, -1],
 [-1, -1, -1]]
```

**Sharpening Intenso:**

```
[[1,  1,  1],
 [1, -7,  1],
 [1,  1,  1]]
```

**Realce de Bordes:**

```
[[-1, -1, -1, -1, -1],
 [-1,  2,  2,  2, -1],
 [-1,  2,  8,  2, -1],
 [-1,  2,  2,  2, -1],
 [-1, -1, -1, -1, -1]] / 8.0
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👥 Autor

- **Daniel** - _Trabajo inicial_
