# ROOP GPU Batch Processing - Optimizado para Google Colab

Este script optimizado permite procesar múltiples videos con face swap usando GPU en Google Colab, sin necesidad de entorno virtual.

## 🚀 Características

- ✅ **GPU Optimizado**: Usa CUDA directamente sin entorno virtual
- ✅ **Procesamiento por Carpetas**: Procesa todos los videos de una carpeta automáticamente
- ✅ **Gestión de Memoria**: Libera memoria GPU entre videos
- ✅ **Face Swapper + Face Enhancer**: Ambos procesadores incluidos
- ✅ **Compatible con T4**: Optimizado para Google Colab T4 (15GB VRAM)

## 📁 Estructura de Carpetas

```
/content/
├── DanielaAS.jpg          # Imagen fuente
├── videos/                # Carpeta con videos a procesar
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── resultados/            # Carpeta donde se guardan los resultados
    ├── DanielaAS_video1.mp4
    ├── DanielaAS_video2.mp4
    └── ...
```

## 🎯 Uso en Google Colab

### 1. Instalación Completa (RECOMENDADA)

```python
# Ejecutar instalador completo
!python install_roop_colab.py
```

### 2. Instalación Rápida

```python
# Ejecutar instalación rápida
!python colab_quick_install.py
```

### 3. Instalación Manual

```python
# Instalar dependencias
!pip install onnxruntime-gpu tensorflow-gpu torch torchvision opencv-python pillow numpy scipy psutil tqdm insightface basicsr facexlib gfpgan realesrgan albumentations ffmpeg-python moviepy imageio imageio-ffmpeg

# Clonar ROOP
!git clone --branch v3 https://github.com/CDavidDv/roop.git
%cd roop

# Descargar modelo de face swap
!wget https://civitai.com/api/download/models/85159 -O inswapper_128.onnx

print("✅ Configuración completada")
```

### 2. Procesar Videos con GPU (Versión Simplificada - RECOMENDADA)

```python
# Ejecutar procesamiento optimizado para GPU
!python run_batch_gpu_simple.py \
  --source /content/DanielaAS.jpg \
  --input-folder /content/videos \
  --output-folder /content/resultados \
  --frame-processors face_swapper face_enhancer \
  --max-memory 12 \
  --execution-threads 8 \
  --temp-frame-quality 100 \
  --gpu-memory-wait 30 \
  --keep-fps
```

### 3. Procesar Videos con GPU (Versión Avanzada)

```python
# Ejecutar procesamiento optimizado para GPU
!python run_batch_gpu.py \
  --source /content/DanielaAS.jpg \
  --input-folder /content/videos \
  --output-folder /content/resultados \
  --frame-processors face_swapper face_enhancer \
  --max-memory 12 \
  --execution-threads 8 \
  --temp-frame-quality 100 \
  --gpu-memory-wait 30 \
  --keep-fps
```

## ⚙️ Parámetros Disponibles

| Parámetro | Descripción | Valor por Defecto |
|-----------|-------------|-------------------|
| `--source` | Imagen fuente | **Requerido** |
| `--input-folder` | Carpeta con videos | **Requerido** |
| `--output-folder` | Carpeta para resultados | **Requerido** |
| `--frame-processors` | Procesadores (face_swapper, face_enhancer) | `face_swapper face_enhancer` |
| `--max-memory` | Memoria máxima en GB | `12` |
| `--execution-threads` | Número de hilos | `8` |
| `--temp-frame-quality` | Calidad de frames (0-100) | `100` |
| `--gpu-memory-wait` | Tiempo de espera entre videos (segundos) | `30` |
| `--keep-fps` | Mantener FPS original | `True` |

## 🔧 Optimizaciones para T4

- **Memoria GPU**: Configurado para 12GB (dejando 3GB libres)
- **Threads**: 8 hilos optimizados para T4
- **Calidad**: 100 para mejor resultado
- **Espera GPU**: 30 segundos entre videos para liberar memoria

## 📊 Ejemplo Completo

```python
# 1. Subir tu imagen fuente a /content/DanielaAS.jpg
# 2. Crear carpeta /content/videos y subir tus videos
# 3. Ejecutar:

!git clone --branch v3 https://github.com/CDavidDv/roop.git
%cd roop
!wget https://civitai.com/api/download/models/85159 -O inswapper_128.onnx

!python run_batch_gpu.py \
  --source /content/DanielaAS.jpg \
  --input-folder /content/videos \
  --output-folder /content/resultados \
  --frame-processors face_swapper face_enhancer \
  --max-memory 12 \
  --execution-threads 8 \
  --temp-frame-quality 100 \
  --gpu-memory-wait 30 \
  --keep-fps

print("🎬 ¡Procesamiento con GPU completado!")
```

## 🎨 Formatos Soportados

### Videos de Entrada:
- `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv`, `.webm`

### Imágenes Fuente:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

## ⚠️ Notas Importantes

1. **GPU Memory**: El script libera memoria GPU automáticamente entre videos
2. **Tiempo de Procesamiento**: Depende de la duración y calidad de los videos
3. **Errores**: Si un video falla, continúa con el siguiente
4. **Nombres**: Los videos de salida usan el formato: `{source}_{video}.mp4`

## 🐛 Solución de Problemas

### Error: "No such file or directory"
- Verifica que las rutas de carpetas sean correctas
- Asegúrate de que los archivos existan

### Error: "CUDA out of memory"
- Reduce `--max-memory` a 10 o 8
- Aumenta `--gpu-memory-wait` a 45 o 60

### Error: "Model not found"
- Verifica que `inswapper_128.onnx` esté en el directorio raíz

## 📈 Rendimiento Esperado

- **T4 15GB VRAM**: ~2-5 minutos por minuto de video
- **Memoria RAM**: ~12GB utilizados
- **Almacenamiento**: ~500MB por minuto de video procesado

## 🔄 Diferencias entre Versiones

### Versión Simplificada (`run_batch_gpu_simple.py`) - RECOMENDADA
- ✅ **Más estable**: Usa subprocess para llamar a `run.py` directamente
- ✅ **Menos errores**: Evita problemas de importación y configuración
- ✅ **Más confiable**: Funciona igual que el comando original de ROOP
- ✅ **Fácil de debuggear**: Errores claros y específicos

### Versión Avanzada (`run_batch_gpu.py`)
- ✅ **Más eficiente**: Usa las funciones de ROOP directamente
- ⚠️ **Más compleja**: Puede tener problemas de configuración
- ⚠️ **Requiere más testing**: Necesita más validación

## 🔄 Diferencias con el Script Anterior

| Característica | Script Anterior | Nuevo Script |
|----------------|------------------|---------------|
| Entorno Virtual | ✅ Requerido | ❌ No necesario |
| GPU Directo | ❌ Subprocess | ✅ Directo |
| Carpetas | ❌ Archivos individuales | ✅ Procesamiento por carpetas |
| Memoria GPU | ❌ Sin gestión | ✅ Gestión automática |
| Compatibilidad T4 | ⚠️ Limitada | ✅ Optimizada |

¡Listo para procesar tus videos con GPU optimizado! 🚀 