# ROOP - Face Swap Optimizado

Sistema de face swap optimizado para GPU con las versiones más recientes de todas las dependencias.

## 🚀 Características

- **Versiones actualizadas**: NumPy 2.x, PyTorch 2.2.0, TensorFlow 2.16.1, ONNX Runtime 1.17.0
- **Optimización GPU**: Soporte completo para CUDA con configuración optimizada
- **Procesamiento en lote**: Script optimizado para procesar múltiples videos
- **Compatibilidad**: Python 3.10+ con todas las librerías actualizadas

## 📋 Requisitos

- Python 3.10 o superior
- CUDA 12.1+ (para GPU)
- 8GB+ RAM (recomendado)
- FFmpeg instalado

## 🛠️ Instalación

### Instalación Automática (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/CDavidDv/roop
cd roop

# Instalación automática con versiones actualizadas
python install_roop_colab.py
```

### Instalación Manual

```bash
# Instalar dependencias
pip install -r requirements.txt

# Para entornos headless (sin GUI)
pip install -r requirements-headless.txt
```

## 🎯 Uso

### Procesamiento Individual

```bash
python run_batch_processing.py \
  --source /content/source.jpg \
  --videos /content/video1.mp4 /content/video2.mp4 \
  --output-dir /content/resultados \
  --execution-threads 31 \
  --temp-frame-quality 100 \
  --keep-fps
```

### Procesamiento con GPU Optimizado

```bash
python run_roop_gpu.py \
  --source imagen.jpg \
  --target video.mp4 \
  -o resultado.mp4 \
  --frame-processor face_swapper face_enhancer \
  --execution-provider cuda \
  --execution-threads 31 \
  --temp-frame-quality 100 \
  --keep-fps
```

## ⚙️ Parámetros de Optimización

- `--execution-threads 31`: Número de hilos de procesamiento
- `--temp-frame-quality 100`: Calidad de frames temporales (0-100)
- `--keep-fps`: Mantener FPS original del video
- `--max-memory 8`: Límite de memoria en GB
- `--gpu-memory-wait 30`: Tiempo de espera entre procesadores (segundos)

## 🔧 Optimizaciones GPU

### Configuración Automática
- Detección automática de GPU
- Configuración optimizada de memoria CUDA
- Priorización de proveedores GPU sobre CPU
- Gestión automática de memoria del sistema

### Variables de Entorno
```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
```

## 📊 Versiones Actualizadas

| Librería | Versión | Optimización |
|----------|---------|--------------|
| NumPy | 2.1.4 | Compatibilidad mejorada |
| PyTorch | 2.2.0+cu121 | Soporte CUDA 12.1 |
| TensorFlow | 2.16.1 | GPU optimizado |
| ONNX Runtime | 1.17.0 | GPU acceleration |
| OpenCV | 4.9.0.80 | Rendimiento mejorado |
| InsightFace | 0.7.3 | Face detection optimizado |

## 🚀 Rendimiento

### Con GPU
- **Velocidad**: 2-5x más rápido que CPU
- **Memoria**: Gestión automática de memoria GPU
- **Calidad**: Mantiene calidad original con optimizaciones

### Sin GPU
- **Fallback**: Procesamiento CPU optimizado
- **Hilos**: Configuración automática de hilos
- **Memoria**: Gestión eficiente de RAM

## 🔍 Verificación de Instalación

```python
import torch
import tensorflow as tf
import onnxruntime as ort

# Verificar GPU
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU'}")

# Verificar TensorFlow
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs TensorFlow: {len(gpus)}")

# Verificar ONNX
providers = ort.get_available_providers()
print(f"Proveedores ONNX: {providers}")
```

## 📝 Notas Importantes

1. **NumPy 2.x**: Compatible con todas las librerías actualizadas
2. **GPU Memory**: Configuración automática para evitar OOM
3. **Batch Processing**: Optimizado para procesar múltiples videos
4. **NSFW Skip**: Opción para saltar verificación NSFW en GPU

## 🐛 Solución de Problemas

### Error de Memoria GPU
```bash
# Reducir memoria máxima
--max-memory 4
```

### Error de CUDA
```bash
# Usar solo CPU
--execution-provider cpu
```

### Error de NumPy
```bash
# Reinstalar NumPy
pip uninstall numpy -y
pip install numpy==2.1.4
```

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue o pull request.

---

**Optimizado para rendimiento máximo con las versiones más recientes de todas las dependencias.**
