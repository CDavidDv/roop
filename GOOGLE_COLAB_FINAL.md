# ROOP - Google Colab Optimizado

Instrucciones completas para ejecutar ROOP en Google Colab con las versiones más recientes.

## 🚀 Instalación Rápida

### Paso 1: Clonar y Configurar

```python
# Clonar el repositorio
!git clone https://github.com/CDavidDv/roop
%cd roop

print("✅ Repositorio clonado")
```

### Paso 2: Instalación Automática

```python
# Ejecutar instalación automática
!python install_roop_colab.py
```

### Paso 3: Verificar Instalación

```python
# Verificar que todo funciona
!python test_gpu_usage.py
```

## 🔧 Instalación Manual (Si es necesario)

### Paso 1: Limpiar Entorno

```python
# Desinstalar versiones conflictivas
!pip uninstall -y numpy torch torchvision torchaudio tensorflow onnxruntime onnxruntime-gpu
```

### Paso 2: Instalar Dependencias Actualizadas

```python
# Instalar NumPy 2.x
!pip install numpy==2.1.4

# Instalar PyTorch con CUDA 12.1
!pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Instalar TensorFlow 2.16.1
!pip install tensorflow==2.16.1

# Instalar ONNX Runtime GPU
!pip install onnxruntime-gpu==1.17.0

# Instalar otras dependencias
!pip install opencv-python==4.9.0.80 insightface==0.7.3 gfpgan==1.3.8 basicsr==1.4.2 facexlib==0.3.0 filterpy==1.4.5 opennsfw2==0.10.2 pillow==10.2.0 tqdm==4.66.1 psutil==5.9.8 coloredlogs==15.0.1 humanfriendly==10.0 sqlalchemy==2.0.31 addict==2.4.0 pydantic==2.8.0 pydantic-core==2.20.0 lmdb==1.5.1 typing-extensions==4.10.0
```

### Paso 3: Configurar Entorno

```python
# Configurar variables de entorno
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("✅ Entorno configurado")
```

### Paso 4: Descargar Modelo

```python
# Descargar modelo de face swap
!wget https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx

print("✅ Modelo descargado")
```

### Paso 5: Verificar GPU

```python
# Verificar que GPU funciona
import torch
import tensorflow as tf
import onnxruntime as ort

print(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

print(f"✅ TensorFlow GPUs: {len(tf.config.list_physical_devices('GPU'))}")
print(f"✅ ONNX Providers: {ort.get_available_providers()}")
```

## 🎯 Uso

### Procesamiento Individual

```python
# Procesar un video
!python run.py \
  --source /content/source.jpg \
  --target /content/video.mp4 \
  -o /content/resultado.mp4 \
  --frame-processor face_swapper face_enhancer \
  --execution-provider cuda \
  --execution-threads 31 \
  --temp-frame-quality 100 \
  --keep-fps
```

### Procesamiento en Lote

```python
# Procesar múltiples videos
!python run_batch_processing.py \
  --source /content/source.jpg \
  --videos /content/video1.mp4 /content/video2.mp4 /content/video3.mp4 \
  --output-dir /content/resultados \
  --execution-threads 31 \
  --temp-frame-quality 100 \
  --keep-fps
```

## 📊 Verificación de Rendimiento

### Verificar GPU

```python
# Verificar uso de GPU
import torch
import psutil

# Información del sistema
print(f"CPU: {psutil.cpu_count()} cores")
print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")

# Información de GPU
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    
    # Probar rendimiento
    import time
    device = torch.device('cuda')
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    start_time = time.time()
    z = torch.mm(x, y)
    torch.cuda.synchronize()
    end_time = time.time()
    
    print(f"Rendimiento GPU: {(end_time - start_time)*1000:.2f}ms")
```

### Verificar ONNX Runtime

```python
# Verificar proveedores ONNX
import onnxruntime as ort

providers = ort.get_available_providers()
print(f"Proveedores disponibles: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✅ CUDA disponible para ONNX Runtime")
else:
    print("❌ CUDA no disponible para ONNX Runtime")
```

## 🔧 Optimizaciones

### Configuración de Memoria

```python
# Configurar límites de memoria
import tensorflow as tf

# Configurar crecimiento de memoria
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Configurar límite de memoria (opcional)
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)
# ])

print("✅ Configuración de memoria aplicada")
```

### Configuración de PyTorch

```python
# Configurar PyTorch para mejor rendimiento
import torch

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("✅ Optimizaciones PyTorch aplicadas")
```

## 🐛 Solución de Problemas

### Error de Memoria GPU

```python
# Reducir memoria máxima
!python run.py \
  --source /content/source.jpg \
  --target /content/video.mp4 \
  -o /content/resultado.mp4 \
  --max-memory 4 \
  --execution-threads 16
```

### Error de CUDA

```python
# Usar solo CPU
!python run.py \
  --source /content/source.jpg \
  --target /content/video.mp4 \
  -o /content/resultado.mp4 \
  --execution-provider cpu
```

### Error de NumPy

```python
# Reinstalar NumPy
!pip uninstall numpy -y
!pip install numpy==2.1.4
```

## 📈 Comparación de Rendimiento

### Con GPU (Optimizado)
- **Velocidad**: 2-5x más rápido
- **Memoria**: Gestión automática
- **Calidad**: Mantiene calidad original

### Sin GPU (Fallback)
- **Velocidad**: Procesamiento CPU
- **Memoria**: Gestión eficiente de RAM
- **Calidad**: Misma calidad, más lento

## 🎉 Resultado Final

Con estas optimizaciones, deberías tener:

1. ✅ **NumPy 2.1.4** - Versión más reciente
2. ✅ **PyTorch 2.2.0** - Con soporte CUDA 12.1
3. ✅ **TensorFlow 2.16.1** - GPU optimizado
4. ✅ **ONNX Runtime 1.17.0** - GPU acceleration
5. ✅ **Todas las librerías actualizadas** - Compatibilidad total
6. ✅ **Optimización GPU completa** - Rendimiento máximo

## 📝 Notas Importantes

1. **NumPy 2.x**: Compatible con todas las librerías actualizadas
2. **GPU Memory**: Configuración automática para evitar OOM
3. **Batch Processing**: Optimizado para múltiples videos
4. **NSFW Skip**: Opción para saltar verificación en GPU

---

**¡Sistema completamente actualizado y optimizado para rendimiento máximo!** 