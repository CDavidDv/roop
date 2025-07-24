# 🚀 Roop GPU Optimizado - Google Colab

## Instalación Rápida

### Celda 1: Clonar y Configurar
```python
!git clone https://github.com/CDavidDv/roop
%cd roop
!pip uninstall numpy -y
!pip install numpy==1.26.4 --no-cache-dir --force-reinstall
print("✅ Configuración inicial completada")
```

### Celda 2: Instalar Dependencias
```python
!pip install -r requirements.txt
!pip install opencv-python insightface gfpgan basicsr facexlib sympy onnx
print("✅ Dependencias instaladas")
```

### Celda 3: Instalar GPU
```python
!pip uninstall tensorflow tensorflow-gpu -y
!pip install tensorflow==2.15.0
!pip uninstall onnxruntime onnxruntime-gpu -y
!pip install onnxruntime-gpu==1.15.1
!pip install nvidia-cudnn-cu12==8.9.4.25
print("✅ GPU dependencies instaladas")
```

### Celda 4: Configurar Entorno
```python
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("✅ Entorno configurado")
```

### Celda 5: Descargar Modelo
```python
!wget https://civitai.com/api/download/models/85159 -O inswapper_128.onnx
print("✅ Modelo descargado")
```

### Celda 6: Verificar GPU
```python
import torch
import onnxruntime as ort
import tensorflow as tf

print(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
print(f"✅ ONNX Providers: {ort.get_available_providers()}")
print(f"✅ TensorFlow GPU: {tf.config.list_physical_devices('GPU')}")
print("✅ GPU configurado correctamente")
```

## Uso

### Procesamiento Individual
```python
!python run.py --source /content/source.jpg --target /content/video.mp4 --output /content/result.mp4 --execution-provider cuda --execution-threads 31 --temp-frame-quality 100 --keep-fps
```

### Procesamiento por Lotes
```python
!python run_batch_processing.py --source /content/source.jpg --videos /content/video1.mp4 /content/video2.mp4 --output-dir /content/resultados --execution-threads 31 --temp-frame-quality 100 --keep-fps
```

### Probar GPU
```python
!python test_gpu_force.py
```

## Características

- ✅ **Sin entorno virtual** - Instalación directa
- ✅ **GPU acceleration completa** - Tesla T4 optimizado
- ✅ **NSFW desactivado** - Sin verificaciones que ralenticen
- ✅ **31 hilos** - Máximo rendimiento
- ✅ **Calidad 100** - Máxima calidad de frames
- ✅ **Progreso detallado** - Líneas de actualización por video

## Notas

- **NumPy 1.26.4** - Compatible con todas las librerías
- **ONNX Runtime GPU** - Aceleración CUDA
- **TensorFlow 2.15.0** - Soporte GPU completo
- **Verificación NSFW desactivada** - Para evitar conflictos de GPU 