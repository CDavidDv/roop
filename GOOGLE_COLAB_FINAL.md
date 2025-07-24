# 🚀 Roop GPU Optimizado - Google Colab (VERSIÓN FINAL)

## Instalación Completa

### Celda 1: Clonar y Configurar
```python
!git clone https://github.com/CDavidDv/roop
%cd roop
!pip uninstall numpy -y
!pip install numpy==1.26.4 --no-cache-dir --force-reinstall
print("✅ Configuración inicial completada")
```

### Celda 2: Instalar Dependencias Básicas
```python
!pip install -r requirements.txt
!pip install opencv-python==4.8.1.78 insightface gfpgan basicsr facexlib sympy onnx customtkinter opennsfw2 torchvision
print("✅ Dependencias básicas instaladas")
```

### Celda 3: Instalar GPU Dependencies
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

### Celda 6: Solucionar Problemas Finales
```python
!python fix_final_issues.py
print("✅ Problemas finales solucionados")
```

### Celda 7: Verificar GPU
```python
import torch
import onnxruntime as ort
import tensorflow as tf

print(f"✅ PyTorch CUDA: {torch.cuda.is_available()}")
print(f"✅ ONNX Providers: {ort.get_available_providers()}")
print(f"✅ TensorFlow GPU: {tf.config.list_physical_devices('GPU')}")
print("✅ GPU configurado correctamente")
```

### Celda 8: Probar Todo
```python
!python test_gpu_force.py
print("✅ Pruebas completadas")
```

## Uso

### Procesamiento Individual
```python
!python run.py --source /content/source.jpg --target /content/video.mp4 --output /content/result.mp4 --execution-provider cuda --execution-threads 31 --temp-frame-quality 100 --keep-fps
```

### Procesamiento por Lotes (31 hilos, calidad 100)
```python
!python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/113.mp4 /content/114.mp4 /content/115.mp4 /content/116.mp4 /content/117.mp4 /content/118.mp4 /content/119.mp4 /content/120.mp4 --output-dir /content/resultados --execution-threads 31 --temp-frame-quality 100 --keep-fps
```

## Características

- ✅ **NumPy 1.26.4** - Compatible con todas las librerías
- ✅ **OpenCV 4.8.1.78** - Versión compatible con NumPy 1.x
- ✅ **GPU acceleration completa** - Face swapper, enhancer y analyser
- ✅ **NSFW desactivado** - Sin verificaciones que ralenticen
- ✅ **31 hilos** - Máximo rendimiento
- ✅ **Calidad 100** - Máxima calidad de frames
- ✅ **Progreso detallado** - Líneas de actualización por video
- ✅ **Sin entorno virtual** - Instalación simple

## Solución de Problemas

Si encuentras errores:

1. **Reinicia el runtime** después de instalar NumPy
2. **Ejecuta `python fix_final_issues.py`** para solucionar problemas
3. **Verifica GPU** con `python test_gpu_force.py`

## Resultado Esperado

```
🎭 PROBANDO FACE SWAPPER CON GPU:
========================================
[ROOP.FACE-SWAPPER] ✅ Forzando uso de GPU (CUDA)
[ROOP.FACE-SWAPPER] Cargando modelo con proveedores: ['CUDAExecutionProvider']
✅ Face swapper cargado exitosamente
✅ Face swapper usando GPU

✨ PROBANDO FACE ENHANCER CON GPU:
========================================
Dispositivo detectado: cuda
✅ Face enhancer configurado para usar GPU

🔍 PROBANDO FACE ANALYSER CON GPU:
========================================
✅ Analizador de rostros cargado exitosamente
✅ Analizador usando GPU
``` 