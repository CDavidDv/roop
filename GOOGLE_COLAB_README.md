# ROOP con GPU en Google Colab

## 🚀 Configuración Rápida

### Paso 1: Clonar el repositorio
```bash
!git clone https://github.com/s0md3v/roop.git
%cd roop
```

### Paso 2: Ejecutar script de configuración
```bash
!python setup_colab_gpu.py
```

### Paso 3: Verificar GPU
```bash
!python test_gpu_force.py
```

### Paso 4: Procesar video
```bash
!python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4 --frame-processor face_swapper face_enhancer --gpu-memory-wait 30 --keep-fps
```

## 🔧 Problema y Solución

### El Problema
En Google Colab, el face swapper no usa GPU porque:
1. Se instala `onnxruntime==1.15.0` (versión CPU) en lugar de `onnxruntime-gpu==1.15.1`
2. El modelo de insightface falla al crear `CUDAExecutionProvider`
3. El face enhancer sí funciona con GPU porque usa PyTorch directamente

### La Solución
1. **Desinstalar onnxruntime CPU** e instalar **onnxruntime-gpu**
2. **Configurar múltiples estrategias** de proveedores en el face swapper
3. **Usar gestión de memoria** entre procesadores

## 📊 Rendimiento Esperado

### Con CPU (actual)
- **6 FPS** en Google Colab Tesla T4
- Tiempo de procesamiento: ~5-10 minutos para 1 minuto de video

### Con GPU (después de la configuración)
- **15-25 FPS** en Google Colab Tesla T4
- Tiempo de procesamiento: ~2-3 minutos para 1 minuto de video
- **Mejora de 3-4x** en velocidad

## 🛠️ Scripts Disponibles

### 1. `setup_colab_gpu.py`
Configuración completa automática:
- Desinstala onnxruntime CPU
- Instala onnxruntime-gpu
- Instala PyTorch con CUDA
- Instala TensorFlow GPU
- Descarga modelos necesarios
- Prueba face swapper y face enhancer

### 2. `test_gpu_force.py`
Verifica que GPU esté funcionando:
- Verifica ONNX Runtime providers
- Prueba face swapper con GPU
- Prueba face enhancer con GPU
- Prueba face analyser con GPU

### 3. `run_roop_with_memory_management.py`
Procesamiento optimizado con gestión de memoria:
```bash
python run_roop_with_memory_management.py \
  --source imagen.jpg \
  --target video.mp4 \
  --output resultado.mp4 \
  --gpu-memory-wait 30 \
  --keep-fps
```

## 🔍 Verificación de GPU

### Verificar ONNX Runtime
```python
import onnxruntime as ort
providers = ort.get_available_providers()
print(providers)
# Debe mostrar: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Verificar PyTorch
```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
```

### Verificar TensorFlow
```python
import tensorflow as tf
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
```

## 🎯 Comandos de Procesamiento

### Procesamiento Básico
```bash
python run.py \
  --source imagen.jpg \
  --target video.mp4 \
  --output resultado.mp4 \
  --frame-processor face_swapper face_enhancer \
  --keep-fps
```

### Procesamiento Optimizado (Recomendado)
```bash
python run_roop_with_memory_management.py \
  --source imagen.jpg \
  --target video.mp4 \
  --output resultado.mp4 \
  --gpu-memory-wait 30 \
  --keep-fps
```

### Procesamiento Secuencial
```bash
python run_roop_sequential.py
```

## ⚡ Optimizaciones para Google Colab

### 1. Gestión de Memoria GPU
- Pausa de 30 segundos entre face_swapper y face_enhancer
- Liberación automática de cachés de PyTorch y TensorFlow
- Monitoreo de VRAM en tiempo real

### 2. Configuración de Calidad
```bash
--temp-frame-quality 100  # Máxima calidad de frames
--output-video-quality 35 # Calidad de video balanceada
--keep-fps               # Mantener FPS original
```

### 3. Configuración de Memoria
```bash
--max-memory 12          # 12GB de RAM (Tesla T4)
--execution-threads 8    # 8 hilos para GPU
```

## 🐛 Solución de Problemas

### Error: "Failed to create CUDAExecutionProvider"
**Causa**: onnxruntime CPU instalado en lugar de GPU
**Solución**: Ejecutar `python setup_colab_gpu.py`

### Error: "Out of memory"
**Causa**: VRAM insuficiente
**Solución**: 
- Usar `--gpu-memory-wait 60` (esperar más tiempo)
- Procesar en lotes más pequeños
- Usar procesamiento secuencial

### Error: "No face detected"
**Causa**: Imagen fuente sin rostro claro
**Solución**: 
- Usar imagen con rostro frontal y bien iluminado
- Verificar que la imagen sea de buena calidad

## 📈 Comparación de Rendimiento

| Configuración | FPS | Tiempo (1 min video) | VRAM Usado |
|---------------|-----|---------------------|------------|
| CPU (actual)  | 6   | 5-10 minutos       | 0GB        |
| GPU (optimizado) | 15-25 | 2-3 minutos    | 2-4GB      |

## 🎉 Resultado Final

Después de la configuración, deberías ver:
- ✅ ONNX Runtime GPU disponible
- ✅ Face swapper usando CUDA
- ✅ Face enhancer usando CUDA
- ✅ Mejora de 3-4x en velocidad
- ✅ Procesamiento estable sin errores de memoria

¡Disfruta del procesamiento rápido con GPU! 🚀 