# Procesamiento por Lotes con GPU - ROOP

## 🚀 Configuración Optimizada para Google Colab Tesla T4

Este script permite procesar múltiples videos automáticamente con configuración optimizada para GPU Tesla T4.

## ✨ Características

- **GPU Forzado**: Face swapper, face enhancer y face analyser usan GPU automáticamente
- **Progreso en Tiempo Real**: Muestra el progreso detallado de cada video
- **Gestión de Memoria**: Pausas automáticas entre procesadores para evitar errores de VRAM
- **31 Hilos**: Configuración optimizada para máximo rendimiento
- **Calidad Máxima**: Temp frame quality 100 por defecto

## 🎯 Uso Rápido

```bash
python run_batch_processing.py \
  --source /content/DanielaAS.jpg \
  --videos /content/113.mp4 /content/114.mp4 /content/115.mp4 /content/116.mp4 /content/117.mp4 /content/118.mp4 /content/119.mp4 /content/120.mp4 \
  --output-dir /content/resultados \
  --execution-threads 31 \
  --temp-frame-quality 100 \
  --keep-fps
```

## ⚙️ Parámetros

| Parámetro | Descripción | Valor por Defecto |
|-----------|-------------|-------------------|
| `--source` | Imagen fuente | Requerido |
| `--videos` | Lista de videos a procesar | Requerido |
| `--output-dir` | Directorio de salida | Opcional |
| `--execution-threads` | Número de hilos | 31 |
| `--temp-frame-quality` | Calidad de frames temporales | 100 |
| `--max-memory` | Memoria máxima en GB | 12 |
| `--gpu-memory-wait` | Espera entre procesadores (segundos) | 30 |
| `--keep-fps` | Mantener FPS original | True |

## 🔧 Configuración GPU

### Face Swapper
- **GPU**: CUDA forzado automáticamente
- **Proveedores**: `['CUDAExecutionProvider']`
- **Fallback**: CPU si CUDA no está disponible

### Face Enhancer
- **GPU**: CUDA detectado automáticamente
- **Dispositivo**: `'cuda'` cuando está disponible
- **Fallback**: CPU si CUDA no está disponible

### Face Analyser
- **GPU**: CUDA forzado automáticamente
- **Proveedores**: `['CUDAExecutionProvider']`
- **Fallback**: CPU si CUDA no está disponible

## 📊 Progreso en Tiempo Real

El script muestra:

```
🚀 INICIANDO PROCESAMIENTO EN LOTE
============================================================
📸 Source: /content/DanielaAS.jpg
🎬 Videos a procesar: 8
⚙️ Configuración:
   • GPU Memory Wait: 30s
   • Max Memory: 12GB
   • Execution Threads: 31
   • Temp Frame Quality: 100
   • Keep FPS: True
============================================================

📊 PROGRESO GENERAL: 1/8 (12.5%)
⏱️ Tiempo transcurrido: 0.0s
✅ Completados: 0 | ❌ Fallidos: 0

🎬 PROCESANDO VIDEO: 113.mp4
📸 Source: DanielaAS.jpg
💾 Output: DanielaAS113.mp4
============================================================
🔄 Iniciando procesamiento...
⚙️ Configuración: 31 hilos, 12GB RAM, 30s GPU wait
📊 Progreso en tiempo real:
----------------------------------------
  📈 [ROOP.FACE-SWAPPER] ✅ Forzando uso de GPU (CUDA)
  📈 [ROOP.FACE-SWAPPER] Cargando modelo con proveedores: ['CUDAExecutionProvider']
  📈 [ROOP.CORE] Creating temporary resources...
  📈 [ROOP.CORE] Extracting frames with 30 FPS...
  📈 [FACE-SWAPPER] Iniciando procesamiento...
  📈 Processing face_swapper: 100%|██████████| 150/150 [02:30<00:00, 1.00frame/s]
  📈 [FACE-ENHANCER] Iniciando procesamiento...
  📈 Processing face_enhancer: 100%|██████████| 150/150 [01:45<00:00, 1.43frame/s]
  📈 [ROOP.CORE] Creating video with 30 FPS...
  📈 [ROOP.CORE] Restoring audio...
  📈 [ROOP.CORE] Cleaning temporary resources...
----------------------------------------
✅ Video procesado exitosamente: DanielaAS113.mp4
⏱️ Tiempo de procesamiento: 245.32 segundos
```

## 🎭 Verificación GPU

Para verificar que GPU está funcionando:

```bash
python test_gpu_force.py
```

Salida esperada:
```
🚀 INICIANDO PRUEBAS DE GPU FORZADO
==================================================
🔍 VERIFICACIÓN DE GPU:
========================================
ONNX Runtime providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
✅ CUDA GPU disponible para ONNX Runtime
PyTorch CUDA: True
PyTorch GPU: Tesla T4
PyTorch VRAM: 0.00GB

🎭 PROBANDO FACE SWAPPER CON GPU:
========================================
[ROOP.FACE-SWAPPER] ✅ Forzando uso de GPU (CUDA)
[ROOP.FACE-SWAPPER] Cargando modelo con proveedores: ['CUDAExecutionProvider']
✅ Face swapper cargado exitosamente

✨ PROBANDO FACE ENHANCER CON GPU:
========================================
[ROOP.FACE-ENHANCER] Forzando uso de GPU (CUDA)
Dispositivo detectado: cuda
✅ Face enhancer configurado para usar GPU

🔍 PROBANDO FACE ANALYSER CON GPU:
========================================
[FACE_ANALYSER] Forzando uso de GPU (CUDA)
✅ Analizador de rostros cargado exitosamente

🎉 PRUEBAS COMPLETADAS
==================================================
```

## 🚨 Solución de Problemas

### Error: "Failed to create CUDAExecutionProvider"
- **Causa**: ONNX Runtime GPU no instalado
- **Solución**: Instalar `onnxruntime-gpu` en lugar de `onnxruntime`

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### Error: "Out of memory"
- **Causa**: VRAM insuficiente
- **Solución**: Aumentar `--gpu-memory-wait` a 45-60 segundos

### Error: "CUDA not available"
- **Causa**: Drivers NVIDIA no actualizados
- **Solución**: Actualizar drivers NVIDIA

## 📈 Rendimiento Esperado

### Tesla T4 (15GB VRAM)
- **Face Swapper**: ~2-3 FPS
- **Face Enhancer**: ~1-2 FPS
- **Tiempo por video (1 min)**: ~3-5 minutos
- **Lote de 8 videos**: ~30-45 minutos

### Configuración Optimizada
- **Hilos**: 31 (máximo para Colab)
- **Memoria**: 12GB RAM
- **GPU Wait**: 30s entre procesadores
- **Calidad**: 100 (máxima)

## 🎯 Ejemplo Completo

```bash
# 1. Verificar GPU
python test_gpu_force.py

# 2. Procesar lote de videos
python run_batch_processing.py \
  --source /content/DanielaAS.jpg \
  --videos /content/113.mp4 /content/114.mp4 /content/115.mp4 \
  --output-dir /content/resultados \
  --execution-threads 31 \
  --temp-frame-quality 100 \
  --keep-fps

# 3. Verificar resultados
ls -la /content/resultados/
```

## 📝 Notas Importantes

1. **GPU Forzado**: El face swapper ahora usa GPU automáticamente
2. **Progreso Detallado**: Se muestra el progreso de cada paso
3. **Gestión de Memoria**: Pausas automáticas evitan errores de VRAM
4. **31 Hilos**: Configuración optimizada para Colab
5. **Calidad Máxima**: Temp frame quality 100 por defecto

## 🔄 Actualizaciones

- ✅ GPU forzado en face swapper
- ✅ Progreso en tiempo real
- ✅ Gestión de memoria GPU
- ✅ 31 hilos por defecto
- ✅ Calidad máxima por defecto 