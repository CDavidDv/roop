# Forzar Uso de GPU en ROOP

## Cambios Realizados

Se han modificado los siguientes archivos para forzar el uso de GPU (CUDA) siempre que esté disponible:

### 1. `roop/processors/frame/face_swapper.py`
- **Función modificada**: `get_face_swapper()`
- **Cambio**: Ahora detecta automáticamente si CUDA está disponible y lo usa prioritariamente
- **Comportamiento**: 
  - Si CUDA está disponible → usa `['CUDAExecutionProvider']`
  - Si CUDA no está disponible → usa la configuración del usuario como fallback

### 2. `roop/processors/frame/face_enhancer.py`
- **Función modificada**: `get_device()`
- **Cambio**: Detecta automáticamente CUDA y lo prioriza sobre CPU
- **Comportamiento**:
  - Si CUDA está disponible → devuelve `'cuda'`
  - Si CoreML está disponible → devuelve `'mps'`
  - Si ninguno está disponible → devuelve `'cpu'`

### 3. `roop/face_analyser.py`
- **Función modificada**: `get_face_analyser()`
- **Cambio**: Fuerza el uso de CUDA para el análisis de rostros
- **Comportamiento**: Similar al face swapper, prioriza CUDA sobre CPU

### 4. `roop/core.py`
- **Línea modificada**: Valor por defecto del argumento `--execution-provider`
- **Cambio**: Cambiado de `['cpu']` a `['cuda']`
- **Efecto**: Ahora por defecto intenta usar GPU en lugar de CPU

### 5. **NUEVA FUNCIONALIDAD**: Gestión de Memoria GPU
- **Archivo**: `roop/processors/frame/core.py`
- **Funciones añadidas**:
  - `clear_gpu_memory()`: Libera memoria GPU y cachés
  - `wait_for_gpu_memory_release()`: Espera y monitorea liberación de memoria
  - `process_video_with_memory_management()`: Procesa con pausas entre procesadores

### 6. **NUEVO ARGUMENTO**: `--gpu-memory-wait`
- **Propósito**: Configurar tiempo de espera entre procesadores
- **Valor por defecto**: 15 segundos
- **Uso**: `--gpu-memory-wait 30` para esperar 30 segundos

## Cómo Funciona

1. **Detección Automática**: El código detecta automáticamente si CUDA está disponible usando `onnxruntime.get_available_providers()`

2. **Priorización GPU**: Si CUDA está disponible, se usa prioritariamente sobre CPU

3. **Fallback Seguro**: Si CUDA no está disponible, el código cae automáticamente a la configuración del usuario

4. **Gestión de Memoria**: Entre procesadores (face_swapper → face_enhancer):
   - Libera cachés de PyTorch y TensorFlow
   - Ejecuta garbage collection
   - Espera tiempo configurable (por defecto 15s)
   - Monitorea uso de VRAM en tiempo real

5. **Mensajes Informativos**: Se muestran mensajes en consola indicando qué proveedor se está usando y el estado de la memoria

## Verificación

Para verificar que los cambios funcionan, ejecuta:

```bash
python test_gpu_force.py
```

Este script verificará:
- Disponibilidad de GPU
- Configuración del face swapper
- Configuración del face enhancer  
- Configuración del analizador de rostros

## Uso

### Opción 1: Comando original (funciona igual)
```bash
python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4 \
  --execution-provider cuda \
  --max-memory 12 \
  --execution-threads 33 \
  --frame-processor face_swapper face_enhancer \
  --temp-frame-quality 100 \
  --keep-fps
```

### Opción 2: Comando simplificado (recomendado)
```bash
python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4 \
  --frame-processor face_swapper face_enhancer \
  --gpu-memory-wait 30 \
  --keep-fps
```

### Opción 3: Script especializado
```bash
python run_roop_with_memory_management.py \
  --source imagen.jpg \
  --target video.mp4 \
  --output resultado.mp4 \
  --gpu-memory-wait 30 \
  --keep-fps
```

## Ventajas

1. **Rendimiento Mejorado**: El face swapper será significativamente más rápido con GPU
2. **Configuración Automática**: No necesitas especificar manualmente el proveedor de ejecución
3. **Compatibilidad**: Mantiene compatibilidad con sistemas sin GPU
4. **Transparencia**: Muestra claramente qué proveedor se está usando
5. **Gestión de Memoria**: Evita errores de memoria GPU entre procesadores
6. **Configurabilidad**: Puedes ajustar el tiempo de espera según tu GPU

## Solución al Problema de Memoria

**Problema**: Entre face_swapper y face_enhancer se agota la memoria GPU
**Solución**: 
- Pausa automática de 15s (configurable) entre procesadores
- Liberación de cachés GPU (PyTorch + TensorFlow)
- Garbage collection forzado
- Monitoreo de VRAM en tiempo real

## Notas Importantes

- Asegúrate de tener instalado `onnxruntime-gpu` en lugar de `onnxruntime`
- Verifica que tienes los drivers de NVIDIA actualizados
- El uso de GPU requiere más memoria VRAM
- En sistemas sin GPU, el rendimiento será el mismo que antes
- El tiempo de espera entre procesadores es configurable según tu GPU
- Para GPUs con más VRAM, puedes reducir el tiempo de espera
- Para GPUs con menos VRAM, aumenta el tiempo de espera 