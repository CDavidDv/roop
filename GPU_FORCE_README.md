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

## Cómo Funciona

1. **Detección Automática**: El código detecta automáticamente si CUDA está disponible usando `onnxruntime.get_available_providers()`

2. **Priorización GPU**: Si CUDA está disponible, se usa prioritariamente sobre CPU

3. **Fallback Seguro**: Si CUDA no está disponible, el código cae automáticamente a la configuración del usuario

4. **Mensajes Informativos**: Se muestran mensajes en consola indicando qué proveedor se está usando

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

Ahora el face swapper siempre intentará usar GPU por defecto:

```bash
# Usar GPU (por defecto)
python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4

# Forzar CPU (si es necesario)
python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4 --execution-provider cpu
```

## Ventajas

1. **Rendimiento Mejorado**: El face swapper será significativamente más rápido con GPU
2. **Configuración Automática**: No necesitas especificar manualmente el proveedor de ejecución
3. **Compatibilidad**: Mantiene compatibilidad con sistemas sin GPU
4. **Transparencia**: Muestra claramente qué proveedor se está usando

## Notas Importantes

- Asegúrate de tener instalado `onnxruntime-gpu` en lugar de `onnxruntime`
- Verifica que tienes los drivers de NVIDIA actualizados
- El uso de GPU requiere más memoria VRAM
- En sistemas sin GPU, el rendimiento será el mismo que antes 