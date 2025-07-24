# Solución para Forzar GPU en Face Swapper

## Problema Identificado

Según tu salida, el face swapper está fallando al crear el `CUDAExecutionProvider` y está cayendo de vuelta al CPU:

```
Failed to create CUDAExecutionProvider. Please reference https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements to ensure all dependencies are met.
```

## Causas del Problema

1. **Versión incorrecta de onnxruntime**: Puede estar instalada la versión CPU en lugar de la GPU
2. **Dependencias faltantes**: TensorRT, cuDNN, o versiones incompatibles
3. **Configuración de entorno**: Variables de entorno no configuradas correctamente
4. **Conflictos de versiones**: En Google Colab, las versiones pueden estar en conflicto

## Soluciones

### Opción 1: Script Automático para Google Colab

Ejecuta este script que configura automáticamente todo:

```bash
python colab_cuda_setup.py
```

Este script:
- ✅ Verifica que estés en Google Colab
- ✅ Instala la versión correcta de `onnxruntime-gpu`
- ✅ Configura variables de entorno
- ✅ Prueba la creación de sesiones CUDA
- ✅ Verifica que el face swapper use GPU

### Opción 2: Instalación Manual

Si el script automático no funciona, sigue estos pasos:

#### Paso 1: Desinstalar onnxruntime actual
```bash
pip uninstall -y onnxruntime
```

#### Paso 2: Instalar onnxruntime-gpu
```bash
pip install onnxruntime-gpu==1.16.3
```

#### Paso 3: Configurar variables de entorno
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

#### Paso 4: Verificar instalación
```bash
python test_gpu_force.py
```

### Opción 3: Script de Diagnóstico Detallado

Para un diagnóstico completo:

```bash
python install_cuda_dependencies.py
```

Este script verifica:
- ✅ CUDA Toolkit
- ✅ Drivers NVIDIA
- ✅ onnxruntime-gpu
- ✅ TensorRT
- ✅ cuDNN

## Verificación

Después de aplicar cualquiera de las soluciones, ejecuta:

```bash
python test_gpu_force.py
```

Deberías ver:
```
✅ CUDA GPU disponible para ONNX Runtime
✅ GPU CUDA confirmado en uso para face swapper
```

## Cambios Realizados en el Código

### 1. Face Swapper Mejorado (`roop/processors/frame/face_swapper.py`)

- **Configuración robusta de GPU**: Intenta múltiples configuraciones de proveedores
- **Manejo de errores mejorado**: Si una configuración falla, prueba la siguiente
- **Verificación de GPU**: Confirma que realmente se está usando CUDA
- **Fallback seguro**: Si todo falla, usa CPU como último recurso

### 2. Script de Diagnóstico Mejorado (`test_gpu_force.py`)

- **Verificación de dependencias**: Revisa CUDA Toolkit, drivers, onnxruntime-gpu
- **Prueba manual de CUDA**: Crea sesiones ONNX Runtime para verificar funcionamiento
- **Diagnóstico detallado**: Muestra exactamente qué está fallando

### 3. Scripts de Instalación

- **`colab_cuda_setup.py`**: Específico para Google Colab
- **`install_cuda_dependencies.py`**: General para cualquier entorno

## Configuraciones de GPU Probadas

El face swapper ahora intenta estas configuraciones en orden:

1. `['CUDAExecutionProvider']` - Solo CUDA
2. `['CUDAExecutionProvider', 'CPUExecutionProvider']` - CUDA con fallback a CPU
3. `['TensorrtExecutionProvider', 'CUDAExecutionProvider']` - TensorRT + CUDA
4. `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']` - Completo

## Opciones de CUDA

Para cada configuración, se aplican estas opciones:

```python
provider_options['CUDAExecutionProvider'] = {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
    'cudnn_conv_use_max_workspace': '1',
    'do_copy_in_default_stream': '1',
}
```

## Rendimiento Esperado

Con GPU funcionando correctamente:

- **Face Swapper**: 10-50x más rápido que CPU
- **Face Enhancer**: Ya usa GPU (funciona bien)
- **Face Analyser**: 5-20x más rápido que CPU

## Troubleshooting

### Error: "CUDAExecutionProvider not available"

1. Verifica que `onnxruntime-gpu` esté instalado:
   ```bash
   pip list | grep onnxruntime
   ```

2. Reinstala con versión específica:
   ```bash
   pip uninstall -y onnxruntime
   pip install onnxruntime-gpu==1.16.3
   ```

### Error: "CUDA out of memory"

1. Reduce el límite de memoria GPU:
   ```python
   'gpu_mem_limit': 1 * 1024 * 1024 * 1024,  # 1GB en lugar de 2GB
   ```

2. Usa el argumento `--gpu-memory-wait`:
   ```bash
   python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4 --gpu-memory-wait 30
   ```

### Error: "Failed to create CUDAExecutionProvider"

1. Verifica drivers NVIDIA:
   ```bash
   nvidia-smi
   ```

2. Verifica CUDA Toolkit:
   ```bash
   nvcc --version
   ```

3. Ejecuta el script de diagnóstico:
   ```bash
   python test_gpu_force.py
   ```

## Comandos Recomendados

### Para Google Colab:
```bash
# Configurar CUDA
python colab_cuda_setup.py

# Verificar
python test_gpu_force.py

# Ejecutar ROOP con GPU
python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4 --frame-processor face_swapper face_enhancer --gpu-memory-wait 30 --keep-fps
```

### Para entorno local:
```bash
# Instalar dependencias
python install_cuda_dependencies.py

# Verificar
python test_gpu_force.py

# Ejecutar ROOP con GPU
python run.py --source imagen.jpg --target video.mp4 --output resultado.mp4 --frame-processor face_swapper face_enhancer --gpu-memory-wait 30 --keep-fps
```

## Notas Importantes

1. **Google Colab**: Usa el script `colab_cuda_setup.py` específicamente diseñado para Colab
2. **Tesla T4**: Con 15GB VRAM, puedes usar configuraciones más agresivas
3. **Memoria**: El face swapper puede usar hasta 2-4GB de VRAM
4. **Tiempo de espera**: Entre face_swapper y face_enhancer, usa `--gpu-memory-wait 30`

## Resultado Esperado

Después de aplicar estas soluciones, deberías ver en la salida:

```
[ROOP.FACE-SWAPPER] ✅ GPU CUDA confirmado en uso
[ROOP.FACE-SWAPPER] Modelo cargado con proveedores: ['CUDAExecutionProvider']
```

Y el procesamiento será significativamente más rápido que los 6 frames por segundo actuales. 