# Procesamiento en Lote con ROOP

## Scripts Disponibles

### 1. `run_batch_array.py` (RECOMENDADO - Más Simple)
**Solo modifica el array y ejecuta:**

```python
VIDEOS_TO_PROCESS = [
    "/content/17.mp4",
    "/content/18.mp4", 
    "/content/19.mp4",
    "/content/20.mp4"
]
```

**Uso:**
```bash
python run_batch_array.py
```

### 2. `run_batch_simple.py` (Configuración Completa)
**Modifica la configuración completa:**

```python
# Imagen fuente
SOURCE_IMAGE = "/content/SakuraAS.png"

# Array de videos
VIDEOS_TO_PROCESS = [
    "/content/17.mp4",
    "/content/18.mp4", 
    "/content/19.mp4",
    "/content/20.mp4"
]

# Directorio de salida
OUTPUT_DIR = "/content/resultados"

# Configuración
GPU_MEMORY_WAIT = 30
MAX_MEMORY = 12
EXECUTION_THREADS = 8
TEMP_FRAME_QUALITY = 100
KEEP_FPS = True
```

**Uso:**
```bash
python run_batch_simple.py
```

### 3. `run_batch_processing.py` (Línea de Comandos)
**Usa argumentos de línea de comandos:**

```bash
python run_batch_processing.py \
  --source /content/SakuraAS.png \
  --videos /content/17.mp4 /content/18.mp4 /content/19.mp4 /content/20.mp4 \
  --output-dir /content/resultados \
  --gpu-memory-wait 30 \
  --keep-fps
```

## Características

### ✅ **Gestión Automática de Memoria**
- Pausa de 30s entre procesadores (face_swapper → face_enhancer)
- Pausa de 10s entre videos diferentes
- Liberación automática de cachés GPU
- Monitoreo de VRAM en tiempo real

### ✅ **Nombres de Salida Automáticos**
- **Entrada**: `17.mp4`, `18.mp4`, `19.mp4`, `20.mp4`
- **Salida**: `SakuraAS17.mp4`, `SakuraAS18.mp4`, `SakuraAS19.mp4`, `SakuraAS20.mp4`

### ✅ **Progreso y Estadísticas**
- Muestra progreso: `🎬 PROCESANDO VIDEO 2/4: 18.mp4`
- Tiempo de procesamiento por video
- Resumen final con tasa de éxito
- Manejo de errores individual

### ✅ **Verificaciones Automáticas**
- Verifica que el source existe
- Verifica que cada video existe
- Crea directorio de salida si no existe
- Continúa con el siguiente video si uno falla

## Ejemplo de Salida

```
🚀 PROCESAMIENTO EN LOTE AUTOMÁTICO
============================================================
📸 Source: /content/SakuraAS.png
🎬 Videos a procesar: 4
📁 Output: /content/resultados
============================================================

🎬 PROCESANDO VIDEO 1/4: 17.mp4
============================================================
[FACE-SWAPPER] Forzando uso de GPU (CUDA)
[FACE-ENHANCER] Forzando uso de GPU (CUDA)
✅ Video procesado exitosamente: SakuraAS17.mp4
⏱️ Tiempo: 245.32 segundos

⏳ Esperando 10 segundos...

🎬 PROCESANDO VIDEO 2/4: 18.mp4
============================================================
...
```

## Configuración Recomendada

### Para GPUs con 8GB VRAM:
```python
GPU_MEMORY_WAIT = 30
MAX_MEMORY = 8
EXECUTION_THREADS = 6
```

### Para GPUs con 12GB+ VRAM:
```python
GPU_MEMORY_WAIT = 20
MAX_MEMORY = 12
EXECUTION_THREADS = 8
```

### Para GPUs con 24GB+ VRAM:
```python
GPU_MEMORY_WAIT = 15
MAX_MEMORY = 16
EXECUTION_THREADS = 12
```

## Ventajas del Procesamiento en Lote

1. **Automatización Completa**: Solo configura el array y ejecuta
2. **Gestión de Memoria**: Evita errores de memoria GPU
3. **Nombres Automáticos**: No necesitas especificar cada nombre de salida
4. **Progreso Visual**: Ve el progreso en tiempo real
5. **Recuperación de Errores**: Si un video falla, continúa con el siguiente
6. **Estadísticas**: Resumen final con tasa de éxito
7. **Configurabilidad**: Ajusta parámetros según tu GPU

## Uso Rápido

1. **Edita el array** en `run_batch_array.py`:
```python
VIDEOS_TO_PROCESS = [
    "/content/17.mp4",
    "/content/18.mp4", 
    "/content/19.mp4",
    "/content/20.mp4"
]
```

2. **Ejecuta**:
```bash
python run_batch_array.py
```

3. **Resultados** en `/content/resultados/`:
- `SakuraAS17.mp4`
- `SakuraAS18.mp4`
- `SakuraAS19.mp4`
- `SakuraAS20.mp4`

¡Y listo! 🎉 