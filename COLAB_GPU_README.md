# ROOP Optimizado para Google Colab T4

## 🚀 Configuración Rápida para Google Colab

Este proyecto ha sido optimizado específicamente para Google Colab con GPU T4, proporcionando mejor rendimiento y gestión eficiente de memoria.

### 📋 Requisitos
- Google Colab con GPU T4 (15GB VRAM)
- 12GB RAM
- 500GB almacenamiento

## 🛠️ Instalación

### Paso 1: Clonar el repositorio
```bash
!git clone https://github.com/s0md3v/roop.git
%cd roop
```

### Paso 2: Instalar ROOP y dependencias optimizadas
```bash
!python install_roop_colab.py
```

### Paso 3: Optimizar GPU
```bash
!python optimize_colab_gpu.py
```

## 🎯 Uso Optimizado

### Procesamiento Individual
```bash
!python run_colab_gpu_optimized.py \
  --source imagen.jpg \
  --target video.mp4 \
  -o resultado.mp4 \
  --gpu-memory-wait 30 \
  --keep-fps
```

### Procesamiento en Lote
```bash
!python run_colab_gpu_optimized.py \
  --source imagen.jpg \
  --target carpeta_videos \
  --batch \
  --output-dir resultados \
  --gpu-memory-wait 30 \
  --keep-fps
```

## ⚙️ Parámetros Optimizados para T4

| Parámetro | Valor Recomendado | Descripción |
|-----------|-------------------|-------------|
| `--gpu-memory-wait` | 30 | Tiempo de espera entre procesadores (segundos) |
| `--max-memory` | 12 | Memoria máxima en GB |
| `--execution-threads` | 4 | Hilos de ejecución (optimizado para T4) |
| `--temp-frame-quality` | 100 | Calidad de frames temporales |
| `--keep-fps` | ✓ | Mantener FPS original |

## 🔧 Optimizaciones Implementadas

### 1. Gestión de Memoria GPU
- **Limpieza automática**: Libera memoria GPU entre procesadores
- **Monitoreo en tiempo real**: Controla uso de VRAM
- **Pausas inteligentes**: Espera memoria disponible automáticamente

### 2. Configuración T4 Específica
- **VRAM limitada**: Optimizado para 15GB de VRAM
- **Threads reducidos**: 4 hilos para evitar saturación
- **Memoria compartida**: Configuración para memoria unificada

### 3. Variables de Entorno Optimizadas
```python
TF_FORCE_GPU_ALLOW_GROWTH = 'true'
TF_CPP_MIN_LOG_LEVEL = '2'
CUDA_VISIBLE_DEVICES = '0'
TF_FORCE_UNIFIED_MEMORY = '1'
TF_MEMORY_ALLOCATION = '0.8'
TF_GPU_MEMORY_LIMIT = '12'
```

## 📊 Monitoreo de Rendimiento

### Verificar GPU
```bash
!python test_gpu_force.py
```

### Monitorear Memoria
```bash
!python monitor_gpu.py
```

## 🎬 Ejemplos de Uso

### Ejemplo 1: Video Individual
```python
# Subir archivos
from google.colab import files
uploaded = files.upload()

# Procesar
!python run_colab_gpu_optimized.py \
  --source "imagen_fuente.jpg" \
  --target "video_objetivo.mp4" \
  -o "resultado_faceswap.mp4" \
  --gpu-memory-wait 30 \
  --keep-fps
```

### Ejemplo 2: Lote de Videos
```python
# Crear carpeta de videos
!mkdir videos_entrada
!mkdir videos_salida

# Subir videos a la carpeta
# (usar interfaz de Colab para subir múltiples archivos)

# Procesar lote
!python run_colab_gpu_optimized.py \
  --source "imagen_fuente.jpg" \
  --target "videos_entrada" \
  --batch \
  --output-dir "videos_salida" \
  --gpu-memory-wait 30 \
  --keep-fps
```

## 🔍 Solución de Problemas

### Error: "CUDA out of memory"
```bash
# Reducir memoria máxima
!python run_colab_gpu_optimized.py \
  --source imagen.jpg \
  --target video.mp4 \
  -o resultado.mp4 \
  --max-memory 8 \
  --gpu-memory-wait 45
```

### Error: "GPU not available"
```bash
# Verificar GPU
!nvidia-smi

# Reinstalar dependencias GPU
!pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### Rendimiento Lento
```bash
# Optimizar configuración
!python optimize_colab_gpu.py

# Usar menos threads
!python run_colab_gpu_optimized.py \
  --source imagen.jpg \
  --target video.mp4 \
  -o resultado.mp4 \
  --execution-threads 2
```

## 📈 Comparación de Rendimiento

| Configuración | Tiempo Estimado | Memoria VRAM |
|---------------|-----------------|--------------|
| CPU | 10-15 min | 0GB |
| GPU T4 (optimizado) | 2-3 min | 8-12GB |
| GPU T4 (sin optimizar) | 4-5 min | 12-15GB |

## 🎨 Formatos Soportados

### Entrada
- **Imágenes**: JPG, PNG, BMP
- **Videos**: MP4, AVI, MOV, MKV

### Salida
- **Videos**: MP4 (H.264)
- **Calidad**: Mantiene calidad original
- **FPS**: Opcional mantener FPS original

## 🔄 Procesamiento en Lote Eficiente

### Características del Modo Lote
- **Gestión automática de memoria**: Limpia GPU entre videos
- **Progreso en tiempo real**: Muestra avance del procesamiento
- **Recuperación de errores**: Continúa con siguiente video si falla uno
- **Resumen final**: Estadísticas de éxito/fallo

### Configuración Recomendada para Lotes
```bash
!python run_colab_gpu_optimized.py \
  --source imagen.jpg \
  --target carpeta_videos \
  --batch \
  --output-dir resultados \
  --gpu-memory-wait 45 \
  --max-memory 10 \
  --execution-threads 2 \
  --keep-fps
```

## 🚨 Notas Importantes

1. **Reiniciar runtime**: Si cambias de GPU a CPU o viceversa
2. **Limpiar memoria**: Ejecutar `optimize_colab_gpu.py` antes de cada sesión
3. **Monitorear VRAM**: Usar `monitor_gpu.py` para verificar uso de memoria
4. **Tiempo de espera**: Entre 30-45 segundos entre procesadores es óptimo para T4
5. **Calidad vs Velocidad**: Reducir `temp-frame-quality` para mayor velocidad

## 📞 Soporte

Si encuentras problemas:
1. Ejecuta `test_gpu_force.py` para diagnosticar
2. Verifica que tienes GPU T4 asignada
3. Reinicia el runtime de Colab
4. Reinstala dependencias con `install_colab_gpu.py`

---

**Optimizado para Google Colab T4 - Rendimiento mejorado con gestión eficiente de memoria** 