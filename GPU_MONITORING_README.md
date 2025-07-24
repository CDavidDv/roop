# 🎮 Monitoreo y Optimización de GPU para Roop

Este conjunto de herramientas te permite monitorear el uso de GPU durante el procesamiento de Roop y optimizar el rendimiento para tu GPU de 15GB.

## 📊 Herramientas Disponibles

### 1. `monitor_gpu_advanced.py` - Monitoreo Avanzado
Monitoreo en tiempo real con recomendaciones específicas para tu hardware.

```bash
# Verificar recursos del sistema
python monitor_gpu_advanced.py --check

# Monitoreo en tiempo real
python monitor_gpu_advanced.py --monitor

# Ver consejos de optimización
python monitor_gpu_advanced.py --tips
```

### 2. `run_roop_with_monitoring.py` - Roop con Monitoreo Integrado
Ejecuta Roop con monitoreo de GPU automático.

```bash
python run_roop_with_monitoring.py cara.jpg video.mp4 resultado.mp4
```

### 3. `optimize_for_15gb_gpu.py` - Optimizador Específico
Optimización automática para GPU de 15GB con configuraciones predefinidas.

```bash
python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4
```

## 🎯 Optimizaciones para 15GB VRAM

### Configuraciones Recomendadas
- **RAM máxima**: 8GB (para dejar espacio a VRAM)
- **Proveedor**: CUDA
- **Threads**: 8
- **Espera GPU**: 10 segundos entre procesadores
- **Formato frames**: JPG (ahorra espacio)
- **Encoder**: h264_nvenc (aceleración NVIDIA)
- **Calidad video**: 35 (balanceada)

### Monitoreo de Recursos
- **VRAM > 90%**: Reducir batch size
- **VRAM > 80%**: Monitorear uso
- **RAM > 90%**: Cerrar otras aplicaciones
- **RAM > 80%**: Monitorear uso

## 📈 Métricas que se Monitorean

### GPU
- Uso de VRAM (MB y porcentaje)
- Utilización de GPU (%)
- Temperatura (°C)
- Consumo de energía (W)
- Procesos usando GPU

### Sistema
- Uso de RAM (GB y porcentaje)
- Uso de CPU (%)
- Espacio disponible en disco

### Recomendaciones Automáticas
- Alertas cuando VRAM/RAM están altas
- Sugerencias de optimización
- Consejos específicos para tu hardware

## 🚀 Uso Rápido

### Opción 1: Monitoreo Manual
```bash
# Terminal 1: Iniciar monitoreo
python monitor_gpu_advanced.py --monitor

# Terminal 2: Ejecutar roop
python run.py -s cara.jpg -t video.mp4 -o resultado.mp4 --execution-provider cuda --max-memory 8
```

### Opción 2: Monitoreo Integrado
```bash
# Ejecutar con monitoreo automático
python run_roop_with_monitoring.py cara.jpg video.mp4 resultado.mp4
```

### Opción 3: Optimización Automática
```bash
# Optimización completa para 15GB VRAM
python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4 --keep-fps
```

## ⚡ Consejos de Rendimiento

### Para Máximo Rendimiento
1. **Cierra otras aplicaciones** que usen GPU
2. **Usa SSD** para archivos temporales
3. **Mantén drivers actualizados**
4. **Monitorea temperatura** (ideal < 80°C)

### Para Ahorrar Recursos
1. **Usa `--temp-frame-format jpg`** para ahorrar espacio
2. **Ajusta `--output-video-quality`** según necesites
3. **Considera `--skip-audio`** si no es necesario
4. **Usa `--keep-fps`** para mantener sincronización

### Para Procesamiento de Larga Duración
1. **Monitorea continuamente** con `monitor_gpu_advanced.py`
2. **Ajusta `--gpu-memory-wait`** según tu GPU
3. **Usa `--max-memory`** para limitar RAM
4. **Considera procesar en lotes** para videos largos

## 🔧 Solución de Problemas

### Error: "CUDA out of memory"
```bash
# Solución: Reducir uso de memoria
python run.py -s cara.jpg -t video.mp4 -o resultado.mp4 --max-memory 6 --gpu-memory-wait 15
```

### Error: "GPU not detected"
```bash
# Verificar drivers NVIDIA
nvidia-smi

# Verificar ONNX Runtime
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

### Rendimiento Lento
```bash
# Verificar uso de recursos
python monitor_gpu_advanced.py --check

# Optimizar configuraciones
python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4
```

## 📊 Interpretación de Métricas

### Barras de Progreso
```
VRAM: [████████████░░░░░░░░] 60.0%
RAM:  [████████░░░░░░░░░░░░] 40.0%
CPU:  [████████████████░░░░] 80.0%
```

### Niveles de Alerta
- **🟢 Verde**: < 70% - Óptimo
- **🟡 Amarillo**: 70-90% - Monitorear
- **🔴 Rojo**: > 90% - Tomar acción

### Recomendaciones
- **⚠️ VRAM alta**: Reducir batch size
- **⚠️ RAM alta**: Cerrar aplicaciones
- **✅ Balanceado**: Puedes aumentar rendimiento

## 📝 Logs y Historial

Los logs se guardan en `gpu_monitor.log` con formato JSON:
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "metrics": {
    "gpu_info": [...],
    "ram": {...},
    "cpu_percent": 45.2,
    "optimization": {...}
  }
}
```

## 🎯 Configuraciones Específicas para 15GB

### Para Videos Cortos (< 5 min)
```bash
python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4 --keep-fps
```

### Para Videos Largos (> 5 min)
```bash
python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4 --skip-audio --temp-frame-format jpg
```

### Para Múltiples Caras
```bash
python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4 --many-faces --gpu-memory-wait 15
```

### Para Máxima Calidad
```bash
python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4 --output-video-quality 20 --temp-frame-format png
```

¡Con estas herramientas podrás aprovechar al máximo tu GPU de 15GB y monitorear el rendimiento en tiempo real! 