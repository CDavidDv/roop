# 🚀 ROOP GPU Processing para Google Colab

Scripts optimizados para procesar videos con face swap usando GPU en Google Colab T4.

## 📋 Requisitos

- Google Colab con GPU T4 (15GB VRAM)
- 12GB RAM
- 500GB almacenamiento

## 🛠️ Instalación Rápida

### 1. Clonar ROOP
```bash
!git clone --branch v3 https://github.com/CDavidDv/roop.git
%cd roop
```

### 2. Configurar entorno
```bash
!python colab_setup.py
```

### 3. Descargar modelo (si no se descargó automáticamente)
```bash
!wget https://civitai.com/api/download/models/85159 -O inswapper_128.onnx
```

## 📁 Estructura de Carpetas

```
/content/
├── sources/          # Imágenes fuente
├── videos/           # Videos a procesar
└── resultados/       # Videos procesados
```

## 🎯 Uso Rápido

### Opción 1: Configuración Automática
```python
# Ejecutar configuración
!python colab_setup.py

# Procesar videos (edita las rutas en el script)
!python run_colab_gpu.py
```

### Opción 2: Configuración Manual
```python
# Procesar con parámetros personalizados
!python run_batch_processing.py \
  --source /content/sources/mi_imagen.jpg \
  --input-folder /content/videos \
  --output-dir /content/resultados \
  --max-memory 12 \
  --execution-threads 30 \
  --temp-frame-quality 100 \
  --keep-fps
```

## ⚙️ Parámetros Optimizados para T4

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `--max-memory` | 12 | Memoria máxima en GB |
| `--execution-threads` | 30 | Hilos de procesamiento |
| `--temp-frame-quality` | 100 | Calidad de frames |
| `--gpu-memory-wait` | 30 | Espera entre procesadores |
| `--keep-fps` | True | Mantener FPS original |

## 📊 Características

- ✅ **Procesamiento automático** de todos los videos en una carpeta
- ✅ **Optimización GPU** para T4
- ✅ **Sin entorno virtual** - usa Python directamente
- ✅ **Manejo de errores** mejorado
- ✅ **Progreso en tiempo real**
- ✅ **Liberación de memoria** entre videos

## 🎬 Formatos Soportados

- **Entrada**: MP4, AVI, MOV, MKV, WMV, FLV, WEBM
- **Salida**: MP4

## 🔧 Solución de Problemas

### Error: "No such file or directory: 'roop_env/bin/python'"
**Solución**: Los scripts actualizados usan `python` directamente sin entorno virtual.

### Error: "Unable to register cuDNN factory"
**Solución**: Este warning es normal en Colab y no afecta el funcionamiento.

### Error: "CUDA out of memory"
**Solución**: Reduce `--max-memory` a 10 o 8 GB.

### Videos muy lentos
**Solución**: 
- Reduce `--execution-threads` a 15-20
- Reduce `--temp-frame-quality` a 80-90

## 📈 Optimización de Rendimiento

### Para videos largos:
```bash
--max-memory 10 \
--execution-threads 20 \
--temp-frame-quality 80
```

### Para máxima calidad:
```bash
--max-memory 12 \
--execution-threads 30 \
--temp-frame-quality 100
```

### Para procesamiento rápido:
```bash
--max-memory 8 \
--execution-threads 15 \
--temp-frame-quality 70
```

## 🎯 Ejemplo Completo

```python
# 1. Configurar
!python colab_setup.py

# 2. Subir archivos (manual o automático)
# - Imagen fuente a /content/sources/
# - Videos a /content/videos/

# 3. Procesar
!python run_colab_gpu.py

# 4. Descargar resultados
from google.colab import files
import os

# Comprimir resultados
!zip -r resultados.zip /content/resultados/

# Descargar
files.download('resultados.zip')
```

## 📝 Notas Importantes

- ⏱️ **Tiempo de procesamiento**: 2-5 minutos por minuto de video
- 💾 **Espacio requerido**: 3-5x el tamaño del video original
- 🔄 **Memoria**: Se libera automáticamente entre videos
- ⚠️ **Interrupciones**: El procesamiento se puede reanudar

## 🆘 Soporte

Si encuentras problemas:

1. Verifica que tienes GPU habilitada en Colab
2. Ejecuta `!python colab_setup.py` para verificar dependencias
3. Revisa los logs de error para identificar el problema específico
4. Reduce los parámetros de memoria si hay errores de CUDA

---

**¡Disfruta procesando tus videos con GPU! 🎬✨** 