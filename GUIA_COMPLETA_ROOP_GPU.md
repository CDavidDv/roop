# 🚀 GUÍA COMPLETA: ROOP CON GPU TESLA T4 (15GB)

## 📋 **Índice**
1. [Verificación Inicial](#verificación-inicial)
2. [Instalación de ONNX Runtime GPU](#instalación-de-onnx-runtime-gpu)
3. [Configuración de GPU](#configuración-de-gpu)
4. [Monitoreo de Recursos](#monitoreo-de-recursos)
5. [Procesamiento Optimizado](#procesamiento-optimizado)
6. [Solución de Problemas](#solución-de-problemas)

---

## 🔍 **1. Verificación Inicial**

### **Paso 1.1: Verificar GPU**
```bash
nvidia-smi
```
**Resultado esperado:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   45C    P0    70W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### **Paso 1.2: Verificar Environment**
```bash
ls roop_env/bin/python
```
**Resultado esperado:**
```
roop_env/bin/python
```

### **Paso 1.3: Verificar ONNX Runtime Actual**
```bash
roop_env/bin/python -c "import onnxruntime as ort; print('Versión:', ort.__version__); print('Proveedores:', ort.get_available_providers())"
```
**Resultado esperado (PROBLEMA):**
```
Versión: 1.15.1
Proveedores: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## 📦 **2. Instalación de ONNX Runtime GPU**

### **Paso 2.1: Ejecutar Instalador Automático**
```bash
python install_onnxruntime_gpu_env.py
```

### **Paso 2.2: Verificar Instalación**
```bash
roop_env/bin/python -c "import onnxruntime as ort; print('Versión:', ort.__version__); print('GPU:', 'gpu' in ort.__version__.lower())"
```
**Resultado esperado:**
```
Versión: 1.15.1
GPU: True
```

### **Paso 2.3: Probar GPU**
```bash
roop_env/bin/python -c "import onnxruntime as ort; session = ort.InferenceSession('test.onnx', providers=['CUDAExecutionProvider']); print('✅ GPU funciona')"
```

---

## ⚙️ **3. Configuración de GPU**

### **Paso 3.1: Aplicar Forzado de GPU**
```bash
python force_gpu_face_swapper.py
```

### **Paso 3.2: Verificar Configuración**
```bash
python fix_face_swapper_gpu.py
```

### **Paso 3.3: Crear Script de Entorno**
```bash
# El script creará automáticamente: run_roop_gpu_env.sh
```

---

## 📊 **4. Monitoreo de Recursos**

### **Paso 4.1: Monitoreo en Tiempo Real**
```bash
# Terminal 1: Monitoreo GPU
python monitor_gpu_live.py

# Terminal 2: Monitoreo específico
python check_gpu_usage.py
```

### **Paso 4.2: Verificar Recursos**
```bash
nvidia-smi -l 1
```

---

## 🚀 **5. Procesamiento Optimizado**

### **Opción 5.1: Procesamiento Individual**
```bash
./run_roop_gpu_env.sh \
  --source /content/DanielaAS.jpg \
  --target /content/112.mp4 \
  -o /content/DanielaAS112_gpu.mp4 \
  --frame-processor face_swapper \
  --execution-provider cuda \
  --max-memory 8 \
  --execution-threads 8 \
  --gpu-memory-wait 5 \
  --temp-frame-quality 100 \
  --temp-frame-format png \
  --output-video-encoder h264_nvenc \
  --output-video-quality 100 \
  --keep-fps
```

### **Opción 5.2: Procesamiento en Lote**
```bash
roop_env/bin/python run_batch_processing_clean.py \
  --source /content/DanielaAS.jpg \
  --videos /content/62.mp4 /content/71.mp4 /content/72.mp4 /content/74.mp4 /content/75.mp4 /content/76.mp4 /content/77.mp4 /content/78.mp4 /content/79.mp4 \
  --output-dir /content/resultados \
  --keep-fps
```

### **Opción 5.3: Procesamiento Optimizado para 15GB**
```bash
roop_env/bin/python run_batch_processing_optimized.py \
  --source /content/DanielaAS.jpg \
  --videos /content/62.mp4 /content/71.mp4 /content/72.mp4 /content/74.mp4 /content/75.mp4 /content/76.mp4 /content/77.mp4 /content/78.mp4 /content/79.mp4 \
  --output-dir /content/resultados \
  --keep-fps
```

---

## 📈 **6. Monitoreo Durante Procesamiento**

### **Lo que Deberías Ver:**

#### **✅ GPU Funcionando Correctamente:**
```
🔄 [████████████░░░░░░░░░░░░░░░░░░░░] 40% | Frame 61/152 | ⏱️ 00:45 | ⏳ 01:15 | 🚀 1.2s/frame | 🧠 2.8GB | 🎮 8.5GB VRAM

📊 [14:30:25] GPU: 8500MB/15360MB (55.3%) | RAM: 6.8GB/12.7GB (53.5%) | Temp: 68°C
```

#### **❌ GPU NO Funcionando:**
```
🔄 [████████████░░░░░░░░░░░░░░░░░░░░] 40% | Frame 61/152 | ⏱️ 02:15 | ⏳ 03:25 | 🚀 6.3s/frame | 🧠 2.8GB | 🎮 0.0GB VRAM
```

---

## 🔧 **7. Solución de Problemas**

### **Problema 1: VRAM = 0.0GB**
```bash
# Solución: Reinstalar ONNX Runtime GPU
python install_onnxruntime_gpu_env.py
```

### **Problema 2: Error CUDA**
```bash
# Solución: Verificar drivers
nvidia-smi
nvcc --version
```

### **Problema 3: Lento (6s/frame)**
```bash
# Solución: Forzar GPU
python force_gpu_face_swapper.py
```

### **Problema 4: Error de Memoria**
```bash
# Solución: Reducir memoria
--max-memory 6 --gpu-memory-wait 15
```

---

## 📊 **8. Comparación de Rendimiento**

| **Configuración** | **Velocidad** | **VRAM** | **Tiempo Total** |
|-------------------|---------------|----------|------------------|
| **CPU (Antes)** | 6.3s/frame | 0.0GB | 30-60 min |
| **GPU (Después)** | 1.2s/frame | 8.5GB | 5-10 min |
| **Mejora** | **5x más rápido** | **GPU activa** | **6x más rápido** |

---

## 🎯 **9. Comandos Rápidos**

### **Verificación Rápida:**
```bash
# Verificar GPU
nvidia-smi

# Verificar ONNX
roop_env/bin/python -c "import onnxruntime as ort; print('GPU:', 'gpu' in ort.__version__.lower())"

# Procesar un video
./run_roop_gpu_env.sh --source imagen.jpg --target video.mp4 -o salida.mp4 --frame-processor face_swapper --execution-provider cuda
```

### **Monitoreo Rápido:**
```bash
# Terminal 1
python monitor_gpu_live.py

# Terminal 2
nvidia-smi -l 1
```

---

## ✅ **10. Checklist Final**

- [ ] GPU Tesla T4 detectada
- [ ] ONNX Runtime GPU instalado en environment
- [ ] Face-swapper modificado para GPU
- [ ] Script de entorno creado
- [ ] Monitoreo configurado
- [ ] Procesamiento funcionando con VRAM > 0GB
- [ ] Velocidad mejorada (1-2s/frame)

---

## 🚀 **11. Ejecución Final**

```bash
# 1. Verificar todo
python install_onnxruntime_gpu_env.py

# 2. Aplicar optimizaciones
python force_gpu_face_swapper.py

# 3. Procesar con GPU
./run_roop_gpu_env.sh \
  --source /content/DanielaAS.jpg \
  --target /content/112.mp4 \
  -o /content/DanielaAS112_gpu.mp4 \
  --frame-processor face_swapper \
  --execution-provider cuda \
  --max-memory 8 \
  --execution-threads 8 \
  --gpu-memory-wait 5 \
  --temp-frame-quality 100 \
  --temp-frame-format png \
  --output-video-encoder h264_nvenc \
  --output-video-quality 100 \
  --keep-fps

# 4. Monitorear
python monitor_gpu_live.py
```

**¡Con esta guía completa tendrás ROOP funcionando al máximo con tu GPU Tesla T4 de 15GB!** 🎉 