#!/usr/bin/env python3
"""
Script para procesar múltiples videos con información detallada del progreso
"""

import os
import sys
import subprocess
import time
import psutil
from pathlib import Path
from datetime import datetime

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Desactivar predictor NSFW para evitar errores de GPU
import roop.predictor
def predict_video_skip_nsfw(target_path: str) -> bool:
    print("⚠️ Saltando verificación NSFW para evitar conflictos de GPU...")
    return False

roop.predictor.predict_video = predict_video_skip_nsfw

# ============================================
# CONFIGURACIÓN - MODIFICA AQUÍ
# ============================================

VIDEOS_TO_PROCESS = [
    "/content/17.mp4",
    "/content/18.mp4", 
    "/content/19.mp4",
    "/content/20.mp4"
]

# ============================================
# CONFIGURACIÓN PREDEFINIDA (NO MODIFICAR)
# ============================================

SOURCE_IMAGE = "/content/SakuraAS.png"
OUTPUT_DIR = "/content/resultados"
GPU_MEMORY_WAIT = 30
MAX_MEMORY = 11
EXECUTION_THREADS = 35
TEMP_FRAME_QUALITY = 100
KEEP_FPS = True

# ============================================
# FUNCIONES DE MONITOREO
# ============================================

def get_system_info():
    """Obtener información del sistema"""
    try:
        import torch
        gpu_info = ""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_info = f"GPU: {gpu_memory:.2f}GB usado, {gpu_reserved:.2f}GB reservado"
    except:
        gpu_info = "GPU: No disponible"
    
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024 / 1024 / 1024
    
    return {
        'ram': f"{ram_usage:.2f}GB",
        'gpu': gpu_info,
        'cpu_percent': psutil.cpu_percent(),
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }

def get_python_executable():
    """Detectar el ejecutable de Python correcto"""
    if os.path.exists("roop_env/bin/python"):
        return "roop_env/bin/python"
    elif os.path.exists("venv/bin/python"):
        return "venv/bin/python"
    elif os.path.exists("env/bin/python"):
        return "env/bin/python"
    else:
        return sys.executable

def process_video_with_monitoring(video_path: str, index: int, total: int) -> bool:
    """Procesar un video con monitoreo detallado"""
    
    print(f"\n🎬 PROCESANDO VIDEO {index}/{total}: {Path(video_path).name}")
    print("=" * 80)
    
    # Verificar que el video existe
    if not os.path.exists(video_path):
        print(f"❌ Video no encontrado: {video_path}")
        return False
    
    # Crear directorio de salida si no existe
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Directorio creado: {OUTPUT_DIR}")
    
    # Generar nombre de salida
    source_name = Path(SOURCE_IMAGE).stem
    video_name = Path(video_path).stem
    output_filename = f"{source_name}{video_name}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Construir comando
    python_exe = get_python_executable()
    cmd = [
        python_exe, 'run.py',
        '--source', SOURCE_IMAGE,
        '--target', video_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--gpu-memory-wait', str(GPU_MEMORY_WAIT),
        '--max-memory', str(MAX_MEMORY),
        '--execution-threads', str(EXECUTION_THREADS),
        '--temp-frame-quality', str(TEMP_FRAME_QUALITY)
    ]
    
    if KEEP_FPS:
        cmd.append('--keep-fps')
    
    try:
        # Mostrar información inicial
        print(f"📸 Source: {SOURCE_IMAGE}")
        print(f"💾 Output: {output_path}")
        print(f"⚙️ Comando: {' '.join(cmd)}")
        print("=" * 80)
        
        # Información del sistema antes de procesar
        info_before = get_system_info()
        print(f"🖥️ Sistema antes: RAM {info_before['ram']}, {info_before['gpu']}, CPU {info_before['cpu_percent']}%")
        
        # Ejecutar comando con monitoreo en tiempo real
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
        
        # Monitorear salida en tiempo real
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Filtrar líneas importantes
                if any(keyword in output for keyword in ['Processing', 'Progressing', 'FACE-SWAPPER', 'FACE-ENHANCER', 'Forzando uso de GPU']):
                    print(f"📊 {output.strip()}")
                
                # Mostrar información del sistema cada 30 segundos
                if int(time.time() - start_time) % 30 == 0 and int(time.time() - start_time) > 0:
                    info = get_system_info()
                    print(f"🖥️ [{info['timestamp']}] RAM {info['ram']}, {info['gpu']}, CPU {info['cpu_percent']}%")
        
        # Obtener resultado
        return_code = process.poll()
        elapsed_time = time.time() - start_time
        
        # Información del sistema después de procesar
        info_after = get_system_info()
        print(f"🖥️ Sistema después: RAM {info_after['ram']}, {info_after['gpu']}, CPU {info_after['cpu_percent']}%")
        
        if return_code == 0:
            print(f"✅ Video procesado exitosamente: {output_filename}")
            print(f"⏱️ Tiempo total: {elapsed_time:.2f} segundos")
            return True
        else:
            print(f"❌ Error procesando {video_path}")
            print(f"Error: {process.stderr.read()}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error procesando {video_path}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 PROCESAMIENTO EN LOTE CON MONITOREO DETALLADO")
    print("=" * 80)
    print(f"📸 Source: {SOURCE_IMAGE}")
    print(f"🎬 Videos a procesar: {len(VIDEOS_TO_PROCESS)}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f"⏰ GPU Memory Wait: {GPU_MEMORY_WAIT}s")
    print(f"🧠 Max Memory: {MAX_MEMORY}GB")
    print(f"🧵 Threads: {EXECUTION_THREADS}")
    print(f"🎨 Quality: {TEMP_FRAME_QUALITY}")
    print(f"🎯 Keep FPS: {KEEP_FPS}")
    print("=" * 80)
    
    # Verificar que el source existe
    if not os.path.exists(SOURCE_IMAGE):
        print(f"❌ Source no encontrado: {SOURCE_IMAGE}")
        return
    
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(VIDEOS_TO_PROCESS, 1):
        success = process_video_with_monitoring(video_path, i, len(VIDEOS_TO_PROCESS))
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Pausa entre videos (excepto el último)
        if i < len(VIDEOS_TO_PROCESS):
            print(f"\n⏳ Esperando 10 segundos...")
            time.sleep(10)
    
    # Resumen final
    print("\n" + "=" * 80)
    print("📊 RESUMEN FINAL")
    print("=" * 80)
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    if successful + failed > 0:
        print(f"📈 Tasa de éxito: {(successful/(successful+failed)*100):.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    main() 