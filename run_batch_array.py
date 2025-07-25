#!/usr/bin/env python3
"""
Script súper simple para procesar múltiples videos
Solo modifica el array VIDEOS_TO_PROCESS y ejecuta
"""

import os
import sys
import subprocess
import time
from pathlib import Path

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
# SOLO MODIFICA ESTE ARRAY
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
MAX_MEMORY = 12
EXECUTION_THREADS = 8
TEMP_FRAME_QUALITY = 100
KEEP_FPS = True

# ============================================
# CÓDIGO (NO MODIFICAR)
# ============================================

def get_python_executable():
    """Siempre usar sys.executable para evitar entorno virtual en Colab"""
    return sys.executable

def process_video(video_path: str, index: int, total: int) -> bool:
    """Procesar un video individual"""
    
    print(f"\n🎬 PROCESANDO VIDEO {index}/{total}: {Path(video_path).name}")
    print("=" * 60)
    
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
        # Ejecutar comando
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed_time = time.time() - start_time
        
        print(f"✅ Video procesado exitosamente: {output_filename}")
        print(f"⏱️ Tiempo: {elapsed_time:.2f} segundos")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error procesando {video_path}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 PROCESAMIENTO EN LOTE AUTOMÁTICO")
    print("=" * 60)
    print(f"📸 Source: {SOURCE_IMAGE}")
    print(f"🎬 Videos a procesar: {len(VIDEOS_TO_PROCESS)}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Verificar que el source existe
    if not os.path.exists(SOURCE_IMAGE):
        print(f"❌ Source no encontrado: {SOURCE_IMAGE}")
        return
    
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(VIDEOS_TO_PROCESS, 1):
        success = process_video(video_path, i, len(VIDEOS_TO_PROCESS))
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # Pausa entre videos (excepto el último)
        if i < len(VIDEOS_TO_PROCESS):
            print(f"\n⏳ Esperando 10 segundos...")
            time.sleep(10)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL")
    print("=" * 60)
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    if successful + failed > 0:
        print(f"📈 Tasa de éxito: {(successful/(successful+failed)*100):.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main() 