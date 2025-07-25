#!/usr/bin/env python3
"""
Script simple para procesar múltiples videos con array predefinido
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
# CONFIGURACIÓN - MODIFICA AQUÍ
# ============================================

# Imagen fuente
SOURCE_IMAGE = "/content/SakuraAS.png"

# Array de videos a procesar
VIDEOS_TO_PROCESS = [
    "/content/17.mp4",
    "/content/18.mp4", 
    "/content/19.mp4",
    "/content/20.mp4"
]

# Directorio de salida (opcional, dejar None para usar directorio actual)
OUTPUT_DIR = "/content/resultados"

# Configuración de procesamiento
GPU_MEMORY_WAIT = 30  # segundos entre procesadores
MAX_MEMORY = 12        # GB
EXECUTION_THREADS = 8  # hilos
TEMP_FRAME_QUALITY = 100
KEEP_FPS = True

# ============================================
# NO MODIFICAR DESDE AQUÍ
# ============================================

def check_file_exists(file_path: str, file_type: str) -> bool:
    """Verificar si un archivo existe"""
    if not os.path.exists(file_path):
        print(f"❌ {file_type} no encontrado: {file_path}")
        return False
    return True

def get_output_filename(source_name: str, target_name: str) -> str:
    """Generar nombre de archivo de salida"""
    target_base = Path(target_name).stem
    output_name = f"{source_name}{target_base}.mp4"
    return output_name

def process_single_video(source_path: str, target_path: str, output_path: str) -> bool:
    """Procesar un solo video"""
    
    print(f"\n🎬 PROCESANDO: {target_path}")
    print(f"📸 Source: {source_path}")
    print(f"💾 Output: {output_path}")
    print("=" * 60)
    
    # Construir comando
    cmd = [
        sys.executable, 'run.py',
        '--source', source_path,
        '--target', target_path,
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Video procesado exitosamente: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error procesando {target_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE")
    print("=" * 60)
    print(f"📸 Source: {SOURCE_IMAGE}")
    print(f"🎬 Videos a procesar: {len(VIDEOS_TO_PROCESS)}")
    print(f"⏰ GPU Memory Wait: {GPU_MEMORY_WAIT}s")
    print(f"🧠 Max Memory: {MAX_MEMORY}GB")
    print(f"🧵 Threads: {EXECUTION_THREADS}")
    print(f"🎨 Quality: {TEMP_FRAME_QUALITY}")
    print(f"🎯 Keep FPS: {KEEP_FPS}")
    print("=" * 60)
    
    # Verificar que el source existe
    if not check_file_exists(SOURCE_IMAGE, "Source"):
        return
    
    # Crear directorio de salida si no existe
    if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Directorio creado: {OUTPUT_DIR}")
    
    # Extraer nombre base del source
    source_name = Path(SOURCE_IMAGE).stem
    
    successful = 0
    failed = 0
    
    for i, target_video in enumerate(VIDEOS_TO_PROCESS, 1):
        print(f"\n📊 Progreso: {i}/{len(VIDEOS_TO_PROCESS)}")
        
        # Verificar que el video existe
        if not check_file_exists(target_video, "Video"):
            failed += 1
            continue
        
        # Generar nombre de salida
        output_filename = get_output_filename(source_name, target_video)
        if OUTPUT_DIR:
            output_path = os.path.join(OUTPUT_DIR, output_filename)
        else:
            output_path = output_filename
        
        # Procesar video
        start_time = time.time()
        success = process_single_video(
            source_path=SOURCE_IMAGE,
            target_path=target_video,
            output_path=output_path
        )
        
        if success:
            successful += 1
            elapsed_time = time.time() - start_time
            print(f"⏱️ Tiempo de procesamiento: {elapsed_time:.2f} segundos")
        else:
            failed += 1
        
        # Pausa entre videos para liberar memoria
        if i < len(VIDEOS_TO_PROCESS):
            print(f"\n⏳ Esperando 10 segundos antes del siguiente video...")
            time.sleep(10)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL")
    print("=" * 60)
    print(f"✅ Videos procesados exitosamente: {successful}")
    print(f"❌ Videos fallidos: {failed}")
    if successful + failed > 0:
        print(f"📈 Tasa de éxito: {(successful/(successful+failed)*100):.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main() 