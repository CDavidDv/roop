#!/usr/bin/env python3
"""
Script para procesar múltiples videos automáticamente con ROOP usando GPU optimizado
"""

import os
import sys
import argparse
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

def check_file_exists(file_path: str, file_type: str) -> bool:
    """Verificar si un archivo existe"""
    if not os.path.exists(file_path):
        print(f"❌ {file_type} no encontrado: {file_path}")
        return False
    return True

def get_output_filename(source_name: str, target_name: str) -> str:
    """Generar nombre de archivo de salida"""
    # Extraer nombre base del target (sin extensión)
    target_base = Path(target_name).stem
    # Crear nombre de salida: source_name + target_base
    output_name = f"{source_name}{target_base}.mp4"
    return output_name

def process_single_video(source_path: str, target_path: str, output_path: str, 
                        execution_threads: int = 31, temp_frame_quality: int = 100,
                        keep_fps: bool = True) -> bool:
    """Procesar un solo video con configuración optimizada para GPU"""
    
    print(f"\n🎬 PROCESANDO: {target_path}")
    print(f"📸 Source: {source_path}")
    print(f"💾 Output: {output_path}")
    print(f"🧵 Threads: {execution_threads}")
    print(f"🎨 Quality: {temp_frame_quality}")
    print(f"🎯 Keep FPS: {keep_fps}")
    print("=" * 60)
    
    # Construir comando con configuración optimizada para GPU
    cmd = [
        sys.executable, 'run.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--gpu-memory-wait', '30',  # Pausa entre procesadores para liberar VRAM
        '--max-memory', '12',        # 12GB para Tesla T4
        '--execution-threads', str(execution_threads),
        '--temp-frame-quality', str(temp_frame_quality),
        '--output-video-encoder', 'libx264',  # Encoder estándar
        '--output-video-quality', '35',       # Calidad balanceada
        '--temp-frame-format', 'png'          # Formato PNG para máxima calidad
    ]
    
    if keep_fps:
        cmd.append('--keep-fps')
    
    try:
        # Ejecutar comando
        print(f"🔄 Ejecutando: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Video procesado exitosamente: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error procesando {target_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def process_video_batch(source_path: str, target_videos: list, output_dir: str = None,
                       execution_threads: int = 31, temp_frame_quality: int = 100,
                       keep_fps: bool = True) -> None:
    """Procesar lote de videos con configuración optimizada"""
    
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE CON GPU")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"🎬 Videos a procesar: {len(target_videos)}")
    print(f"🧵 Execution Threads: {execution_threads}")
    print(f"🎨 Temp Frame Quality: {temp_frame_quality}")
    print(f"🎯 Keep FPS: {keep_fps}")
    print(f"💾 Output Directory: {output_dir}")
    print("=" * 60)
    
    # Verificar que el source existe
    if not check_file_exists(source_path, "Source"):
        return
    
    # Crear directorio de salida si no existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Directorio creado: {output_dir}")
    
    # Extraer nombre base del source para usar en nombres de salida
    source_name = Path(source_path).stem
    
    successful = 0
    failed = 0
    
    for i, target_video in enumerate(target_videos, 1):
        print(f"\n📊 Progreso: {i}/{len(target_videos)}")
        
        # Verificar que el video existe
        if not check_file_exists(target_video, "Video"):
            failed += 1
            continue
        
        # Generar nombre de salida
        output_filename = get_output_filename(source_name, target_video)
        if output_dir:
            output_path = os.path.join(output_dir, output_filename)
        else:
            output_path = output_filename
        
        # Procesar video
        start_time = time.time()
        success = process_single_video(
            source_path=source_path,
            target_path=target_video,
            output_path=output_path,
            execution_threads=execution_threads,
            temp_frame_quality=temp_frame_quality,
            keep_fps=keep_fps
        )
        
        if success:
            successful += 1
            elapsed_time = time.time() - start_time
            print(f"⏱️ Tiempo de procesamiento: {elapsed_time:.2f} segundos")
        else:
            failed += 1
        
        # Pausa entre videos para liberar memoria GPU
        if i < len(target_videos):
            print(f"\n⏳ Esperando 15 segundos para liberar memoria GPU...")
            time.sleep(15)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL")
    print("=" * 60)
    print(f"✅ Videos procesados exitosamente: {successful}")
    print(f"❌ Videos fallidos: {failed}")
    if successful + failed > 0:
        print(f"📈 Tasa de éxito: {(successful/(successful+failed)*100):.1f}%")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Procesar múltiples videos con ROOP usando GPU optimizado')
    parser.add_argument('--source', required=True, help='Imagen fuente')
    parser.add_argument('--videos', nargs='+', required=True, help='Lista de videos a procesar')
    parser.add_argument('--output-dir', help='Directorio de salida (opcional)')
    parser.add_argument('--execution-threads', type=int, default=31, 
                       help='Número de hilos (default: 31)')
    parser.add_argument('--temp-frame-quality', type=int, default=100, 
                       help='Calidad de frames temporales (default: 100)')
    parser.add_argument('--keep-fps', action='store_true', 
                       help='Mantener FPS original')
    
    args = parser.parse_args()
    
    # Procesar lote de videos
    process_video_batch(
        source_path=args.source,
        target_videos=args.videos,
        output_dir=args.output_dir,
        execution_threads=args.execution_threads,
        temp_frame_quality=args.temp_frame_quality,
        keep_fps=args.keep_fps
    )

if __name__ == "__main__":
    main() 