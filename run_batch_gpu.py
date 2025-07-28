#!/usr/bin/env python3
"""
Script optimizado para procesar múltiples videos con ROOP usando GPU
Sin entorno virtual - directo con Python
"""

import os
import sys
import argparse
import time
import glob
from pathlib import Path
import warnings

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Importar ROOP después de configurar variables de entorno
import roop.globals
import roop.metadata
from roop.processors.frame.core import process_video_with_memory_management
from roop.utilities import normalize_output_path, is_video, is_image

def check_file_exists(file_path: str, file_type: str) -> bool:
    """Verificar si un archivo existe"""
    if not os.path.exists(file_path):
        print(f"❌ {file_type} no encontrado: {file_path}")
        return False
    return True

def get_output_filename(source_name: str, target_name: str) -> str:
    """Generar nombre de archivo de salida"""
    target_base = Path(target_name).stem
    output_name = f"{source_name}_{target_base}.mp4"
    return output_name

def setup_roop_config(source_path: str, target_path: str, output_path: str,
                     frame_processors: list = ['face_swapper', 'face_enhancer'],
                     keep_fps: bool = True, max_memory: int = 12,
                     execution_threads: int = 8, temp_frame_quality: int = 100,
                     gpu_memory_wait: int = 30):
    """Configurar variables globales de ROOP"""
    
    roop.globals.source_path = source_path
    roop.globals.target_path = target_path
    roop.globals.output_path = normalize_output_path(source_path, target_path, output_path)
    roop.globals.headless = True
    roop.globals.frame_processors = frame_processors
    roop.globals.keep_fps = keep_fps
    roop.globals.keep_frames = False
    roop.globals.skip_audio = False
    roop.globals.many_faces = False
    roop.globals.reference_face_position = 0
    roop.globals.reference_frame_number = 0
    roop.globals.similar_face_distance = 0.85
    roop.globals.temp_frame_format = 'png'
    roop.globals.temp_frame_quality = temp_frame_quality
    roop.globals.output_video_encoder = 'libx264'
    roop.globals.output_video_quality = 35
    roop.globals.max_memory = max_memory
    roop.globals.execution_providers = ['CUDAExecutionProvider']
    roop.globals.execution_threads = execution_threads
    roop.globals.gpu_memory_wait_time = gpu_memory_wait

def process_single_video(source_path: str, target_path: str, output_path: str,
                        frame_processors: list = ['face_swapper', 'face_enhancer'],
                        keep_fps: bool = True, max_memory: int = 12,
                        execution_threads: int = 8, temp_frame_quality: int = 100,
                        gpu_memory_wait: int = 30) -> bool:
    """Procesar un solo video usando ROOP directamente"""
    
    print(f"\n🎬 PROCESANDO: {target_path}")
    print(f"📸 Source: {source_path}")
    print(f"💾 Output: {output_path}")
    print("=" * 60)
    
    try:
        # Configurar ROOP
        setup_roop_config(
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            frame_processors=frame_processors,
            keep_fps=keep_fps,
            max_memory=max_memory,
            execution_threads=execution_threads,
            temp_frame_quality=temp_frame_quality,
            gpu_memory_wait=gpu_memory_wait
        )
        
        # Procesar video directamente
        process_video_with_memory_management()
        
        print(f"✅ Video procesado exitosamente: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error procesando {target_path}: {str(e)}")
        return False

def get_videos_from_folder(folder_path: str) -> list:
    """Obtener todos los videos de una carpeta"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    videos = []
    
    for ext in video_extensions:
        videos.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        videos.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    
    return sorted(videos)

def process_video_batch(source_path: str, input_folder: str, output_folder: str,
                       frame_processors: list = ['face_swapper', 'face_enhancer'],
                       keep_fps: bool = True, max_memory: int = 12,
                       execution_threads: int = 8, temp_frame_quality: int = 100,
                       gpu_memory_wait: int = 30) -> None:
    """Procesar lote de videos desde carpetas"""
    
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE CON GPU")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"📁 Input Folder: {input_folder}")
    print(f"📁 Output Folder: {output_folder}")
    print(f"🎨 Frame Processors: {', '.join(frame_processors)}")
    print(f"⏰ GPU Memory Wait: {gpu_memory_wait}s")
    print(f"🧠 Max Memory: {max_memory}GB")
    print(f"🧵 Threads: {execution_threads}")
    print(f"🎨 Quality: {temp_frame_quality}")
    print(f"🎯 Keep FPS: {keep_fps}")
    print("=" * 60)
    
    # Verificar que el source existe
    if not check_file_exists(source_path, "Source"):
        return
    
    # Verificar que la carpeta de entrada existe
    if not os.path.exists(input_folder):
        print(f"❌ Carpeta de entrada no encontrada: {input_folder}")
        return
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Carpeta de salida creada: {output_folder}")
    
    # Obtener videos de la carpeta
    target_videos = get_videos_from_folder(input_folder)
    
    if not target_videos:
        print(f"❌ No se encontraron videos en: {input_folder}")
        return
    
    print(f"🎬 Videos encontrados: {len(target_videos)}")
    
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
        output_path = os.path.join(output_folder, output_filename)
        
        # Procesar video
        start_time = time.time()
        success = process_single_video(
            source_path=source_path,
            target_path=target_video,
            output_path=output_path,
            frame_processors=frame_processors,
            keep_fps=keep_fps,
            max_memory=max_memory,
            execution_threads=execution_threads,
            temp_frame_quality=temp_frame_quality,
            gpu_memory_wait=gpu_memory_wait
        )
        
        if success:
            successful += 1
            elapsed_time = time.time() - start_time
            print(f"⏱️ Tiempo de procesamiento: {elapsed_time:.2f} segundos")
        else:
            failed += 1
        
        # Pausa entre videos para liberar memoria GPU
        if i < len(target_videos):
            print(f"\n⏳ Esperando {gpu_memory_wait} segundos para liberar memoria GPU...")
            time.sleep(gpu_memory_wait)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL")
    print("=" * 60)
    print(f"✅ Videos procesados exitosamente: {successful}")
    print(f"❌ Videos fallidos: {failed}")
    if successful + failed > 0:
        print(f"📈 Tasa de éxito: {(successful/(successful+failed)*100):.1f}%")
    print(f"📁 Videos guardados en: {output_folder}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Procesar videos con ROOP usando GPU')
    parser.add_argument('--source', required=True, help='Imagen fuente')
    parser.add_argument('--input-folder', required=True, help='Carpeta con videos a procesar')
    parser.add_argument('--output-folder', required=True, help='Carpeta para guardar resultados')
    parser.add_argument('--frame-processors', nargs='+', 
                       default=['face_swapper', 'face_enhancer'],
                       help='Procesadores de frames (default: face_swapper face_enhancer)')
    parser.add_argument('--gpu-memory-wait', type=int, default=30, 
                       help='Tiempo de espera entre videos (segundos, default: 30)')
    parser.add_argument('--max-memory', type=int, default=12, 
                       help='Memoria máxima en GB (default: 12)')
    parser.add_argument('--execution-threads', type=int, default=8, 
                       help='Número de hilos (default: 8)')
    parser.add_argument('--temp-frame-quality', type=int, default=100, 
                       help='Calidad de frames temporales (default: 100)')
    parser.add_argument('--keep-fps', action='store_true', 
                       help='Mantener FPS original')
    
    args = parser.parse_args()
    
    # Procesar lote de videos
    process_video_batch(
        source_path=args.source,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        frame_processors=args.frame_processors,
        keep_fps=args.keep_fps,
        max_memory=args.max_memory,
        execution_threads=args.execution_threads,
        temp_frame_quality=args.temp_frame_quality,
        gpu_memory_wait=args.gpu_memory_wait
    )

if __name__ == "__main__":
    main() 