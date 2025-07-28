#!/usr/bin/env python3
"""
Script simplificado para Google Colab - Procesamiento directo sin GUI
"""

import os
import sys
import subprocess
import time
import glob
import argparse
from pathlib import Path

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_video_files_from_folder(folder_path: str) -> list:
    """Obtener todos los archivos de video de una carpeta"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    video_files = []
    
    for ext in video_extensions:
        pattern = os.path.join(folder_path, ext)
        video_files.extend(glob.glob(pattern))
        pattern_upper = os.path.join(folder_path, ext.upper())
        video_files.extend(glob.glob(pattern_upper))
    
    return sorted(video_files)

def get_output_filename(source_name: str, target_name: str) -> str:
    """Generar nombre de archivo de salida"""
    target_base = Path(target_name).stem
    output_name = f"{source_name}_{target_base}.mp4"
    return output_name

def process_video_direct(source_path: str, target_path: str, output_path: str, 
                        gpu_memory_wait: int = 30, max_memory: int = 12, 
                        execution_threads: int = 30, temp_frame_quality: int = 100,
                        keep_fps: bool = True) -> bool:
    """Procesar un solo video directamente sin GUI"""
    
    print(f"\n🎬 PROCESANDO: {target_path}")
    print(f"📸 Source: {source_path}")
    print(f"💾 Output: {output_path}")
    print("=" * 60)
    
    # Construir comando optimizado para calidad máxima
    cmd = [
        "python", 'run.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--gpu-memory-wait', str(gpu_memory_wait),
        '--max-memory', str(max_memory),
        '--execution-threads', str(execution_threads),
        '--temp-frame-quality', str(temp_frame_quality),
        '--output-quality', '100',
        '--temp-frame-format', 'png',
        '--headless'  # Modo sin interfaz gráfica
    ]
    
    if keep_fps:
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
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el comando 'python'")
        return False

def process_videos_batch(source_path: str, input_folder: str, output_folder: str,
                        gpu_memory_wait: int = 30, max_memory: int = 12,
                        execution_threads: int = 30, temp_frame_quality: int = 100,
                        keep_fps: bool = True) -> None:
    """Procesar todos los videos de una carpeta"""
    
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"📁 Carpeta de entrada: {input_folder}")
    print(f"📁 Carpeta de salida: {output_folder}")
    print(f"⏰ GPU Memory Wait: {gpu_memory_wait}s")
    print(f"🧠 Max Memory: {max_memory}GB")
    print(f"🧵 Threads: {execution_threads}")
    print(f"🎨 Quality: {temp_frame_quality}")
    print(f"🎯 Keep FPS: {keep_fps}")
    print("=" * 60)
    
    # Verificar que el source existe
    if not os.path.exists(source_path):
        print(f"❌ Source no encontrado: {source_path}")
        return
    
    # Verificar que la carpeta de entrada existe
    if not os.path.exists(input_folder):
        print(f"❌ Carpeta de entrada no encontrada: {input_folder}")
        return
    
    # Crear carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 Carpeta de salida creada: {output_folder}")
    
    # Obtener todos los videos de la carpeta
    video_files = get_video_files_from_folder(input_folder)
    
    if not video_files:
        print(f"❌ No se encontraron videos en: {input_folder}")
        return
    
    print(f"🎬 Videos encontrados: {len(video_files)}")
    
    # Extraer nombre base del source
    source_name = Path(source_path).stem
    
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n📊 Progreso: {i}/{len(video_files)}")
        
        # Generar nombre de salida
        output_filename = get_output_filename(source_name, video_file)
        output_path = os.path.join(output_folder, output_filename)
        
        # Procesar video
        start_time = time.time()
        success = process_video_direct(
            source_path=source_path,
            target_path=video_file,
            output_path=output_path,
            gpu_memory_wait=gpu_memory_wait,
            max_memory=max_memory,
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
        
        # Pausa entre videos para liberar memoria
        if i < len(video_files):
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

def main():
    """Función principal"""
    
    parser = argparse.ArgumentParser(description='Procesar videos con ROOP usando GPU (modo simple)')
    parser.add_argument('--source', default='/content/DanielaAS.jpg', 
                       help='Ruta de la imagen fuente (default: /content/DanielaAS.jpg)')
    parser.add_argument('--input-folder', default='/content/videos',
                       help='Carpeta con videos a procesar (default: /content/videos)')
    parser.add_argument('--output-folder', default='/content/resultados',
                       help='Carpeta para guardar resultados (default: /content/resultados)')
    parser.add_argument('--gpu-memory-wait', type=int, default=30,
                       help='Tiempo de espera entre procesadores (default: 30)')
    parser.add_argument('--max-memory', type=int, default=12,
                       help='Memoria máxima en GB (default: 12)')
    parser.add_argument('--execution-threads', type=int, default=30,
                       help='Número de hilos (default: 30)')
    parser.add_argument('--temp-frame-quality', type=int, default=100,
                       help='Calidad de frames temporales (default: 100)')
    parser.add_argument('--keep-fps', action='store_true', default=True,
                       help='Mantener FPS original (default: True)')
    
    args = parser.parse_args()
    
    print("🎯 PROCESAMIENTO SIMPLIFICADO PARA GOOGLE COLAB T4")
    print("=" * 60)
    print(f"📸 Source: {args.source}")
    print(f"📁 Input Folder: {args.input_folder}")
    print(f"📁 Output Folder: {args.output_folder}")
    print("=" * 60)
    
    # Procesar videos
    process_videos_batch(
        source_path=args.source,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        gpu_memory_wait=args.gpu_memory_wait,
        max_memory=args.max_memory,
        execution_threads=args.execution_threads,
        temp_frame_quality=args.temp_frame_quality,
        keep_fps=args.keep_fps
    )

if __name__ == "__main__":
    main() 