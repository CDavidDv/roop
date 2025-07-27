#!/usr/bin/env python3
"""
Script para procesar videos con ROOP mostrando progreso detallado y confirmación de GPU
"""

import os
import sys
import time
import subprocess
import psutil
import threading
from datetime import datetime

def get_gpu_info():
    """Obtener información de GPU"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'total': info.total / 1024**3,
            'used': info.used / 1024**3,
            'free': info.free / 1024**3,
            'utilization': pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        }
    except:
        return None

def monitor_gpu_usage():
    """Monitorear uso de GPU en tiempo real"""
    while True:
        gpu_info = get_gpu_info()
        if gpu_info:
            print(f"\r🎮 GPU: {gpu_info['used']:.1f}GB/{gpu_info['total']:.1f}GB "
                  f"({gpu_info['utilization']}% util) | "
                  f"💾 RAM: {psutil.virtual_memory().percent}%", end='', flush=True)
        time.sleep(2)

def process_single_video_with_progress(source_path, target_path, output_path, max_memory=12, execution_threads=30, temp_frame_quality=100):
    """Procesar un video mostrando progreso detallado"""
    print(f"\n🎬 PROCESANDO VIDEO CON PROGRESO DETALLADO")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"🎬 Target: {target_path}")
    print(f"💾 Output: {output_path}")
    print(f"🧠 Max Memory: {max_memory}GB")
    print(f"🧵 Threads: {execution_threads}")
    print(f"🎨 Quality: {temp_frame_quality}")
    print("=" * 60)
    
    # Iniciar monitoreo de GPU en segundo plano
    gpu_monitor = threading.Thread(target=monitor_gpu_usage, daemon=True)
    gpu_monitor.start()
    
    # Construir comando
    cmd = [
        sys.executable, 'run_roop_wrapper.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--max-memory', str(max_memory),
        '--execution-threads', str(execution_threads),
        '--temp-frame-quality', str(temp_frame_quality)
    ]
    
    print(f"\n🚀 EJECUTANDO COMANDO:")
    print(f"   {' '.join(cmd)}")
    print("\n" + "=" * 60)
    
    start_time = time.time()
    
    try:
        # Ejecutar con salida en tiempo real
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Mostrar salida en tiempo real
        for line in process.stdout:
            line = line.strip()
            if line:
                # Filtrar líneas importantes
                if any(keyword in line.lower() for keyword in [
                    'frame', 'processing', 'gpu', 'memory', 'progress', 
                    'face', 'swap', 'enhance', 'video', 'output'
                ]):
                    print(f"📊 {line}")
                elif 'error' in line.lower() or 'exception' in line.lower():
                    print(f"❌ {line}")
                elif 'warning' in line.lower():
                    print(f"⚠️ {line}")
        
        process.wait()
        
        if process.returncode == 0:
            elapsed_time = time.time() - start_time
            print(f"\n✅ VIDEO PROCESADO EXITOSAMENTE")
            print(f"⏱️ Tiempo total: {elapsed_time:.1f} segundos")
            return True
        else:
            print(f"\n❌ ERROR PROCESANDO VIDEO")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR EJECUTANDO COMANDO: {e}")
        return False

def process_folder_batch_with_progress(source_path, input_folder, output_dir, max_memory=12, execution_threads=30, temp_frame_quality=100):
    """Procesar carpeta completa con progreso detallado"""
    print("🚀 PROCESAMIENTO DE CARPETA CON PROGRESO DETALLADO")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"📁 Input Folder: {input_folder}")
    print(f"📁 Output Dir: {output_dir}")
    print(f"🧠 Max Memory: {max_memory}GB")
    print(f"🧵 Threads: {execution_threads}")
    print(f"🎨 Quality: {temp_frame_quality}")
    print("=" * 60)
    
    # Verificar que existe la carpeta de entrada
    if not os.path.exists(input_folder):
        print(f"❌ Error: Carpeta de entrada no encontrada: {input_folder}")
        return False
    
    # Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ Carpeta de salida creada/verificada: {output_dir}")
    
    # Encontrar videos
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    videos = []
    
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            videos.append(file)
    
    if not videos:
        print(f"❌ No se encontraron videos en: {input_folder}")
        return False
    
    print(f"\n🎬 VIDEOS ENCONTRADOS: {len(videos)}")
    for i, video in enumerate(videos, 1):
        print(f"  {i}. {video}")
    
    print("\n" + "=" * 60)
    
    # Procesar cada video
    successful = 0
    failed = 0
    
    for i, video in enumerate(videos, 1):
        print(f"\n📹 PROCESANDO VIDEO {i}/{len(videos)}")
        print("=" * 40)
        
        input_path = os.path.join(input_folder, video)
        output_filename = f"DanielaAS_{video}"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"🎬 Video: {video}")
        print(f"📸 Source: {source_path}")
        print(f"💾 Output: {output_filename}")
        print("=" * 40)
        
        # Procesar video
        if process_single_video_with_progress(
            source_path, input_path, output_path,
            max_memory, execution_threads, temp_frame_quality
        ):
            successful += 1
            print(f"✅ Video {i}/{len(videos)} completado: {output_filename}")
        else:
            failed += 1
            print(f"❌ Error en video {i}/{len(videos)}: {video}")
        
        # Esperar entre videos para liberar memoria
        if i < len(videos):
            print(f"\n⏳ Esperando 5 segundos entre videos...")
            time.sleep(5)
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL")
    print("=" * 60)
    print(f"✅ Videos procesados exitosamente: {successful}")
    print(f"❌ Videos con errores: {failed}")
    print(f"📁 Carpeta de salida: {output_dir}")
    print("=" * 60)
    
    return successful > 0

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ROOP con progreso detallado y monitoreo GPU')
    parser.add_argument('--source', required=True, help='Ruta de la imagen fuente')
    parser.add_argument('--input-folder', help='Carpeta con videos de entrada')
    parser.add_argument('--output-dir', help='Carpeta de salida')
    parser.add_argument('--target', help='Video específico a procesar')
    parser.add_argument('-o', '--output', help='Archivo de salida específico')
    parser.add_argument('--max-memory', type=int, default=12, help='Memoria máxima en GB')
    parser.add_argument('--execution-threads', type=int, default=30, help='Número de threads')
    parser.add_argument('--temp-frame-quality', type=int, default=100, help='Calidad de frames temporales')
    
    args = parser.parse_args()
    
    # Verificar argumentos
    if not os.path.exists(args.source):
        print(f"❌ Error: Imagen fuente no encontrada: {args.source}")
        return False
    
    # Procesar video específico
    if args.target and args.output:
        if not os.path.exists(args.target):
            print(f"❌ Error: Video objetivo no encontrado: {args.target}")
            return False
        
        return process_single_video_with_progress(
            args.source, args.target, args.output,
            args.max_memory, args.execution_threads, args.temp_frame_quality
        )
    
    # Procesar carpeta completa
    elif args.input_folder and args.output_dir:
        return process_folder_batch_with_progress(
            args.source, args.input_folder, args.output_dir,
            args.max_memory, args.execution_threads, args.temp_frame_quality
        )
    
    else:
        print("❌ Error: Debes especificar --target y -o para un video, o --input-folder y --output-dir para carpeta")
        return False

if __name__ == '__main__':
    main() 