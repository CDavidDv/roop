#!/usr/bin/env python3
"""
Script para procesar múltiples videos automáticamente con ROOP
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
    # Crear nombre de salida: SakuraAS + número del video
    output_name = f"{source_name}{target_base}.mp4"
    return output_name

def process_single_video(source_path: str, target_path: str, output_path: str, 
                        gpu_memory_wait: int, max_memory: int, 
                        execution_threads: int, temp_frame_quality: int,
                        keep_fps: bool) -> bool:
    """Procesar un solo video con progreso detallado"""
    
    print(f"\n🎬 PROCESANDO VIDEO: {os.path.basename(target_path)}")
    print(f"📸 Source: {os.path.basename(source_path)}")
    print(f"💾 Output: {os.path.basename(output_path)}")
    print("=" * 60)
    
    # Construir comando
    cmd = [
        sys.executable, 'run.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--gpu-memory-wait', str(gpu_memory_wait),
        '--max-memory', str(max_memory),
        '--execution-threads', str(execution_threads),
        '--temp-frame-quality', str(temp_frame_quality)
    ]
    
    if keep_fps:
        cmd.append('--keep-fps')
    
    try:
        print("🔄 Iniciando procesamiento...")
        print(f"⚙️ Configuración: {execution_threads} hilos, {max_memory}GB RAM, {gpu_memory_wait}s GPU wait")
        print("📊 Progreso en tiempo real:")
        print("-" * 40)
        
        # Ejecutar comando con salida en tiempo real
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Mostrar salida en tiempo real
        output_lines = []
        for line in process.stdout:
            line = line.strip()
            output_lines.append(line)
            if line:
                # Filtrar solo mensajes importantes de progreso
                if any(keyword in line for keyword in [
                    'Progressing', 'Creating', 'Extracting', 'Restoring', 
                    'Cleaning', 'Processing', 'Creating video', 'Extracting frames',
                    'Face-Swapper', 'Face-Enhancer', 'ROOP.CORE'
                ]):
                    print(f"  📈 {line}")

        # Esperar a que termine el proceso
        return_code = process.wait()

        if return_code == 0:
            print("-" * 40)
            print(f"✅ Video procesado exitosamente: {os.path.basename(output_path)}")
            return True
        else:
            print(f"❌ Error en el procesamiento (código: {return_code})")
            print("---- SALIDA COMPLETA DEL PROCESO ----")
            for l in output_lines:
                print(l)
            if process.stderr:
                print("---- STDERR ----")
                print(process.stderr.read())
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error procesando {target_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def process_video_batch(source_path: str, target_videos: list, output_dir: str,
                       gpu_memory_wait: int, max_memory: int,
                       execution_threads: int, temp_frame_quality: int,
                       keep_fps: bool) -> None:
    """Procesar lote de videos con progreso detallado"""
    
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"🎬 Videos a procesar: {len(target_videos)}")
    print(f"⚙️ Configuración:")
    print(f"   • GPU Memory Wait: {gpu_memory_wait}s")
    print(f"   • Max Memory: {max_memory}GB")
    print(f"   • Execution Threads: {execution_threads}")
    print(f"   • Temp Frame Quality: {temp_frame_quality}")
    print(f"   • Keep FPS: {keep_fps}")
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
    total_start_time = time.time()
    
    for i, target_video in enumerate(target_videos, 1):
        print(f"\n📊 PROGRESO GENERAL: {i}/{len(target_videos)} ({(i/len(target_videos)*100):.1f}%)")
        print(f"⏱️ Tiempo transcurrido: {time.time() - total_start_time:.1f}s")
        print(f"✅ Completados: {successful} | ❌ Fallidos: {failed}")
        
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
        if i < len(target_videos):
            print(f"\n⏳ Esperando 10 segundos antes del siguiente video...")
            for countdown in range(10, 0, -1):
                print(f"   ⏰ {countdown} segundos restantes...", end='\r')
                time.sleep(1)
            print()
    
    # Resumen final
    total_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print("📊 RESUMEN FINAL")
    print("=" * 60)
    print(f"✅ Videos procesados exitosamente: {successful}")
    print(f"❌ Videos fallidos: {failed}")
    print(f"📈 Tasa de éxito: {(successful/(successful+failed)*100):.1f}%")
    print(f"⏱️ Tiempo total: {total_time:.2f} segundos")
    print(f"📊 Tiempo promedio por video: {total_time/len(target_videos):.2f} segundos")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description='Procesar múltiples videos con ROOP')
    parser.add_argument('--source', required=True, help='Imagen fuente')
    parser.add_argument('--videos', nargs='+', required=True, help='Lista de videos a procesar')
    parser.add_argument('--output-dir', required=True, help='Directorio de salida')
    parser.add_argument('--gpu-memory-wait', type=int, default=30, 
                       help='Tiempo de espera entre procesadores (segundos, default: 30)')
    parser.add_argument('--max-memory', type=int, default=8, 
                       help='Memoria máxima en GB (default: 8)')
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
        gpu_memory_wait=args.gpu_memory_wait,
        max_memory=args.max_memory,
        execution_threads=args.execution_threads,
        temp_frame_quality=args.temp_frame_quality,
        keep_fps=args.keep_fps
    )

if __name__ == "__main__":
    main() 