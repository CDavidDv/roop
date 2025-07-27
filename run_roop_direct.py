#!/usr/bin/env python3
"""
Script para ejecutar ROOP directamente sin módulo
Usa el código original de ROOP
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_gpu():
    """Configurar GPU"""
    print("🚀 CONFIGURANDO GPU PARA COLAB T4")
    print("=" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detectada: {torch.cuda.get_device_name()}")
            print(f"📊 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            torch.cuda.empty_cache()
            return True
        else:
            print("❌ GPU no disponible")
            return False
    except Exception as e:
        print(f"❌ Error configurando GPU: {e}")
        return False

def download_model():
    """Descargar modelo de face swap si no existe"""
    model_path = "inswapper_128.onnx"
    if not os.path.exists(model_path):
        print("📥 DESCARGANDO MODELO DE FACE SWAP:")
        print("=" * 40)
        
        try:
            cmd = [
                "wget", "https://civitai.com/api/download/models/85159", 
                "-O", model_path
            ]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Modelo descargado exitosamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error descargando modelo: {e}")
            return False
    else:
        print("✅ Modelo ya existe")
        return True

def process_single_video(source_path: str, target_path: str, output_path: str,
                        gpu_memory_wait: int = 30, max_memory: int = 12,
                        execution_threads: int = 8, temp_frame_quality: int = 100,
                        keep_fps: bool = True):
    """Procesar un solo video"""
    
    print(f"\n🎬 PROCESANDO: {target_path}")
    print(f"📸 Source: {source_path}")
    print(f"💾 Output: {output_path}")
    print("=" * 60)
    
    # Construir comando para ROOP original
    cmd = [
        sys.executable, 'run.py',
        '--source', source_path,
        '--target', target_path,
        '-o', output_path,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--execution-provider', 'cuda',
        '--max-memory', str(max_memory),
        '--execution-threads', str(execution_threads),
        '--temp-frame-quality', str(temp_frame_quality)
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

def process_video_batch(source_path: str, target_videos: list, output_dir: str = None,
                       gpu_memory_wait: int = 30, max_memory: int = 12,
                       execution_threads: int = 8, temp_frame_quality: int = 100,
                       keep_fps: bool = True):
    """Procesar lote de videos"""
    
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE - ROOP DIRECTO")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"🎬 Videos a procesar: {len(target_videos)}")
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
    
    # Crear directorio de salida si no existe
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Directorio creado: {output_dir}")
    
    successful = 0
    failed = 0
    
    for i, target_path in enumerate(target_videos, 1):
        print(f"\n📹 Procesando video {i}/{len(target_videos)}")
        
        # Verificar que el target existe
        if not os.path.exists(target_path):
            print(f"❌ Target no encontrado: {target_path}")
            failed += 1
            continue
        
        # Generar nombre de salida
        if output_dir:
            target_name = Path(target_path).name
            source_name = Path(source_path).stem
            output_name = f"{source_name}_{target_name}"
            output_path = os.path.join(output_dir, output_name)
        else:
            output_path = f"output_{i}.mp4"
        
        # Esperar memoria GPU entre videos
        if i > 1:
            print("⏳ Esperando memoria GPU entre videos...")
            time.sleep(gpu_memory_wait)
        
        # Procesar video
        if process_single_video(
            source_path, target_path, output_path,
            gpu_memory_wait, max_memory, execution_threads,
            temp_frame_quality, keep_fps
        ):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 RESUMEN:")
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    if successful + failed > 0:
        print(f"📈 Tasa de éxito: {successful/(successful+failed)*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="ROOP Direct - Procesamiento directo")
    parser.add_argument('--source', required=True, help='Ruta de la imagen fuente')
    parser.add_argument('--target', required=True, help='Ruta del video objetivo')
    parser.add_argument('-o', '--output', help='Ruta de salida (requerido para modo single)')
    parser.add_argument('--gpu-memory-wait', type=int, default=30, help='Tiempo de espera GPU (s)')
    parser.add_argument('--max-memory', type=int, default=12, help='Memoria máxima (GB)')
    parser.add_argument('--execution-threads', type=int, default=8, help='Hilos de ejecución')
    parser.add_argument('--temp-frame-quality', type=int, default=100, help='Calidad de frames temporales')
    parser.add_argument('--keep-fps', action='store_true', help='Mantener FPS original')
    parser.add_argument('--batch', action='store_true', help='Modo lote')
    parser.add_argument('--output-dir', help='Directorio de salida para modo lote')
    
    args = parser.parse_args()
    
    # Configurar GPU
    if not setup_gpu():
        print("❌ No se pudo configurar GPU")
        return
    
    # Descargar modelo si es necesario
    if not download_model():
        print("❌ No se pudo descargar el modelo")
        return
    
    if args.batch:
        # Modo lote
        if not args.output_dir:
            print("❌ Error: --output-dir es requerido en modo batch")
            return
            
        target_videos = [args.target] if os.path.isfile(args.target) else []
        if os.path.isdir(args.target):
            target_videos = [os.path.join(args.target, f) for f in os.listdir(args.target) 
                           if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not target_videos:
            print("❌ No se encontraron videos para procesar")
            return
        
        process_video_batch(
            args.source, target_videos, args.output_dir,
            args.gpu_memory_wait, args.max_memory, args.execution_threads,
            args.temp_frame_quality, args.keep_fps
        )
    else:
        # Modo single
        if not args.output:
            print("❌ Error: -o/--output es requerido en modo single")
            return
            
        process_single_video(
            args.source, args.target, args.output,
            args.gpu_memory_wait, args.max_memory, args.execution_threads,
            args.temp_frame_quality, args.keep_fps
        )

if __name__ == '__main__':
    main() 