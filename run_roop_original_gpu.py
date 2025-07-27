#!/usr/bin/env python3
"""
Script optimizado para ROOP original con GPU
Mantiene la funcionalidad original pero optimizada para Google Colab T4
"""

import os
import sys
import argparse
import subprocess
import time
import gc
from pathlib import Path

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['TF_MEMORY_ALLOCATION'] = '0.8'
os.environ['TF_GPU_MEMORY_LIMIT'] = '12'

def setup_gpu():
    """Configurar GPU para Colab T4"""
    print("🚀 CONFIGURANDO GPU PARA COLAB T4")
    print("=" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detectada: {torch.cuda.get_device_name()}")
            print(f"📊 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # Configurar memoria GPU
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            print("✅ Configuración GPU completada")
        else:
            print("❌ GPU no disponible")
            return False
    except Exception as e:
        print(f"❌ Error configurando GPU: {e}")
        return False
    
    return True

def clear_gpu_memory():
    """Limpiar memoria GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    
    # Garbage collection
    gc.collect()

def process_single_video(source_path: str, target_path: str, output_path: str, 
                        max_memory: int = 12, execution_threads: int = 8, 
                        temp_frame_quality: int = 100, keep_fps: bool = True):
    """Procesar un solo video con ROOP original"""
    
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
        '--max-memory', str(max_memory),
        '--execution-threads', str(execution_threads),
        '--temp-frame-quality', str(temp_frame_quality)
    ]
    
    if keep_fps:
        cmd.append('--keep-fps')
    
    try:
        # Limpiar memoria antes de procesar
        clear_gpu_memory()
        
        # Ejecutar comando
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Limpiar memoria después de procesar
        clear_gpu_memory()
        
        print(f"✅ Video procesado exitosamente: {output_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error procesando {target_path}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def process_video_batch(source_path: str, videos: list, output_dir: str,
                       max_memory: int = 12, execution_threads: int = 8,
                       temp_frame_quality: int = 100, keep_fps: bool = True):
    """Procesar lote de videos con ROOP original"""
    
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE - ROOP ORIGINAL")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"🎬 Videos a procesar: {len(videos)}")
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Directorio creado: {output_dir}")
    
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(videos, 1):
        print(f"\n📹 Procesando video {i}/{len(videos)}")
        
        # Verificar que el video existe
        if not os.path.exists(video_path):
            print(f"❌ Video no encontrado: {video_path}")
            failed += 1
            continue
        
        # Generar nombre de salida
        video_name = Path(video_path).name
        source_name = Path(source_path).stem
        output_name = f"{source_name}_{video_name}"
        output_path = os.path.join(output_dir, output_name)
        
        # Esperar entre videos para liberar memoria
        if i > 1:
            print("⏳ Esperando liberación de memoria...")
            time.sleep(5)
        
        # Procesar video
        if process_single_video(
            source_path, video_path, output_path,
            max_memory, execution_threads, temp_frame_quality, keep_fps
        ):
            successful += 1
        else:
            failed += 1
    
    print(f"\n📊 RESUMEN:")
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    if successful + failed > 0:
        print(f"📈 Tasa de éxito: {successful/(successful+failed)*100:.1f}%")

def process_folder_batch(source_path: str, input_folder: str, output_dir: str,
                        max_memory: int = 12, execution_threads: int = 8,
                        temp_frame_quality: int = 100, keep_fps: bool = True):
    """Procesar todos los videos en una carpeta"""
    
    print("🚀 PROCESAMIENTO DE CARPETA - ROOP ORIGINAL")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"📁 Carpeta de entrada: {input_folder}")
    print(f"📁 Carpeta de salida: {output_dir}")
    print("=" * 60)
    
    # Verificar que la carpeta existe
    if not os.path.exists(input_folder):
        print(f"❌ Carpeta de entrada no encontrada: {input_folder}")
        return
    
    # Buscar videos en la carpeta
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    videos = []
    
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(input_folder, file)
            videos.append(video_path)
    
    if not videos:
        print(f"❌ No se encontraron videos en: {input_folder}")
        return
    
    print(f"🎬 Videos encontrados: {len(videos)}")
    for video in videos:
        print(f"  - {os.path.basename(video)}")
    
    # Procesar videos
    process_video_batch(
        source_path, videos, output_dir,
        max_memory, execution_threads, temp_frame_quality, keep_fps
    )

def fix_directory_structure():
    """Arreglar estructura de directorios si es necesario"""
    print("🔧 VERIFICANDO ESTRUCTURA DE DIRECTORIOS...")
    
    try:
        import roop
        print("✅ Módulo roop disponible")
        return True
    except ImportError:
        print("❌ Módulo roop no disponible, intentando arreglar...")
        
        # Buscar el directorio correcto
        current_dir = os.getcwd()
        
        def search_roop_module(start_path):
            for root, dirs, files in os.walk(start_path):
                if 'run.py' in files:
                    run_path = os.path.join(root, 'run.py')
                    try:
                        with open(run_path, 'r') as f:
                            content = f.read()
                            if 'from roop import' in content:
                                return root
                    except:
                        pass
            return None
        
        roop_dir = search_roop_module(current_dir)
        
        if roop_dir:
            print(f"✅ Directorio encontrado: {roop_dir}")
            os.chdir(roop_dir)
            print(f"✅ Cambiado a: {os.getcwd()}")
            
            # Configurar path
            if roop_dir not in sys.path:
                sys.path.insert(0, roop_dir)
            
            # Probar importación
            try:
                import roop
                print("✅ Módulo roop importado exitosamente")
                return True
            except ImportError as e:
                print(f"❌ Error importando roop: {e}")
                return False
        else:
            print("❌ No se pudo encontrar el directorio correcto")
            return False

def main():
    parser = argparse.ArgumentParser(description="ROOP Original optimizado para Google Colab T4")
    parser.add_argument('--source', required=True, help='Ruta de la imagen fuente')
    parser.add_argument('--videos', nargs='+', help='Lista de videos a procesar')
    parser.add_argument('--input-folder', help='Carpeta con videos a procesar')
    parser.add_argument('--output-dir', required=True, help='Directorio de salida')
    parser.add_argument('--max-memory', type=int, default=12, help='Memoria máxima (GB)')
    parser.add_argument('--execution-threads', type=int, default=8, help='Hilos de ejecución')
    parser.add_argument('--temp-frame-quality', type=int, default=100, help='Calidad de frames temporales')
    parser.add_argument('--keep-fps', action='store_true', help='Mantener FPS original')
    
    args = parser.parse_args()
    
    # Arreglar estructura de directorios si es necesario
    if not fix_directory_structure():
        print("❌ No se pudo arreglar la estructura de directorios")
        return
    
    # Configurar GPU
    if not setup_gpu():
        print("❌ No se pudo configurar GPU")
        return
    
    # Procesar según los argumentos
    if args.videos:
        # Procesar lista de videos
        process_video_batch(
            args.source, args.videos, args.output_dir,
            args.max_memory, args.execution_threads,
            args.temp_frame_quality, args.keep_fps
        )
    elif args.input_folder:
        # Procesar carpeta de videos
        process_folder_batch(
            args.source, args.input_folder, args.output_dir,
            args.max_memory, args.execution_threads,
            args.temp_frame_quality, args.keep_fps
        )
    else:
        print("❌ Error: Debes especificar --videos o --input-folder")

if __name__ == '__main__':
    main() 