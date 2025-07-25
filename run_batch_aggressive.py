#!/usr/bin/env python3
"""
Script para procesamiento agresivo con máximo uso de recursos
"""

import os
import sys
import subprocess
import argparse
import glob
import multiprocessing
from pathlib import Path
import psutil

def setup_aggressive_environment():
    """Configura el entorno para máximo rendimiento"""
    print("🚀 CONFIGURANDO ENTORNO AGRESIVO")
    print("=" * 50)
    
    # Variables de entorno para máximo rendimiento
    env_vars = {
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'CUDA_VISIBLE_DEVICES': '0',
        'MPLBACKEND': 'Agg',
        'NO_ALBUMENTATIONS_UPDATE': '1',
        'ONNXRUNTIME_PROVIDER': 'CUDAExecutionProvider,CPUExecutionProvider',
        'TF_MEMORY_ALLOCATION': '1.0',
        'ONNXRUNTIME_GPU_MEMORY_LIMIT': '8589934592',
        'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', ''),
        'CUDA_MEMORY_FRACTION': '1.0',
        'TF_GPU_MEMORY_FRACTION': '1.0',
        'OMP_NUM_THREADS': '64',
        'MKL_NUM_THREADS': '64',
        'OPENBLAS_NUM_THREADS': '64',
        'VECLIB_MAXIMUM_THREADS': '64',
        'NUMEXPR_NUM_THREADS': '64',
        'BLAS_NUM_THREADS': '64',
        'LAPACK_NUM_THREADS': '64'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key} = {value}")

def get_optimal_threads():
    """Obtiene el número óptimo de hilos"""
    cpu_count = multiprocessing.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Usar todos los cores disponibles
    optimal_threads = min(cpu_count * 2, 128)  # Hasta 128 hilos
    
    print(f"🖥️ CPU cores: {cpu_count}")
    print(f"💾 RAM total: {memory_gb:.1f}GB")
    print(f"🧵 Hilos óptimos: {optimal_threads}")
    
    return optimal_threads

def process_single_video_aggressive(source_path, video_path, output_dir, temp_quality=100, keep_fps=True):
    """Procesa un video con configuración agresiva"""
    print(f"🔥 PROCESANDO AGRESIVO: {os.path.basename(video_path)}")
    
    # Crear nombre de archivo de salida
    video_name = Path(video_path).stem
    source_name = Path(source_path).stem
    output_filename = f"{source_name}_{video_name}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Obtener hilos óptimos
    optimal_threads = get_optimal_threads()
    
    # Comando con configuración agresiva
    command = [
        sys.executable, "run.py",
        "--source", source_path,
        "--target", video_path,
        "-o", output_path,
        "--frame-processor", "face_swapper", "face_enhancer",
        "--execution-provider", "cuda",
        "--execution-threads", str(optimal_threads),
        "--temp-frame-quality", str(temp_quality),
        "--max-memory", "12",
        "--gpu-memory-wait", "2"
    ]
    
    if keep_fps:
        command.append("--keep-fps")
    
    try:
        print(f"🚀 Iniciando procesamiento agresivo: {video_name}")
        print(f"⚡ Hilos: {optimal_threads} | 💾 RAM: 12GB | 🎮 GPU: 8GB")
        
        result = subprocess.run(command, timeout=1800)  # 30 minutos timeout
        
        if result.returncode == 0:
            print(f"✅ Completado agresivo: {output_filename}")
            return True
        else:
            print(f"❌ Error procesando: {video_name}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {video_name}")
        return False
    except Exception as e:
        print(f"❌ Excepción en {video_name}: {e}")
        return False

def process_batch_aggressive(source_path, video_paths, output_dir, temp_quality=100, keep_fps=True):
    """Procesa múltiples videos con configuración agresiva"""
    print("🚀 PROCESAMIENTO AGRESIVO CON MÁXIMO RENDIMIENTO")
    print("=" * 60)
    print(f"📸 Imagen fuente: {source_path}")
    print(f"🎬 Videos a procesar: {len(video_paths)}")
    print(f"📁 Directorio de salida: {output_dir}")
    print(f"⚡ Calidad temporal: {temp_quality}")
    print(f"🎯 Mantener FPS: {keep_fps}")
    print("=" * 60)
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar entorno agresivo
    setup_aggressive_environment()
    
    # Procesar cada video
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n📹 [{i}/{len(video_paths)}] Procesando agresivo: {os.path.basename(video_path)}")
        
        if process_single_video_aggressive(source_path, video_path, output_dir, temp_quality, keep_fps):
            successful += 1
        else:
            failed += 1
    
    # Resumen final
    print("\n🎉 RESUMEN DEL PROCESAMIENTO AGRESIVO")
    print("=" * 50)
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    print(f"📊 Total: {len(video_paths)}")
    
    if successful > 0:
        print(f"\n📁 Archivos guardados en: {output_dir}")
        print("📋 Archivos generados:")
        for video_path in video_paths:
            video_name = Path(video_path).stem
            source_name = Path(source_path).stem
            output_filename = f"{source_name}_{video_name}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            if os.path.exists(output_path):
                print(f"  ✅ {output_filename}")
            else:
                print(f"  ❌ {output_filename} (no encontrado)")
    
    return successful, failed

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Procesamiento agresivo con ROOP GPU")
    parser.add_argument("--source", required=True, help="Ruta de la imagen fuente")
    parser.add_argument("--videos", nargs="+", required=True, help="Rutas de los videos a procesar")
    parser.add_argument("--output-dir", default="/content/resultados", help="Directorio de salida")
    parser.add_argument("--temp-frame-quality", type=int, default=100, help="Calidad de frames temporales (1-100)")
    parser.add_argument("--keep-fps", action="store_true", help="Mantener FPS original")
    
    args = parser.parse_args()
    
    return process_batch_aggressive(
        args.source, 
        args.videos, 
        args.output_dir, 
        args.temp_frame_quality, 
        args.keep_fps
    )

if __name__ == "__main__":
    sys.exit(main()) 