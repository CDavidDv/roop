#!/usr/bin/env python3
"""
Script para procesamiento por lotes con GPU optimizado
Usa la misma configuración que ya funciona
"""

import os
import sys
import subprocess
import argparse
import glob
from pathlib import Path

def setup_environment():
    """Configura las variables de entorno optimizadas"""
    print("⚙️ CONFIGURANDO ENTORNO OPTIMIZADO")
    print("=" * 50)
    
    # Variables de entorno que ya funcionan
    env_vars = {
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'CUDA_VISIBLE_DEVICES': '0',
        'MPLBACKEND': 'Agg',
        'NO_ALBUMENTATIONS_UPDATE': '1',
        'ONNXRUNTIME_PROVIDER': 'CUDAExecutionProvider,CPUExecutionProvider',
        'TF_MEMORY_ALLOCATION': '0.8',
        'ONNXRUNTIME_GPU_MEMORY_LIMIT': '2147483648',
        'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ {key} = {value}")

def process_single_video(source_path, video_path, output_dir, temp_quality=100, keep_fps=True):
    """Procesa un solo video"""
    print(f"🔄 Procesando: {os.path.basename(video_path)}")
    
    # Crear nombre de archivo de salida
    video_name = Path(video_path).stem
    source_name = Path(source_path).stem
    output_filename = f"{source_name}_{video_name}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    # Comando con la configuración que ya funciona
    command = [
        sys.executable, "run.py",
        "--source", source_path,
        "--target", video_path,
        "-o", output_path,
        "--frame-processor", "face_swapper",
        "--execution-provider", "cuda",
        "--execution-threads", "16",
        "--temp-frame-quality", str(temp_quality),
        "--max-memory", "4",
        "--gpu-memory-wait", "15"
    ]
    
    if keep_fps:
        command.append("--keep-fps")
    
    try:
        print(f"🚀 Iniciando procesamiento: {video_name}")
        result = subprocess.run(command, timeout=3600)  # 1 hora timeout
        
        if result.returncode == 0:
            print(f"✅ Completado: {output_filename}")
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

def process_batch(source_path, video_paths, output_dir, temp_quality=100, keep_fps=True):
    """Procesa múltiples videos en lote"""
    print("🚀 PROCESAMIENTO POR LOTES CON GPU")
    print("=" * 60)
    print(f"📸 Imagen fuente: {source_path}")
    print(f"🎬 Videos a procesar: {len(video_paths)}")
    print(f"📁 Directorio de salida: {output_dir}")
    print(f"⚡ Calidad temporal: {temp_quality}")
    print(f"🎯 Mantener FPS: {keep_fps}")
    print("=" * 60)
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurar entorno
    setup_environment()
    
    # Procesar cada video
    successful = 0
    failed = 0
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n📹 [{i}/{len(video_paths)}] Procesando: {os.path.basename(video_path)}")
        
        if process_single_video(source_path, video_path, output_dir, temp_quality, keep_fps):
            successful += 1
        else:
            failed += 1
    
    # Resumen final
    print("\n🎉 RESUMEN DEL PROCESAMIENTO")
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
    parser = argparse.ArgumentParser(description="Procesamiento por lotes con ROOP GPU")
    parser.add_argument("--source", required=True, help="Ruta de la imagen fuente")
    parser.add_argument("--videos", nargs="+", required=True, help="Rutas de los videos a procesar")
    parser.add_argument("--output-dir", default="/content/resultados", help="Directorio de salida")
    parser.add_argument("--temp-frame-quality", type=int, default=100, help="Calidad de frames temporales (1-100)")
    parser.add_argument("--keep-fps", action="store_true", help="Mantener FPS original")
    
    args = parser.parse_args()
    
    # Verificar que los archivos existan
    if not os.path.exists(args.source):
        print(f"❌ Error: Imagen fuente no encontrada: {args.source}")
        return 1
    
    missing_videos = []
    for video in args.videos:
        if not os.path.exists(video):
            missing_videos.append(video)
    
    if missing_videos:
        print(f"❌ Error: Videos no encontrados: {missing_videos}")
        return 1
    
    # Procesar lote
    successful, failed = process_batch(
        args.source, 
        args.videos, 
        args.output_dir, 
        args.temp_frame_quality, 
        args.keep_fps
    )
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 