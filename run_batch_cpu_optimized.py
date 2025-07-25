#!/usr/bin/env python3
"""
Script de batch optimizado para CPU
"""

import os
import sys
import subprocess
import time
import argparse

# Configurar para usar SOLO CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ONNXRUNTIME_PROVIDER'] = 'CPUExecutionProvider'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_MEMORY_ALLOCATION'] = '2048'

def process_video(source, video_path, output_dir, execution_threads=8, temp_frame_quality=80):
    """Procesa un video individual con CPU"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_name = f"{os.path.splitext(os.path.basename(source))[0]}{video_name}.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"🎬 Procesando: {video_path}")
    print(f"📸 Source: {source}")
    print(f"💾 Output: {output_path}")
    print(f"🖥️ Usando CPU optimizado")
    
    # Comando optimizado para CPU
    command = [
        sys.executable, "run_cpu_optimized.py",
        "--source", source,
        "--target", video_path,
        "-o", output_path,
        "--frame-processor", "face_swapper",  # Solo face_swapper para CPU
        "--max-memory", "2",  # 2GB para CPU
        "--execution-threads", str(execution_threads),
        "--temp-frame-quality", str(temp_frame_quality),
        "--execution-provider", "cpu",  # Forzar CPU
        "--keep-fps"
    ]
    
    try:
        print("🔄 Iniciando procesamiento con CPU...")
        result = subprocess.run(command, capture_output=True, text=True, timeout=3600)  # 1 hora timeout
        if result.returncode == 0:
            print(f"✅ Completado: {output_path}")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Timeout: {video_path}")
        return False
    except Exception as e:
        print(f"❌ Excepción: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Procesamiento optimizado en CPU")
    parser.add_argument("--source", required=True, help="Imagen fuente")
    parser.add_argument("--videos", nargs="+", required=True, help="Videos a procesar")
    parser.add_argument("--output-dir", required=True, help="Directorio de salida")
    parser.add_argument("--execution-threads", type=int, default=8, help="Hilos de ejecución (reducido para CPU)")
    parser.add_argument("--temp-frame-quality", type=int, default=80, help="Calidad de frames temporales")
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 INICIANDO PROCESAMIENTO CON CPU OPTIMIZADO")
    print("=" * 60)
    print(f"📸 Source: {args.source}")
    print(f"🎬 Videos: {len(args.videos)}")
    print(f"💾 Output: {args.output_dir}")
    print(f"⚙️ Threads: {args.execution_threads} (CPU)")
    print(f"📊 Quality: {args.temp_frame_quality}")
    print(f"🖥️ Modo: CPU optimizado")
    print("=" * 60)
    
    completed = 0
    failed = 0
    
    for i, video_path in enumerate(args.videos, 1):
        print(f"\n📊 Progreso: {i}/{len(args.videos)} ({i/len(args.videos)*100:.1f}%)")
        
        if process_video(args.source, video_path, args.output_dir, 
                        args.execution_threads, args.temp_frame_quality):
            completed += 1
        else:
            failed += 1
        
        # Pausa entre videos para liberar memoria
        if i < len(args.videos):
            print("⏳ Esperando 20 segundos...")
            time.sleep(20)
    
    print(f"\n🎉 PROCESAMIENTO COMPLETADO")
    print(f"✅ Completados: {completed}")
    print(f"❌ Fallidos: {failed}")

if __name__ == "__main__":
    main() 