#!/usr/bin/env python3
"""
Script para ejecutar ROOP con gestión de memoria GPU entre procesadores
"""

import os
import sys
import argparse

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    parser = argparse.ArgumentParser(description='ROOP con gestión de memoria GPU')
    parser.add_argument('--source', required=True, help='Imagen fuente')
    parser.add_argument('--target', required=True, help='Video objetivo')
    parser.add_argument('--output', required=True, help='Archivo de salida')
    parser.add_argument('--gpu-memory-wait', type=int, default=30, 
                       help='Tiempo de espera entre procesadores (segundos, default: 30)')
    parser.add_argument('--max-memory', type=int, default=12, 
                       help='Memoria máxima en GB (default: 12)')
    parser.add_argument('--execution-threads', type=int, default=8, 
                       help='Número de hilos (default: 8)')
    parser.add_argument('--temp-frame-quality', type=int, default=100, 
                       help='Calidad de frames temporales (default: 100)')
    parser.add_argument('--keep-fps', action='store_true', 
                       help='Mantener FPS original')
    
    args = parser.parse_args()
    
    # Construir comando
    cmd = [
        sys.executable, 'run.py',
        '--source', args.source,
        '--target', args.target,
        '-o', args.output,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--gpu-memory-wait', str(args.gpu_memory_wait),
        '--max-memory', str(args.max_memory),
        '--execution-threads', str(args.execution_threads),
        '--temp-frame-quality', str(args.temp_frame_quality)
    ]
    
    if args.keep_fps:
        cmd.append('--keep-fps')
    
    print("🚀 EJECUTANDO ROOP CON GESTIÓN DE MEMORIA GPU")
    print("=" * 60)
    print(f"📸 Source: {args.source}")
    print(f"🎬 Target: {args.target}")
    print(f"💾 Output: {args.output}")
    print(f"⏰ GPU Memory Wait: {args.gpu_memory_wait}s")
    print(f"🧠 Max Memory: {args.max_memory}GB")
    print(f"🧵 Threads: {args.execution_threads}")
    print(f"🎨 Quality: {args.temp_frame_quality}")
    print(f"🎯 Keep FPS: {args.keep_fps}")
    print("=" * 60)
    
    # Ejecutar comando
    import subprocess
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ Procesamiento completado exitosamente!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error en el procesamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 