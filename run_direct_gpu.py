#!/usr/bin/env python3
"""
Script para ejecutar procesamiento directamente con GPU
"""

import os
import sys
import subprocess

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'

def run_direct_processing():
    """Ejecuta el procesamiento directamente"""
    print("🚀 EJECUTANDO PROCESAMIENTO DIRECTO CON GPU")
    print("=" * 60)
    
    # Comando directo sin usar core.py
    command = [
        sys.executable, "-c", """
import os
import sys

# Configurar GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

# Importar roop directamente
sys.path.insert(0, '.')
from roop import core

# Configurar argumentos
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--source', required=True)
parser.add_argument('--target', required=True)
parser.add_argument('-o', '--output', required=True)
parser.add_argument('--frame-processor', nargs='+', default=['face_swapper'])
parser.add_argument('--execution-provider', default='cuda')
parser.add_argument('--execution-threads', type=int, default=24)
parser.add_argument('--temp-frame-quality', type=int, default=90)
parser.add_argument('--max-memory', type=int, default=8)
parser.add_argument('--gpu-memory-wait', type=int, default=45)
parser.add_argument('--keep-fps', action='store_true')

args = parser.parse_args([
    '--source', '/content/DanielaAS.jpg',
    '--target', '/content/130.mp4',
    '-o', '/content/resultados/DanielaAS130.mp4',
    '--frame-processor', 'face_swapper',
    '--execution-provider', 'cuda',
    '--execution-threads', '24',
    '--temp-frame-quality', '90',
    '--max-memory', '8',
    '--gpu-memory-wait', '45',
    '--keep-fps'
])

# Configurar globals
import roop.globals
roop.globals.source_path = args.source
roop.globals.target_path = args.target
roop.globals.output_path = args.output
roop.globals.frame_processors = args.frame_processor
roop.globals.execution_providers = [args.execution_provider]
roop.globals.execution_threads = args.execution_threads
roop.globals.temp_frame_quality = args.temp_frame_quality
roop.globals.max_memory = args.max_memory
roop.globals.gpu_memory_wait = args.gpu_memory_wait
roop.globals.keep_fps = args.keep_fps

# Ejecutar procesamiento
print('🚀 Iniciando procesamiento con GPU...')
roop.core.start()
print('✅ Procesamiento completado')
"""
    ]
    
    try:
        print("🔄 Ejecutando procesamiento directo...")
        result = subprocess.run(command, timeout=3600)  # 1 hora timeout
        if result.returncode == 0:
            print("✅ Procesamiento completado con GPU")
            return True
        else:
            print("❌ Error en procesamiento")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Timeout")
        return False
    except Exception as e:
        print(f"❌ Excepción: {e}")
        return False

if __name__ == "__main__":
    run_direct_processing() 