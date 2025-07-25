#!/usr/bin/env python3
"""
Script para reiniciar procesamiento con GPU habilitado
"""

import os
import sys
import subprocess

# Configurar para GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider,CPUExecutionProvider'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def restart_processing():
    """Reinicia el procesamiento con GPU"""
    print("🚀 REINICIANDO PROCESAMIENTO CON GPU")
    print("=" * 50)
    
    command = [
        sys.executable, "run.py",
        "--source", "/content/DanielaAS.jpg",
        "--target", "/content/130.mp4",
        "-o", "/content/resultados/DanielaAS130.mp4",
        "--frame-processor", "face_swapper",  # Solo face_swapper para estabilidad
        "--gpu-memory-wait", "45",
        "--max-memory", "8",
        "--execution-threads", "24",
        "--temp-frame-quality", "90",
        "--execution-provider", "cuda",  # Solo cuda, no cuda,cpu
        "--keep-fps"
    ]
    
    try:
        print("🔄 Ejecutando con GPU...")
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
    restart_processing() 