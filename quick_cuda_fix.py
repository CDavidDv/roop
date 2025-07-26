#!/usr/bin/env python3
"""
Fix rápido para CUDA
"""

import subprocess
import sys
import os

def quick_cuda_fix():
    """Fix rápido para librerías CUDA"""
    print("🔧 FIX RÁPIDO PARA CUDA")
    print("=" * 40)
    
    try:
        # 1. Instalar librerías CUDA básicas
        print("1. Instalando librerías CUDA...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "onnxruntime-gpu"
        ], check=True)
        
        # 2. Instalar librerías CUDA adicionales
        print("2. Instalando librerías adicionales...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cudnn-cu12"
        ], check=True)
        
        # 3. Configurar variables de entorno
        print("3. Configurando entorno...")
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        
        # 4. Forzar GPU en globals.py
        print("4. Forzando GPU...")
        globals_content = '''from typing import List, Optional
import onnxruntime as ort

source_path: Optional[str] = None
target_path: Optional[str] = None
output_path: Optional[str] = None
headless: Optional[bool] = None
frame_processors: List[str] = []
keep_fps: Optional[bool] = None
keep_frames: Optional[bool] = None
skip_audio: Optional[bool] = None
many_faces: Optional[bool] = None
reference_face_position: Optional[int] = None
reference_frame_number: Optional[int] = None
similar_face_distance: Optional[float] = None
temp_frame_format: Optional[str] = None
temp_frame_quality: Optional[int] = None
output_video_encoder: Optional[str] = None
output_video_quality: Optional[int] = None
max_memory: Optional[int] = None

# FORZAR GPU SOLO
execution_providers = ['CUDAExecutionProvider']

execution_threads: Optional[int] = None
log_level: str = 'error'
'''
        
        with open('roop/globals.py', 'w') as f:
            f.write(globals_content)
        
        print("✅ Configuración GPU forzada")
        
        # 5. Probar CUDA
        print("5. Probando CUDA...")
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        print(f"✅ Proveedores: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            print("✅ CUDA disponible")
            return True
        else:
            print("❌ CUDA no disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal"""
    if quick_cuda_fix():
        print("\n🎉 ¡CUDA ARREGLADO!")
        print("=" * 30)
        print("Ahora ejecuta:")
        print("python test_original_gpu.py")
        return 0
    else:
        print("\n❌ Error arreglando CUDA")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 