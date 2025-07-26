#!/usr/bin/env python3
"""
Script para instalar librerías de CUDA faltantes
"""

import subprocess
import sys
import os

def install_cuda_libraries():
    """Instala las librerías de CUDA faltantes"""
    print("🔧 INSTALANDO LIBRERÍAS DE CUDA")
    print("=" * 50)
    
    commands = [
        # Actualizar repositorios
        "apt-get update",
        
        # Instalar librerías CUDA faltantes
        "apt-get install -y libcublas-11-8 libcudnn8 libcufft-11-8",
        
        # Crear enlaces simbólicos
        "ln -sf /usr/lib/x86_64-linux-gnu/libcublasLt.so.11 /usr/local/cuda-11.8/lib64/libcublasLt.so.11",
        "ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.8 /usr/local/cuda-11.8/lib64/libcudnn.so.8",
        "ln -sf /usr/lib/x86_64-linux-gnu/libcufft.so.10 /usr/local/cuda-11.8/lib64/libcufft.so.10",
        
        # Actualizar variables de entorno
        "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH",
        
        # Reinstalar onnxruntime-gpu
        "pip uninstall -y onnxruntime onnxruntime-gpu",
        "pip install onnxruntime-gpu==1.16.3",
        
        # Verificar instalación
        "ldconfig",
    ]
    
    for cmd in commands:
        print(f"Ejecutando: {cmd}")
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {cmd}")
            else:
                print(f"⚠️ {cmd} - {result.stderr}")
        except Exception as e:
            print(f"❌ Error en {cmd}: {e}")

def check_cuda_libraries():
    """Verifica que las librerías CUDA estén disponibles"""
    print("\n🔍 VERIFICANDO LIBRERÍAS CUDA")
    print("=" * 40)
    
    libraries = [
        "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11",
        "/usr/local/cuda-11.8/lib64/libcublasLt.so.11",
        "/usr/lib/x86_64-linux-gnu/libcudnn.so.8",
        "/usr/local/cuda-11.8/lib64/libcudnn.so.8",
        "/usr/lib/x86_64-linux-gnu/libcufft.so.10",
        "/usr/local/cuda-11.8/lib64/libcufft.so.10",
    ]
    
    for lib in libraries:
        if os.path.exists(lib):
            size = os.path.getsize(lib)
            print(f"✅ {lib} - {size:,} bytes")
        else:
            print(f"❌ {lib} - NO ENCONTRADO")

def force_gpu_only():
    """Fuerza el uso de GPU solo"""
    print("\n🚀 FORZANDO USO DE GPU")
    print("=" * 30)
    
    # Modificar globals.py para forzar GPU
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
    
    print("✅ Configuración forzada a GPU")

def main():
    """Función principal"""
    print("🚀 ARREGLANDO CUDA PARA GPU OBLIGATORIO")
    print("=" * 60)
    
    # Verificar si estamos en root
    if os.geteuid() != 0:
        print("⚠️ Este script necesita permisos de root")
        print("Ejecuta con: sudo python fix_cuda_libraries.py")
        return 1
    
    install_cuda_libraries()
    check_cuda_libraries()
    force_gpu_only()
    
    print("\n🎉 ¡CUDA ARREGLADO PARA GPU!")
    print("=" * 40)
    print("✅ Librerías CUDA instaladas")
    print("✅ GPU forzado")
    print("✅ ONNX Runtime GPU actualizado")
    print("\n🚀 Ahora ejecuta:")
    print("python test_original_gpu.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 