#!/usr/bin/env python3
"""
Instalación final de librerías CUDA
"""

import subprocess
import sys
import os

def install_cuda_final():
    """Instala las librerías CUDA faltantes"""
    print("🔧 INSTALACIÓN FINAL DE LIBRERÍAS CUDA")
    print("=" * 50)
    
    try:
        # 1. Instalar librerías CUDA del sistema
        print("1. Instalando librerías CUDA del sistema...")
        subprocess.run([
            "apt-get", "update"
        ], check=True)
        
        subprocess.run([
            "apt-get", "install", "-y", "libcublas-11-8", "libcudnn8", "libcufft-11-8"
        ], check=True)
        print("✅ Librerías CUDA del sistema instaladas")
        
        # 2. Reinstalar onnxruntime-gpu
        print("2. Reinstalando onnxruntime-gpu...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.16.3"
        ], check=True)
        print("✅ ONNX Runtime GPU reinstalado")
        
        # 3. Instalar librerías CUDA específicas
        print("3. Instalando librerías CUDA específicas...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cudnn-cu12==8.9.4.25"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cublas-cu12==12.1.3.1"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cufft-cu12==11.0.2.54"
        ], check=True)
        print("✅ Librerías CUDA específicas instaladas")
        
        # 4. Configurar variables de entorno
        print("4. Configurando variables de entorno...")
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        print("✅ Variables de entorno configuradas")
        
        # 5. Crear enlaces simbólicos
        print("5. Creando enlaces simbólicos...")
        subprocess.run([
            "ln", "-sf", "/usr/lib/x86_64-linux-gnu/libcublasLt.so.11", "/usr/local/cuda/lib64/libcublasLt.so.11"
        ], check=True)
        
        subprocess.run([
            "ln", "-sf", "/usr/lib/x86_64-linux-gnu/libcudnn.so.8", "/usr/local/cuda/lib64/libcudnn.so.8"
        ], check=True)
        
        subprocess.run([
            "ln", "-sf", "/usr/lib/x86_64-linux-gnu/libcufft.so.10", "/usr/local/cuda/lib64/libcufft.so.10"
        ], check=True)
        print("✅ Enlaces simbólicos creados")
        
        # 6. Verificar instalación
        print("6. Verificando instalación...")
        test_code = '''
import onnxruntime as ort
import insightface

# Verificar proveedores
providers = ort.get_available_providers()
print(f"Proveedores disponibles: {providers}")

if 'CUDAExecutionProvider' in providers:
    print("✅ CUDA disponible")
    
    # Probar InsightFace con GPU
    app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0)
    print("✅ InsightFace con GPU funcionando")
    
    # Probar modelo de face swap
    model_path = "models/inswapper_128.onnx"
    swapper = insightface.model_zoo.get_model(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print("✅ Face swapper con GPU funcionando")
else:
    print("❌ CUDA no disponible")
'''
        
        result = subprocess.run([
            sys.executable, "-c", test_code
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ CUDA funcionando correctamente")
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def force_gpu_only():
    """Fuerza el uso de GPU solo"""
    print("\n🚀 FORZANDO GPU SOLO")
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
    print("🚀 INSTALACIÓN FINAL DE CUDA")
    print("=" * 60)
    
    # Verificar si estamos en root
    if os.geteuid() != 0:
        print("⚠️ Este script necesita permisos de root")
        print("Ejecuta con: sudo python install_cuda_final.py")
        return 1
    
    if install_cuda_final():
        force_gpu_only()
        
        print("\n🎉 ¡CUDA INSTALADO Y FUNCIONANDO!")
        print("=" * 50)
        print("✅ Librerías CUDA instaladas")
        print("✅ ONNX Runtime GPU reinstalado")
        print("✅ GPU forzado")
        print("✅ Face swap con GPU")
        print("\n🚀 Ahora ejecuta:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
        return 0
    else:
        print("\n❌ Error instalando CUDA")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 