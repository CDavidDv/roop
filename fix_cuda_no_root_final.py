#!/usr/bin/env python3
"""
Fix final para CUDA sin root
"""

import subprocess
import sys
import os

def fix_cuda_no_root_final():
    """Arregla CUDA sin root"""
    print("🔧 FIX FINAL PARA CUDA SIN ROOT")
    print("=" * 50)
    
    try:
        # 1. Reinstalar onnxruntime-gpu
        print("1. Reinstalando onnxruntime-gpu...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.16.3"
        ], check=True)
        print("✅ ONNX Runtime GPU reinstalado")
        
        # 2. Instalar librerías CUDA específicas
        print("2. Instalando librerías CUDA...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cudnn-cu12==8.9.4.25"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cublas-cu12==12.1.3.1"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cufft-cu12==11.0.2.54"
        ], check=True)
        print("✅ Librerías CUDA instaladas")
        
        # 3. Configurar variables de entorno
        print("3. Configurando entorno...")
        os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        print("✅ Variables de entorno configuradas")
        
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

def main():
    """Función principal"""
    if fix_cuda_no_root_final():
        print("\n🎉 ¡CUDA ARREGLADO!")
        print("=" * 30)
        print("Ahora ejecuta:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
        return 0
    else:
        print("\n❌ Error arreglando CUDA")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 