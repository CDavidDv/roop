#!/usr/bin/env python3
"""
Script que fuerza el uso de GPU
"""

import os
import sys

def force_gpu_environment():
    """Configura el entorno para forzar GPU"""
    print("🚀 FORZANDO CONFIGURACIÓN GPU")
    print("=" * 40)
    
    # Configurar variables de entorno para GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['ONNXRUNTIME_PROVIDER'] = 'CUDAExecutionProvider'
    
    # Configurar path de librerías CUDA
    cuda_paths = [
        '/usr/lib/x86_64-linux-gnu',
        '/usr/local/cuda-11.8/lib64',
        '/usr/local/cuda/lib64'
    ]
    
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = ':'.join(cuda_paths + [current_ld_path])
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    
    print("✅ Variables de entorno GPU configuradas")
    print(f"✅ LD_LIBRARY_PATH: {new_ld_path}")
    
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

# FORZAR GPU SOLO - NO CPU
execution_providers = ['CUDAExecutionProvider']

execution_threads: Optional[int] = None
log_level: str = 'error'
'''
    
    with open('roop/globals.py', 'w') as f:
        f.write(globals_content)
    
    print("✅ Configuración forzada a GPU")

def test_gpu_only():
    """Prueba que solo use GPU"""
    print("\n🧪 PROBANDO GPU OBLIGATORIO")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        
        # Verificar proveedores disponibles
        available_providers = ort.get_available_providers()
        print(f"✅ Proveedores disponibles: {available_providers}")
        
        if 'CUDAExecutionProvider' in available_providers:
            print("✅ CUDA disponible")
        else:
            print("❌ CUDA no disponible")
            return False
        
        # Probar face swapper con GPU
        from roop.processors.frame.face_swapper import get_face_swapper
        swapper = get_face_swapper()
        print("✅ Face swapper con GPU cargado")
        
        # Probar face analyser con GPU
        from roop.face_analyser import get_face_analyser
        analyser = get_face_analyser()
        print("✅ Face analyser con GPU cargado")
        
        print("\n🎉 ¡GPU OBLIGATORIO FUNCIONANDO!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    force_gpu_environment()
    
    if test_gpu_only():
        print("\n🚀 ¡LISTO PARA PROCESAR CON GPU!")
        print("=" * 40)
        print("Comando para procesar:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
        return 0
    else:
        print("\n❌ GPU no funciona correctamente")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 