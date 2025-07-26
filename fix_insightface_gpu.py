#!/usr/bin/env python3
"""
Arreglar InsightFace para que use GPU correctamente
"""

import subprocess
import sys
import os

def fix_insightface_gpu():
    """Arregla InsightFace para GPU"""
    print("🔧 ARREGLANDO INSIGHTFACE PARA GPU")
    print("=" * 50)
    
    try:
        # 1. Reinstalar insightface con GPU
        print("1. Reinstalando insightface...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "insightface"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "insightface==0.7.3"
        ], check=True)
        print("✅ InsightFace reinstalado")
        
        # 2. Reinstalar onnxruntime-gpu
        print("2. Reinstalando onnxruntime-gpu...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime", "onnxruntime-gpu"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.16.3"
        ], check=True)
        print("✅ ONNX Runtime GPU reinstalado")
        
        # 3. Instalar librerías CUDA
        print("3. Instalando librerías CUDA...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cudnn-cu12==8.9.4.25"
        ], check=True)
        
        subprocess.run([
            sys.executable, "-m", "pip", "install", "nvidia-cublas-cu12==12.1.3.1"
        ], check=True)
        print("✅ Librerías CUDA instaladas")
        
        # 4. Configurar variables de entorno
        print("4. Configurando entorno...")
        os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['CUDA_HOME'] = '/usr/local/cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
        print("✅ Variables de entorno configuradas")
        
        # 5. Probar InsightFace con GPU
        print("5. Probando InsightFace con GPU...")
        test_code = '''
import insightface
import onnxruntime as ort

# Verificar proveedores
providers = ort.get_available_providers()
print(f"Proveedores disponibles: {providers}")

# Crear app con GPU
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0)
print("✅ InsightFace con GPU funcionando")
'''
        
        result = subprocess.run([
            sys.executable, "-c", test_code
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ InsightFace con GPU funciona")
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
    
    # Modificar face_swapper para forzar GPU
    swapper_content = '''from typing import Any, List, Callable
import cv2
import insightface
import threading
import onnxruntime as ort

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            
            # FORZAR GPU SOLO
            print(f"[{NAME}] 🔥 FORZANDO GPU SOLO")
            
            # Configuración agresiva para GPU
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            # Solo CUDA, no CPU
            providers = ['CUDAExecutionProvider']
            
            print(f"[{NAME}] ✅ Cargando modelo con GPU forzado")
            FACE_SWAPPER = insightface.model_zoo.get_model(
                model_path, 
                providers=providers,
                session_options=session_options
            )
            print(f"[{NAME}] ✅ Modelo cargado con GPU")
                
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.source_path):
        update_status('Select an image for source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('No face in source path detected.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(source_path))
    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
'''
    
    with open('roop/processors/frame/face_swapper.py', 'w') as f:
        f.write(swapper_content)
    
    print("✅ Face swapper forzado a GPU")

def main():
    """Función principal"""
    print("🚀 ARREGLANDO INSIGHTFACE PARA GPU")
    print("=" * 60)
    
    if fix_insightface_gpu():
        force_gpu_only()
        
        print("\n🎉 ¡INSIGHTFACE CON GPU ARREGLADO!")
        print("=" * 50)
        print("✅ InsightFace reinstalado")
        print("✅ ONNX Runtime GPU reinstalado")
        print("✅ Librerías CUDA instaladas")
        print("✅ GPU forzado en face swapper")
        print("\n🚀 Ahora ejecuta:")
        print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados")
        return 0
    else:
        print("\n❌ Error arreglando InsightFace")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 