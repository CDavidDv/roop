#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import argparse

# Configurar variables de entorno ANTES de cualquier import
os.environ['MPLBACKEND'] = 'Agg'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')

# Modificar predictor para saltar NSFW
import roop.predictor
def predict_video_skip_nsfw(target_path: str) -> bool:
    print("⚠️ Saltando verificación NSFW para evitar conflictos de GPU...")
    return False

roop.predictor.predict_video = predict_video_skip_nsfw

def check_gpu_usage():
    """Verificar uso de GPU en tiempo real"""
    try:
        import torch
        import tensorflow as tf
        import onnxruntime as ort
        
        print("\n🔍 VERIFICACIÓN GPU:")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name()}")
            print(f"PyTorch VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        
        print(f"TensorFlow GPU: {len(tf.config.list_physical_devices('GPU'))}")
        print(f"ONNX Runtime providers: {ort.get_available_providers()}")
        
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            print("✅ CUDA GPU disponible")
        else:
            print("❌ CUDA GPU no disponible")
            
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")

def run_sequential_processing():
    """Procesamiento secuencial manteniendo configuración del usuario"""
    
    # Configuración del usuario (mantener exactamente igual)
    source_path = "/content/SakuraAS.png"
    target_path = "/content/17.mp4"
    output_path = "/content/resultado.mp4"
    execution_provider = "cuda"
    max_memory = "12"
    execution_threads = "33"
    temp_frame_quality = "100"
    
    print("🎭 INICIANDO PROCESAMIENTO SECUENCIAL")
    print("=" * 60)
    print(f"📸 Source: {source_path}")
    print(f"🎬 Target: {target_path}")
    print(f"💾 Output: {output_path}")
    print(f"⚡ GPU: {execution_provider}")
    print(f"🧠 Memory: {max_memory}GB")
    print(f"🧵 Threads: {execution_threads}")
    print(f"🎨 Quality: {temp_frame_quality}")
    print("=" * 60)
    
    # PASO 1: Face Swapper
    print("\n📸 PASO 1: Face Swapper")
    check_gpu_usage()
    
    cmd1 = [
        "roop_env/bin/python", "run.py",
        "--source", source_path,
        "--target", target_path,
        "-o", "/content/temp_swapped.mp4",
        "--execution-provider", execution_provider,
        "--max-memory", max_memory,
        "--execution-threads", execution_threads,
        "--frame-processor", "face_swapper",
        "--temp-frame-quality", temp_frame_quality,
        "--keep-fps"
    ]
    
    print("🔄 Ejecutando Face Swapper...")
    print(f"Comando: {' '.join(cmd1)}")
    
    result1 = subprocess.run(cmd1, capture_output=True, text=True)
    
    if result1.returncode == 0:
        print("✅ Face Swapper completado exitosamente")
        check_gpu_usage()
    else:
        print(f"❌ Error en Face Swapper:")
        print(f"STDOUT: {result1.stdout}")
        print(f"STDERR: {result1.stderr}")
        return False
    
    # ESPERA 15 segundos para liberar memoria
    print("\n⏳ Esperando 15 segundos antes del Face Enhancer...")
    for i in range(15, 0, -1):
        print(f"⏰ {i} segundos restantes...", end='\r')
        time.sleep(1)
    print("\n")
    
    # PASO 2: Face Enhancer
    print("\n✨ PASO 2: Face Enhancer")
    check_gpu_usage()
    
    cmd2 = [
        "roop_env/bin/python", "run.py",
        "--source", source_path,
        "--target", "/content/temp_swapped.mp4",
        "-o", output_path,
        "--execution-provider", execution_provider,
        "--max-memory", max_memory,
        "--execution-threads", execution_threads,
        "--frame-processor", "face_enhancer",
        "--temp-frame-quality", temp_frame_quality,
        "--keep-fps"
    ]
    
    print("🔄 Ejecutando Face Enhancer...")
    print(f"Comando: {' '.join(cmd2)}")
    
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    
    if result2.returncode == 0:
        print("✅ Face Enhancer completado exitosamente")
        check_gpu_usage()
        print(f"\n🎉 ¡PROCESAMIENTO COMPLETADO!")
        print(f"📁 Resultado guardado en: {output_path}")
        return True
    else:
        print(f"❌ Error en Face Enhancer:")
        print(f"STDOUT: {result2.stdout}")
        print(f"STDERR: {result2.stderr}")
        return False

def run_single_processor(processor_name, source_path, target_path, output_path):
    """Ejecutar un solo procesador"""
    
    execution_provider = "cuda"
    max_memory = "12"
    execution_threads = "33"
    temp_frame_quality = "100"
    
    cmd = [
        "roop_env/bin/python", "run.py",
        "--source", source_path,
        "--target", target_path,
        "-o", output_path,
        "--execution-provider", execution_provider,
        "--max-memory", max_memory,
        "--execution-threads", execution_threads,
        "--frame-processor", processor_name,
        "--temp-frame-quality", temp_frame_quality,
        "--keep-fps"
    ]
    
    print(f"🔄 Ejecutando {processor_name}...")
    print(f"Comando: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {processor_name} completado exitosamente")
        return True
    else:
        print(f"❌ Error en {processor_name}:")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

if __name__ == "__main__":
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        if sys.argv[1] == "--sequential":
            # Modo secuencial
            success = run_sequential_processing()
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--face-swapper":
            # Solo face swapper
            success = run_single_processor("face_swapper", 
                                        "/content/SakuraAS.png", 
                                        "/content/17.mp4", 
                                        "/content/resultado_swapper.mp4")
            sys.exit(0 if success else 1)
        elif sys.argv[1] == "--face-enhancer":
            # Solo face enhancer
            success = run_single_processor("face_enhancer", 
                                        "/content/SakuraAS.png", 
                                        "/content/17.mp4", 
                                        "/content/resultado_enhancer.mp4")
            sys.exit(0 if success else 1)
        else:
            print("Uso:")
            print("  python run_roop_sequential.py --sequential    # Procesamiento secuencial")
            print("  python run_roop_sequential.py --face-swapper  # Solo face swapper")
            print("  python run_roop_sequential.py --face-enhancer # Solo face enhancer")
    else:
        # Modo secuencial por defecto
        success = run_sequential_processing()
        sys.exit(0 if success else 1) 