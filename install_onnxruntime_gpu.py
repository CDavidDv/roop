#!/usr/bin/env python3
"""
Script para instalar onnxruntime-gpu y solucionar el problema de GPU
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_current_onnx():
    """Verificar instalación actual de ONNX Runtime"""
    print("🔍 VERIFICANDO INSTALACIÓN ACTUAL DE ONNX RUNTIME")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
        print(f"📦 Versión actual: {ort.__version__}")
        
        # Verificar si es versión GPU
        if 'gpu' in ort.__version__.lower() or 'cuda' in ort.__version__.lower():
            print("✅ ONNX Runtime GPU detectado")
            return True
        else:
            print("❌ ONNX Runtime CPU detectado")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando ONNX Runtime: {e}")
        return False

def uninstall_onnxruntime_cpu():
    """Desinstalar ONNX Runtime CPU"""
    print("\n🗑️ DESINSTALANDO ONNX RUNTIME CPU")
    print("=" * 40)
    
    try:
        # Desinstalar onnxruntime
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', 'onnxruntime', '-y'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ONNX Runtime CPU desinstalado")
            return True
        else:
            print(f"⚠️ Error desinstalando: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def install_onnxruntime_gpu():
    """Instalar ONNX Runtime GPU"""
    print("\n📦 INSTALANDO ONNX RUNTIME GPU")
    print("=" * 40)
    
    try:
        # Instalar onnxruntime-gpu
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'onnxruntime-gpu==1.15.1'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ONNX Runtime GPU instalado")
            return True
        else:
            print(f"❌ Error instalando: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def verify_gpu_installation():
    """Verificar instalación de GPU"""
    print("\n✅ VERIFICANDO INSTALACIÓN DE GPU")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        
        print(f"📦 Versión: {ort.__version__}")
        
        # Verificar proveedores
        providers = ort.get_available_providers()
        print(f"🔧 Proveedores: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible")
            
            # Probar sesión con CUDA
            try:
                import numpy as np
                
                # Crear un tensor simple para probar
                test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                
                # Crear sesión con CUDA
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                
                # Crear un modelo simple para probar
                import onnx
                from onnx import helper, numpy_helper
                
                # Crear un modelo ONNX simple
                X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
                Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
                
                # Operación simple: identidad
                node = helper.make_node('Identity', inputs=['X'], outputs=['Y'])
                
                graph = helper.make_graph([node], 'test', [X], [Y])
                model = helper.make_model(graph)
                
                # Probar con CUDA
                session = ort.InferenceSession(
                    model.SerializeToString(),
                    session_options,
                    providers=['CUDAExecutionProvider']
                )
                
                print("✅ Sesión CUDA creada exitosamente")
                print("✅ GPU funciona correctamente")
                return True
                
            except Exception as e:
                print(f"⚠️ Error probando sesión CUDA: {e}")
                return False
        else:
            print("❌ CUDA GPU NO disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error verificando: {e}")
        return False

def test_face_swapper_gpu():
    """Probar face-swapper con GPU"""
    print("\n🧪 PROBANDO FACE-SWAPPER CON GPU")
    print("=" * 40)
    
    # Comando de prueba
    test_cmd = [
        "roop_env/bin/python", 'run.py',
        '--source', '/content/DanielaAS.jpg',
        '--target', '/content/112.mp4',
        '-o', '/content/test_gpu_face_swapper.mp4',
        '--frame-processor', 'face_swapper',
        '--execution-provider', 'cuda',
        '--max-memory', '8',
        '--execution-threads', '8',
        '--gpu-memory-wait', '5',
        '--temp-frame-quality', '100',
        '--temp-frame-format', 'png',
        '--output-video-encoder', 'h264_nvenc',
        '--output-video-quality', '100',
        '--keep-fps'
    ]
    
    print("Comando de prueba:")
    print(" ".join(test_cmd))
    
    print("\n💡 Para probar:")
    print("1. Ejecuta el comando anterior")
    print("2. Monitorea con: nvidia-smi -l 1")
    print("3. Verifica que VRAM > 0GB durante el procesamiento")
    print("4. Busca mensajes de diagnóstico GPU en la salida")

def create_environment_script():
    """Crear script con variables de entorno para GPU"""
    
    env_script = '''#!/bin/bash
# Variables de entorno para forzar GPU en ROOP
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=1
export ONNXRUNTIME_PROVIDER_SHARED_LIB=/usr/local/cuda/lib64/libonnxruntime_providers_cuda.so

echo "🔧 Variables de entorno configuradas para GPU"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "TF_FORCE_GPU_ALLOW_GROWTH=$TF_FORCE_GPU_ALLOW_GROWTH"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Ejecutar ROOP con GPU
roop_env/bin/python run.py "$@"
'''
    
    with open('run_roop_gpu.sh', 'w') as f:
        f.write(env_script)
    
    # Hacer ejecutable
    os.chmod('run_roop_gpu.sh', 0o755)
    
    print("\n📝 Script creado: run_roop_gpu.sh")
    print("💡 Uso: ./run_roop_gpu.sh --source imagen.jpg --target video.mp4 -o salida.mp4")

def main():
    print("🔧 INSTALADOR DE ONNX RUNTIME GPU")
    print("=" * 60)
    
    # Verificar instalación actual
    current_is_gpu = check_current_onnx()
    
    if current_is_gpu:
        print("\n✅ Ya tienes ONNX Runtime GPU instalado")
        verify_gpu_installation()
        test_face_swapper_gpu()
        create_environment_script()
        return
    
    print("\n❌ Necesitas instalar ONNX Runtime GPU")
    print("=" * 40)
    
    # Confirmar instalación
    response = input("¿Instalar ONNX Runtime GPU? (y/n): ").lower()
    if response != 'y':
        print("❌ Instalación cancelada")
        return
    
    # Desinstalar CPU
    if not uninstall_onnxruntime_cpu():
        print("⚠️ No se pudo desinstalar ONNX Runtime CPU")
    
    # Instalar GPU
    if install_onnxruntime_gpu():
        print("\n✅ Instalación completada")
        
        # Verificar instalación
        if verify_gpu_installation():
            print("\n🎉 ¡ONNX Runtime GPU instalado correctamente!")
            
            # Probar face-swapper
            test_face_swapper_gpu()
            
            # Crear script de entorno
            create_environment_script()
            
            print("\n" + "=" * 60)
            print("🚀 PRÓXIMOS PASOS:")
            print("=" * 60)
            print("1. Usa el script: ./run_roop_gpu.sh")
            print("2. O ejecuta con variables de entorno:")
            print("   CUDA_VISIBLE_DEVICES=0 roop_env/bin/python run.py ...")
            print("3. Monitorea con: nvidia-smi -l 1")
            print("4. Verifica que VRAM > 0GB durante el procesamiento")
        else:
            print("\n❌ Error verificando instalación de GPU")
    else:
        print("\n❌ Error instalando ONNX Runtime GPU")

if __name__ == "__main__":
    main() 