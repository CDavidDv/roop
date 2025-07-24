#!/usr/bin/env python3
"""
Script para solucionar problemas de GPU en Google Colab
"""

import os
import sys
import subprocess
import time

def check_current_installation():
    """Verificar instalación actual"""
    print("🔍 VERIFICANDO INSTALACIÓN ACTUAL")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"ONNX Runtime providers: {ort.get_available_providers()}")
        
        # Verificar si es GPU o CPU
        if 'onnxruntime-gpu' in ort.__file__:
            print("✅ ONNX Runtime GPU instalado")
        else:
            print("❌ ONNX Runtime CPU instalado")
            
    except ImportError as e:
        print(f"❌ Error importando ONNX Runtime: {e}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name()}")
    except ImportError as e:
        print(f"❌ Error importando PyTorch: {e}")

def install_gpu_dependencies():
    """Instalar dependencias GPU"""
    print("\n🚀 INSTALANDO DEPENDENCIAS GPU")
    print("=" * 50)
    
    # Desinstalar onnxruntime CPU si está instalado
    print("📦 Desinstalando ONNX Runtime CPU...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "onnxruntime"], 
                      capture_output=True, text=True)
        print("✅ ONNX Runtime CPU desinstalado")
    except Exception as e:
        print(f"⚠️ Error desinstalando: {e}")
    
    # Instalar onnxruntime-gpu
    print("📦 Instalando ONNX Runtime GPU...")
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", "onnxruntime-gpu==1.15.1"], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ONNX Runtime GPU instalado exitosamente")
        else:
            print(f"❌ Error instalando: {result.stderr}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Verificar CUDA toolkit
    print("📦 Verificando CUDA toolkit...")
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA toolkit encontrado")
            print(f"   {result.stdout.split('release')[0].strip()}")
        else:
            print("⚠️ CUDA toolkit no encontrado")
    except FileNotFoundError:
        print("⚠️ CUDA toolkit no encontrado (normal en Colab)")

def test_gpu_after_fix():
    """Probar GPU después de la instalación"""
    print("\n🧪 PROBANDO GPU DESPUÉS DE LA INSTALACIÓN")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible")
            
            # Probar crear una sesión con CUDA
            try:
                session = ort.InferenceSession("", providers=['CUDAExecutionProvider'])
                print("✅ Sesión CUDA creada exitosamente")
            except Exception as e:
                print(f"⚠️ Error creando sesión CUDA: {e}")
        else:
            print("❌ CUDA GPU no disponible")
            
    except Exception as e:
        print(f"❌ Error probando GPU: {e}")

def create_gpu_test_script():
    """Crear script de prueba GPU"""
    print("\n📝 CREANDO SCRIPT DE PRUEBA GPU")
    print("=" * 50)
    
    test_script = '''#!/usr/bin/env python3
"""
Script de prueba GPU mejorado
"""

import os
import sys

# Configurar variables de entorno
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_gpu_detailed():
    """Prueba detallada de GPU"""
    print("🔍 PRUEBA DETALLADA DE GPU")
    print("=" * 50)
    
    # ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime: {ort.__version__}")
        print(f"ONNX Runtime file: {ort.__file__}")
        providers = ort.get_available_providers()
        print(f"Providers disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA disponible en ONNX Runtime")
            
            # Probar sesión CUDA
            try:
                import numpy as np
                # Crear un modelo simple para probar
                import onnx
                from onnx import helper, numpy_helper
                
                # Crear un modelo ONNX simple
                X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
                Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
                
                node = helper.make_node('Identity', ['X'], ['Y'])
                graph = helper.make_graph([node], 'test', [X], [Y])
                model = helper.make_model(graph)
                
                # Probar con CUDA
                session = ort.InferenceSession(model.SerializeToString(), 
                                             providers=['CUDAExecutionProvider'])
                print("✅ Sesión CUDA funcionando correctamente")
                
            except Exception as e:
                print(f"⚠️ Error en sesión CUDA: {e}")
        else:
            print("❌ CUDA no disponible en ONNX Runtime")
            
    except Exception as e:
        print(f"❌ Error ONNX Runtime: {e}")
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name()}")
            print(f"PyTorch VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    except Exception as e:
        print(f"❌ Error PyTorch: {e}")
    
    # TensorFlow
    try:
        import tensorflow as tf
        print(f"TensorFlow: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPUs: {len(gpus)}")
        if gpus:
            print(f"TensorFlow GPU: {gpus[0]}")
    except Exception as e:
        print(f"❌ Error TensorFlow: {e}")

def test_face_swapper_gpu():
    """Probar face swapper con GPU"""
    print("\\n🎭 PROBANDO FACE SWAPPER CON GPU")
    print("=" * 50)
    
    try:
        import roop.processors.frame.face_swapper as face_swapper
        
        print("Cargando modelo de face swapper...")
        swapper = face_swapper.get_face_swapper()
        
        if swapper:
            print("✅ Face swapper cargado exitosamente")
            
            # Verificar proveedores
            if hasattr(swapper, 'providers'):
                print(f"Proveedores del modelo: {swapper.providers}")
                if 'CUDAExecutionProvider' in swapper.providers:
                    print("✅ Face swapper usando GPU")
                else:
                    print("⚠️ Face swapper usando CPU")
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando face swapper")
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")

if __name__ == "__main__":
    test_gpu_detailed()
    test_face_swapper_gpu()
'''
    
    with open('test_gpu_detailed.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Script de prueba creado: test_gpu_detailed.py")

def main():
    print("🔧 SOLUCIONADOR DE PROBLEMAS GPU - GOOGLE COLAB")
    print("=" * 60)
    
    # Verificar instalación actual
    check_current_installation()
    
    # Preguntar si instalar dependencias
    response = input("\n¿Instalar dependencias GPU? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        install_gpu_dependencies()
        test_gpu_after_fix()
        create_gpu_test_script()
        
        print("\n🎉 INSTALACIÓN COMPLETADA")
        print("=" * 50)
        print("Para verificar que todo funciona, ejecuta:")
        print("python test_gpu_detailed.py")
        print()
        print("Si aún hay problemas, ejecuta:")
        print("python test_gpu_force.py")
    else:
        print("❌ Instalación cancelada")

if __name__ == "__main__":
    main() 