#!/usr/bin/env python3
"""
Script para optimizar la configuración de GPU y verificar compatibilidad
"""

import os
import sys
import warnings
import subprocess
import platform

# Suprimir warnings
warnings.filterwarnings('ignore')

def check_cuda_installation():
    """Verificar instalación de CUDA"""
    print("🔍 Verificando instalación de CUDA...")
    
    try:
        # Verificar nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi disponible")
            print(result.stdout)
            return True
        else:
            print("❌ nvidia-smi no disponible")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi no encontrado")
        return False

def check_pytorch_gpu():
    """Verificar PyTorch con GPU"""
    print("\n🔍 Verificando PyTorch GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ PyTorch GPU disponible: {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   • GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Test básico de GPU
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000).to(device)
            y = torch.mm(x, x)
            print("✅ Test de GPU PyTorch exitoso")
            return True
        else:
            print("❌ PyTorch GPU no disponible")
            return False
    except ImportError:
        print("❌ PyTorch no instalado")
        return False
    except Exception as e:
        print(f"❌ Error en test de PyTorch GPU: {e}")
        return False

def check_onnx_gpu():
    """Verificar ONNX Runtime GPU"""
    print("\n🔍 Verificando ONNX Runtime GPU...")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"Proveedores disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime CUDA disponible")
            
            # Test básico de ONNX GPU
            import numpy as np
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Crear un modelo simple para test
            import onnx
            from onnx import helper, numpy_helper
            
            # Crear un modelo simple: y = x + 1
            X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 1])
            Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 1])
            
            node = helper.make_node('Add', ['X', 'one'], ['Y'])
            one = numpy_helper.from_array(np.array([1.0], dtype=np.float32), 'one')
            
            graph = helper.make_graph([node], 'test', [X], [Y], [one])
            model = helper.make_model(graph)
            
            # Test con CUDA
            session = ort.InferenceSession(
                model.SerializeToString(),
                session_options,
                providers=['CUDAExecutionProvider']
            )
            
            input_data = np.array([[2.0]], dtype=np.float32)
            result = session.run(['Y'], {'X': input_data})
            print(f"✅ Test ONNX GPU exitoso: {result[0]}")
            return True
        else:
            print("❌ ONNX Runtime CUDA no disponible")
            return False
    except ImportError:
        print("❌ ONNX Runtime no instalado")
        return False
    except Exception as e:
        print(f"❌ Error en test de ONNX GPU: {e}")
        return False

def check_tensorflow_gpu():
    """Verificar TensorFlow GPU"""
    print("\n🔍 Verificando TensorFlow GPU...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✅ TensorFlow GPU disponible: {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"   • {gpu}")
            
            # Test básico de TensorFlow GPU
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✅ Test TensorFlow GPU exitoso: {c.numpy()}")
            return True
        else:
            print("❌ TensorFlow GPU no disponible")
            return False
    except ImportError:
        print("❌ TensorFlow no instalado")
        return False
    except Exception as e:
        print(f"❌ Error en test de TensorFlow GPU: {e}")
        return False

def optimize_environment():
    """Optimizar variables de entorno para GPU"""
    print("\n🔧 Optimizando variables de entorno...")
    
    # Variables de entorno para optimización GPU
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'OMP_NUM_THREADS': '1',
        'CUDA_LAUNCH_BLOCKING': '0',
        'TORCH_CUDA_ARCH_LIST': '7.5;8.0;8.6',  # Arquitecturas comunes
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"   • {var} = {value}")
    
    print("✅ Variables de entorno optimizadas")

def check_system_info():
    """Verificar información del sistema"""
    print("\n💻 Información del sistema:")
    print(f"   • Sistema operativo: {platform.system()} {platform.release()}")
    print(f"   • Arquitectura: {platform.machine()}")
    print(f"   • Python: {sys.version}")
    
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / 1024**3
        print(f"   • CPU: {cpu_count} núcleos")
        print(f"   • RAM: {memory_gb:.1f}GB")
    except ImportError:
        print("   • psutil no disponible")

def main():
    """Función principal"""
    print("🚀 VERIFICACIÓN Y OPTIMIZACIÓN DE GPU")
    print("=" * 50)
    
    # Verificar información del sistema
    check_system_info()
    
    # Optimizar entorno
    optimize_environment()
    
    # Verificar componentes GPU
    cuda_ok = check_cuda_installation()
    pytorch_ok = check_pytorch_gpu()
    onnx_ok = check_onnx_gpu()
    tf_ok = check_tensorflow_gpu()
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 50)
    print(f"✅ CUDA: {'Sí' if cuda_ok else 'No'}")
    print(f"✅ PyTorch GPU: {'Sí' if pytorch_ok else 'No'}")
    print(f"✅ ONNX Runtime GPU: {'Sí' if onnx_ok else 'No'}")
    print(f"✅ TensorFlow GPU: {'Sí' if tf_ok else 'No'}")
    
    if cuda_ok and (pytorch_ok or onnx_ok):
        print("\n🎉 ¡GPU configurada correctamente para ROOP!")
        print("💡 El proyecto está optimizado para usar GPU")
    else:
        print("\n⚠️ Algunos componentes GPU no están disponibles")
        print("💡 El proyecto funcionará con CPU pero será más lento")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 