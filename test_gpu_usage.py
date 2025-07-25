#!/usr/bin/env python3
"""
Script para verificar el uso de GPU y diagnosticar problemas
"""

import os
import sys
import subprocess
import platform

def check_system_info():
    """Verificar información del sistema"""
    print("🖥️ INFORMACIÓN DEL SISTEMA")
    print("=" * 50)
    print(f"Sistema operativo: {platform.system()} {platform.release()}")
    print(f"Arquitectura: {platform.machine()}")
    print(f"Procesador: {platform.processor()}")
    print(f"Python: {sys.version}")
    print()

def check_gpu_info():
    """Verificar información de GPU"""
    print("🎮 INFORMACIÓN DE GPU")
    print("=" * 50)
    
    # Verificar nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi disponible")
            print(result.stdout)
        else:
            print("❌ nvidia-smi no disponible")
    except FileNotFoundError:
        print("❌ nvidia-smi no encontrado")
    
    print()

def check_python_packages():
    """Verificar paquetes de Python"""
    print("🐍 PAQUETES DE PYTHON")
    print("=" * 50)
    
    packages = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('tensorflow', 'TensorFlow'),
        ('onnxruntime', 'ONNX Runtime'),
        ('cv2', 'OpenCV'),
        ('insightface', 'InsightFace')
    ]
    
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Desconocida')
            print(f"✅ {name}: {version}")
            
            # Verificaciones específicas
            if package == 'torch':
                cuda_available = module.cuda.is_available()
                print(f"   CUDA disponible: {cuda_available}")
                if cuda_available:
                    print(f"   GPU: {module.cuda.get_device_name(0)}")
                    print(f"   Dispositivos CUDA: {module.cuda.device_count()}")
            
            elif package == 'tensorflow':
                gpus = module.config.list_physical_devices('GPU')
                print(f"   GPUs TensorFlow: {len(gpus)}")
                for i, gpu in enumerate(gpus):
                    print(f"   GPU {i}: {gpu}")
            
            elif package == 'onnxruntime':
                providers = module.get_available_providers()
                print(f"   Proveedores: {providers}")
            
        except ImportError:
            print(f"❌ {name}: No instalado")
    except Exception as e:
            print(f"⚠️ {name}: Error - {e}")
    
    print()

def check_environment_variables():
    """Verificar variables de entorno"""
    print("🔧 VARIABLES DE ENTORNO")
    print("=" * 50)
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'TF_CPP_MIN_LOG_LEVEL',
        'CUDA_LAUNCH_BLOCKING',
        'TORCH_CUDNN_V8_API_ENABLED'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'No definida')
        print(f"{var}: {value}")
    
    print()

def test_gpu_performance():
    """Probar rendimiento de GPU"""
    print("⚡ PRUEBA DE RENDIMIENTO GPU")
    print("=" * 50)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print("✅ CUDA disponible - ejecutando pruebas...")
            
            # Prueba básica de tensor
            device = torch.device('cuda')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            
            import time
            start_time = time.time()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            end_time = time.time()
            
            print(f"✅ Multiplicación de matrices: {(end_time - start_time)*1000:.2f}ms")
            
            # Verificar memoria GPU
            memory_allocated = torch.cuda.memory_allocated() / 1024**2
            memory_reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"✅ Memoria GPU: {memory_allocated:.1f}MB asignada, {memory_reserved:.1f}MB reservada")
            
        else:
            print("❌ CUDA no disponible")
            
    except Exception as e:
        print(f"❌ Error en prueba de GPU: {e}")

    print()

def check_roop_installation():
    """Verificar instalación de ROOP"""
    print("🔍 VERIFICACIÓN DE ROOP")
    print("=" * 50)
    
    # Verificar archivos principales
    files = [
        'run.py',
        'run_batch_processing.py',
        'requirements.txt',
        'roop/core.py',
        'inswapper_128.onnx'
    ]
    
    for file in files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - No encontrado")
    
    # Verificar modelo
    model_path = 'inswapper_128.onnx'
    if os.path.exists(model_path):
        size = os.path.getsize(model_path) / (1024**2)
        print(f"✅ Modelo: {size:.1f}MB")
    else:
        print("❌ Modelo no encontrado")
    
    print()

def main():
    """Función principal"""
    print("🔍 DIAGNÓSTICO COMPLETO DE GPU")
    print("=" * 60)
    print()
    
    check_system_info()
    check_gpu_info()
    check_python_packages()
    check_environment_variables()
    test_gpu_performance()
    check_roop_installation()
    
    print("=" * 60)
    print("📋 RESUMEN")
    print("=" * 60)
    print("✅ Si todas las verificaciones son exitosas, el sistema está listo")
    print("❌ Si hay errores, revisa la instalación de las dependencias")
    print("🔧 Para problemas de GPU, verifica los drivers de NVIDIA")
    print("📦 Para reinstalar: python install_roop_colab.py")
    print("=" * 60) 

if __name__ == "__main__":
    main() 