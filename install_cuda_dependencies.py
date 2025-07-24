#!/usr/bin/env python3
"""
Script para instalar dependencias de CUDA en Google Colab
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Ejecutar comando y mostrar resultado"""
    print(f"\n🔄 {description}")
    print(f"Comando: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        print(f"Código de salida: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error ejecutando comando: {e}")
        return False

def check_cuda_installation():
    """Verificar instalación de CUDA"""
    print("🔍 VERIFICANDO INSTALACIÓN DE CUDA")
    print("=" * 50)
    
    # Verificar CUDA Toolkit
    success = run_command("nvcc --version", "Verificando CUDA Toolkit")
    if not success:
        print("❌ CUDA Toolkit no encontrado")
        return False
    
    # Verificar drivers NVIDIA
    success = run_command("nvidia-smi", "Verificando drivers NVIDIA")
    if not success:
        print("❌ Drivers NVIDIA no encontrados")
        return False
    
    print("✅ CUDA y drivers NVIDIA funcionando correctamente")
    return True

def install_onnxruntime_gpu():
    """Instalar onnxruntime-gpu"""
    print("\n📦 INSTALANDO ONNX RUNTIME GPU")
    print("=" * 50)
    
    # Desinstalar onnxruntime si está instalado
    run_command("pip uninstall -y onnxruntime", "Desinstalando onnxruntime")
    
    # Instalar onnxruntime-gpu
    success = run_command("pip install onnxruntime-gpu", "Instalando onnxruntime-gpu")
    if not success:
        print("❌ Error instalando onnxruntime-gpu")
        return False
    
    # Verificar instalación
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime GPU instalado: {ort.__version__}")
        
        # Verificar proveedores disponibles
        providers = ort.get_available_providers()
        print(f"Proveedores disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDAExecutionProvider disponible")
        else:
            print("❌ CUDAExecutionProvider no disponible")
            
        return True
    except ImportError:
        print("❌ Error importando onnxruntime-gpu")
        return False

def install_tensorrt():
    """Instalar TensorRT si es posible"""
    print("\n🚀 INSTALANDO TENSORRT")
    print("=" * 50)
    
    # Intentar instalar TensorRT
    success = run_command("pip install tensorrt", "Instalando TensorRT")
    if success:
        print("✅ TensorRT instalado")
    else:
        print("⚠️ TensorRT no se pudo instalar (puede no estar disponible)")
    
    return True

def install_cudnn():
    """Instalar cuDNN si es posible"""
    print("\n🧠 INSTALANDO CUDNN")
    print("=" * 50)
    
    # Intentar instalar cuDNN
    success = run_command("pip install nvidia-cudnn-cu12", "Instalando cuDNN")
    if success:
        print("✅ cuDNN instalado")
    else:
        print("⚠️ cuDNN no se pudo instalar (puede no estar disponible)")
    
    return True

def configure_environment():
    """Configurar variables de entorno"""
    print("\n⚙️ CONFIGURANDO VARIABLES DE ENTORNO")
    print("=" * 50)
    
    # Configurar variables de entorno
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'OMP_NUM_THREADS': '1',
        'CUDA_LAUNCH_BLOCKING': '1'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var}={value}")
    
    return True

def test_gpu_after_installation():
    """Probar GPU después de la instalación"""
    print("\n🧪 PROBANDO GPU DESPUÉS DE INSTALACIÓN")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDAExecutionProvider disponible")
            
            # Probar creación de sesión con CUDA
            import numpy as np
            from onnx import helper
            
            # Crear modelo simple
            X = helper.make_tensor_value_info('X', helper.TensorProto.FLOAT, [1, 3, 224, 224])
            Y = helper.make_tensor_value_info('Y', helper.TensorProto.FLOAT, [1, 3, 224, 224])
            node = helper.make_node('Identity', inputs=['X'], outputs=['Y'])
            graph = helper.make_graph([node], 'test', [X], [Y])
            model = helper.make_model(graph)
            
            # Intentar crear sesión con CUDA
            try:
                session = ort.InferenceSession(
                    model.SerializeToString(),
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                print("✅ Sesión ONNX Runtime con CUDA creada exitosamente")
                print(f"Proveedores aplicados: {session.get_providers()}")
                return True
            except Exception as e:
                print(f"❌ Error creando sesión con CUDA: {e}")
                return False
        else:
            print("❌ CUDAExecutionProvider no disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error probando GPU: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALADOR DE DEPENDENCIAS CUDA PARA GOOGLE COLAB")
    print("=" * 60)
    
    # Verificar instalación de CUDA
    if not check_cuda_installation():
        print("❌ CUDA no está instalado correctamente")
        return False
    
    # Instalar onnxruntime-gpu
    if not install_onnxruntime_gpu():
        print("❌ Error instalando onnxruntime-gpu")
        return False
    
    # Instalar TensorRT
    install_tensorrt()
    
    # Instalar cuDNN
    install_cudnn()
    
    # Configurar variables de entorno
    configure_environment()
    
    # Probar GPU después de instalación
    if test_gpu_after_installation():
        print("\n🎉 ¡INSTALACIÓN COMPLETADA EXITOSAMENTE!")
        print("✅ CUDA y onnxruntime-gpu están funcionando correctamente")
        print("✅ Puedes ejecutar python test_gpu_force.py para verificar")
        return True
    else:
        print("\n❌ LA INSTALACIÓN NO SE COMPLETÓ CORRECTAMENTE")
        print("⚠️ Revisa los errores anteriores")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 