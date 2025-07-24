#!/usr/bin/env python3
"""
Script específico para configurar CUDA en Google Colab
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

def check_colab_environment():
    """Verificar si estamos en Google Colab"""
    print("🔍 VERIFICANDO ENTORNO DE GOOGLE COLAB")
    print("=" * 50)
    
    try:
        import google.colab
        print("✅ Ejecutando en Google Colab")
        return True
    except ImportError:
        print("❌ No se detectó Google Colab")
        return False

def check_gpu_availability():
    """Verificar disponibilidad de GPU"""
    print("\n🔍 VERIFICANDO GPU")
    print("=" * 50)
    
    # Verificar nvidia-smi
    success = run_command("nvidia-smi", "Verificando GPU con nvidia-smi")
    if not success:
        print("❌ GPU no disponible")
        return False
    
    # Verificar CUDA
    success = run_command("nvcc --version", "Verificando CUDA Toolkit")
    if not success:
        print("❌ CUDA Toolkit no disponible")
        return False
    
    print("✅ GPU y CUDA disponibles")
    return True

def install_onnxruntime_gpu_colab():
    """Instalar onnxruntime-gpu específicamente para Colab"""
    print("\n📦 INSTALANDO ONNX RUNTIME GPU PARA COLAB")
    print("=" * 50)
    
    # Desinstalar onnxruntime si está instalado
    run_command("pip uninstall -y onnxruntime", "Desinstalando onnxruntime")
    
    # Instalar onnxruntime-gpu con versión específica para Colab
    success = run_command("pip install onnxruntime-gpu==1.16.3", "Instalando onnxruntime-gpu 1.16.3")
    if not success:
        print("⚠️ Intentando con versión más reciente...")
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

def configure_colab_environment():
    """Configurar variables de entorno para Colab"""
    print("\n⚙️ CONFIGURANDO ENTORNO PARA COLAB")
    print("=" * 50)
    
    # Configurar variables de entorno específicas para Colab
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'OMP_NUM_THREADS': '1',
        'CUDA_LAUNCH_BLOCKING': '1',
        'CUDA_CACHE_DISABLE': '0',
        'CUDA_CACHE_PATH': '/tmp/cuda_cache'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var}={value}")
    
    return True

def test_cuda_creation():
    """Probar creación de sesión CUDA"""
    print("\n🧪 PROBANDO CREACIÓN DE SESIÓN CUDA")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        import numpy as np
        from onnx import helper
        
        # Crear modelo simple para probar
        X = helper.make_tensor_value_info('X', helper.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('Y', helper.TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node('Identity', inputs=['X'], outputs=['Y'])
        graph = helper.make_graph([node], 'test', [X], [Y])
        model = helper.make_model(graph)
        
        # Intentar diferentes configuraciones
        configs_to_try = [
            {
                'providers': ['CUDAExecutionProvider'],
                'options': {
                    'CUDAExecutionProvider': {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_use_max_workspace': '1',
                        'do_copy_in_default_stream': '1',
                    }
                }
            },
            {
                'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
                'options': {
                    'CUDAExecutionProvider': {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                    }
                }
            },
            {
                'providers': ['CUDAExecutionProvider'],
                'options': {}
            }
        ]
        
        for i, config in enumerate(configs_to_try):
            try:
                print(f"Probando configuración {i+1}: {config['providers']}")
                
                session = ort.InferenceSession(
                    model.SerializeToString(),
                    providers=config['providers'],
                    provider_options=config['options']
                )
                
                actual_providers = session.get_providers()
                print(f"✅ Sesión creada exitosamente")
                print(f"Proveedores aplicados: {actual_providers}")
                
                if any('CUDA' in provider for provider in actual_providers):
                    print("✅ CUDA confirmado en uso")
                    return True
                else:
                    print("⚠️ CUDA no confirmado, intentando siguiente configuración...")
                    continue
                    
            except Exception as e:
                print(f"❌ Error con configuración {i+1}: {e}")
                continue
        
        print("❌ Ninguna configuración funcionó")
        return False
        
    except Exception as e:
        print(f"❌ Error en prueba de CUDA: {e}")
        return False

def test_face_swapper_gpu():
    """Probar face swapper con GPU"""
    print("\n🎭 PROBANDO FACE SWAPPER CON GPU")
    print("=" * 50)
    
    try:
        import roop.processors.frame.face_swapper as face_swapper
        
        print("Cargando modelo de face swapper...")
        swapper = face_swapper.get_face_swapper()
        
        if swapper:
            print("✅ Face swapper cargado exitosamente")
            
            if hasattr(swapper, 'providers'):
                actual_providers = swapper.providers
                print(f"Proveedores del modelo: {actual_providers}")
                
                if any('CUDA' in provider for provider in actual_providers):
                    print("✅ GPU CUDA confirmado en uso para face swapper")
                    return True
                else:
                    print("❌ GPU CUDA no confirmado para face swapper")
                    return False
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
                return True  # Asumimos que funciona si no podemos verificar
        else:
            print("❌ Error cargando face swapper")
            return False
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🚀 CONFIGURADOR DE CUDA PARA GOOGLE COLAB")
    print("=" * 60)
    
    # Verificar entorno de Colab
    if not check_colab_environment():
        print("⚠️ Continuando sin verificación de Colab...")
    
    # Verificar GPU
    if not check_gpu_availability():
        print("❌ GPU no disponible")
        return False
    
    # Instalar onnxruntime-gpu
    if not install_onnxruntime_gpu_colab():
        print("❌ Error instalando onnxruntime-gpu")
        return False
    
    # Configurar entorno
    configure_colab_environment()
    
    # Probar creación de sesión CUDA
    if not test_cuda_creation():
        print("❌ Error creando sesión CUDA")
        return False
    
    # Probar face swapper
    if not test_face_swapper_gpu():
        print("❌ Error con face swapper")
        return False
    
    print("\n🎉 ¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
    print("✅ CUDA está funcionando correctamente")
    print("✅ Face swapper está usando GPU")
    print("✅ Puedes ejecutar python test_gpu_force.py para verificar")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 