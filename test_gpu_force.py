#!/usr/bin/env python3
"""
Script de prueba para verificar que el face swapper está usando GPU
"""

import os
import sys
import subprocess

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def check_cuda_dependencies():
    """Verificar dependencias de CUDA"""
    print("🔧 VERIFICACIÓN DE DEPENDENCIAS CUDA:")
    print("=" * 50)
    
    # Verificar versión de CUDA
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA Toolkit instalado:")
            print(result.stdout.split('\n')[0])
        else:
            print("❌ CUDA Toolkit no encontrado")
    except FileNotFoundError:
        print("❌ CUDA Toolkit no encontrado (nvcc no está en PATH)")
    
    # Verificar drivers NVIDIA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Drivers NVIDIA funcionando:")
            lines = result.stdout.split('\n')
            for line in lines[:3]:  # Mostrar solo las primeras líneas
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ nvidia-smi falló")
    except FileNotFoundError:
        print("❌ nvidia-smi no encontrado")
    
    # Verificar onnxruntime-gpu
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime instalado: {ort.__version__}")
        
        # Verificar si es la versión GPU
        try:
            import onnxruntime.capi.onnxruntime_pybind11_state as ort_state
            print("✅ ONNX Runtime GPU disponible")
        except ImportError:
            print("❌ ONNX Runtime GPU no disponible (¿instalaste onnxruntime-gpu?)")
            
    except ImportError:
        print("❌ ONNX Runtime no instalado")

def test_gpu_availability():
    """Verificar disponibilidad de GPU"""
    print("\n🔍 VERIFICACIÓN DE GPU:")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible para ONNX Runtime")
        else:
            print("❌ CUDA GPU no disponible para ONNX Runtime")
            
        if 'TensorrtExecutionProvider' in providers:
            print("✅ TensorRT disponible")
        else:
            print("❌ TensorRT no disponible")
            
    except Exception as e:
        print(f"❌ Error ONNX Runtime: {e}")
    
    try:
        import torch
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name()}")
            print(f"PyTorch VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            print(f"PyTorch VRAM Total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    except Exception as e:
        print(f"❌ Error PyTorch: {e}")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPU devices: {len(gpus)}")
        if gpus:
            print(f"TensorFlow GPU: {gpus[0]}")
    except Exception as e:
        print(f"❌ Error TensorFlow: {e}")

def test_face_swapper_gpu():
    """Probar que el face swapper usa GPU"""
    print("\n🎭 PROBANDO FACE SWAPPER CON GPU:")
    print("=" * 40)
    
    try:
        # Importar el módulo de face swapper
        import roop.processors.frame.face_swapper as face_swapper
        
        # Verificar que el modelo se carga con GPU
        print("Cargando modelo de face swapper...")
        swapper = face_swapper.get_face_swapper()
        
        if swapper:
            print("✅ Face swapper cargado exitosamente")
            # Verificar qué proveedores está usando
            if hasattr(swapper, 'providers'):
                print(f"Proveedores del modelo: {swapper.providers}")
                
                # Verificar si realmente está usando GPU
                if any('CUDA' in provider for provider in swapper.providers):
                    print("✅ GPU CUDA confirmado en uso")
                else:
                    print("❌ GPU CUDA no confirmado en uso")
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando face swapper")
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")
        import traceback
        traceback.print_exc()

def test_face_enhancer_gpu():
    """Probar que el face enhancer usa GPU"""
    print("\n✨ PROBANDO FACE ENHANCER CON GPU:")
    print("=" * 40)
    
    try:
        # Importar el módulo de face enhancer
        import roop.processors.frame.face_enhancer as face_enhancer
        
        # Verificar que el dispositivo se detecta correctamente
        device = face_enhancer.get_device()
        print(f"Dispositivo detectado: {device}")
        
        if device == 'cuda':
            print("✅ Face enhancer configurado para usar GPU")
        elif device == 'mps':
            print("✅ Face enhancer configurado para usar CoreML")
        else:
            print("⚠️ Face enhancer usando CPU")
            
    except Exception as e:
        print(f"❌ Error probando face enhancer: {e}")

def test_face_analyser_gpu():
    """Probar que el analizador de rostros usa GPU"""
    print("\n🔍 PROBANDO FACE ANALYSER CON GPU:")
    print("=" * 40)
    
    try:
        # Importar el módulo de face analyser
        import roop.face_analyser as face_analyser
        
        # Verificar que el analizador se carga con GPU
        print("Cargando analizador de rostros...")
        analyser = face_analyser.get_face_analyser()
        
        if analyser:
            print("✅ Analizador de rostros cargado exitosamente")
            # Verificar qué proveedores está usando
            if hasattr(analyser, 'providers'):
                print(f"Proveedores del analizador: {analyser.providers}")
                
                # Verificar si realmente está usando GPU
                if any('CUDA' in provider for provider in analyser.providers):
                    print("✅ GPU CUDA confirmado en uso")
                else:
                    print("❌ GPU CUDA no confirmado en uso")
            else:
                print("Analizador cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando analizador de rostros")
            
    except Exception as e:
        print(f"❌ Error probando face analyser: {e}")

def test_manual_cuda_creation():
    """Probar creación manual de CUDAExecutionProvider"""
    print("\n🔧 PROBANDO CREACIÓN MANUAL DE CUDA:")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        
        # Intentar crear sesión con CUDA
        print("Intentando crear sesión ONNX Runtime con CUDA...")
        
        # Crear un modelo simple para probar
        import numpy as np
        from onnx import helper, numpy_helper
        
        # Crear un modelo ONNX simple
        X = helper.make_tensor_value_info('X', helper.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('Y', helper.TensorProto.FLOAT, [1, 3, 224, 224])
        
        # Crear un nodo de identidad
        node = helper.make_node('Identity', inputs=['X'], outputs=['Y'])
        
        # Crear el grafo
        graph = helper.make_graph([node], 'test', [X], [Y])
        
        # Crear el modelo
        model = helper.make_model(graph)
        
        # Intentar crear sesión con diferentes configuraciones
        providers_to_try = [
            ['CUDAExecutionProvider'],
            ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
        ]
        
        for providers in providers_to_try:
            try:
                print(f"Probando con proveedores: {providers}")
                
                provider_options = {}
                if 'CUDAExecutionProvider' in providers:
                    provider_options['CUDAExecutionProvider'] = {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                        'cudnn_conv_use_max_workspace': '1',
                        'do_copy_in_default_stream': '1',
                    }
                
                session = ort.InferenceSession(
                    model.SerializeToString(),
                    providers=providers,
                    provider_options=provider_options
                )
                
                print(f"✅ Sesión creada exitosamente con: {session.get_providers()}")
                break
                
            except Exception as e:
                print(f"❌ Error con {providers}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error en prueba manual: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO PRUEBAS DE GPU FORZADO")
    print("=" * 50)
    
    # Verificar dependencias de CUDA
    check_cuda_dependencies()
    
    # Verificar disponibilidad de GPU
    test_gpu_availability()
    
    # Probar creación manual de CUDA
    test_manual_cuda_creation()
    
    # Probar face swapper
    test_face_swapper_gpu()
    
    # Probar face enhancer
    test_face_enhancer_gpu()
    
    # Probar face analyser
    test_face_analyser_gpu()
    
    print("\n🎉 PRUEBAS COMPLETADAS")
    print("=" * 50) 