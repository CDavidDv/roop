#!/usr/bin/env python3
"""
Script para verificar el uso de GPU en cada procesador
"""

import os
import sys

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_gpu_providers():
    """Verificar proveedores de GPU disponibles"""
    print("🔍 VERIFICACIÓN DE PROVEEDORES GPU")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible para ONNX Runtime")
        else:
            print("❌ CUDA GPU no disponible para ONNX Runtime")
    except Exception as e:
        print(f"❌ Error ONNX Runtime: {e}")
    
    try:
        import torch
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name()}")
            print(f"PyTorch VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    except Exception as e:
        print(f"❌ Error PyTorch: {e}")

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
            # Verificar qué proveedores está usando
            if hasattr(swapper, 'providers'):
                print(f"Proveedores del modelo: {swapper.providers}")
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando face swapper")
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")

def test_face_enhancer_gpu():
    """Probar face enhancer con GPU"""
    print("\n✨ PROBANDO FACE ENHANCER CON GPU")
    print("=" * 50)
    
    try:
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
    """Probar face analyser con GPU"""
    print("\n🔍 PROBANDO FACE ANALYSER CON GPU")
    print("=" * 50)
    
    try:
        import roop.face_analyser as face_analyser
        
        # Verificar que el analizador se carga con GPU
        print("Cargando analizador de rostros...")
        analyser = face_analyser.get_face_analyser()
        
        if analyser:
            print("✅ Analizador de rostros cargado exitosamente")
            # Verificar qué proveedores está usando
            if hasattr(analyser, 'providers'):
                print(f"Proveedores del analizador: {analyser.providers}")
            else:
                print("Analizador cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando analizador de rostros")
            
    except Exception as e:
        print(f"❌ Error probando face analyser: {e}")

def test_gpu_memory_usage():
    """Verificar uso de memoria GPU"""
    print("\n💾 VERIFICACIÓN DE MEMORIA GPU")
    print("=" * 50)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
            print(f"VRAM Usado: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            print(f"VRAM Reservado: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")
            print(f"VRAM Libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3:.2f}GB")
        else:
            print("❌ CUDA no disponible")
    except Exception as e:
        print(f"❌ Error verificando memoria GPU: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO VERIFICACIÓN COMPLETA DE GPU")
    print("=" * 60)
    
    # Verificar proveedores de GPU
    test_gpu_providers()
    
    # Verificar memoria GPU
    test_gpu_memory_usage()
    
    # Probar face swapper
    test_face_swapper_gpu()
    
    # Probar face enhancer
    test_face_enhancer_gpu()
    
    # Probar face analyser
    test_face_analyser_gpu()
    
    print("\n🎉 VERIFICACIÓN COMPLETADA")
    print("=" * 60) 