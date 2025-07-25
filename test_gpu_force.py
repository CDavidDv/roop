#!/usr/bin/env python3
"""
Script de prueba para verificar que el face swapper está usando GPU
"""

import os
import sys

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_gpu_availability():
    """Verificar disponibilidad de GPU"""
    print("🔍 VERIFICACIÓN DE GPU:")
    print("=" * 40)
    
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
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando face swapper")
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")

def test_face_enhancer_gpu():
    """Probar que el face enhancer usa GPU"""
    print("\n✨ PROBANDO FACE ENHANCER CON GPU:")
    print("=" * 40)
    
    try:
        # Importar el módulo de face enhancer
        import roop.processors.frame.face_enhancer as face_enhancer
        
        # Verificar el dispositivo
        device = face_enhancer.get_device()
        print(f"Dispositivo detectado: {device}")
        
        if device == 'cuda':
            print("✅ Face enhancer configurado para usar GPU")
        else:
            print(f"⚠️ Face enhancer usando: {device}")
            
    except Exception as e:
        print(f"❌ Error probando face enhancer: {e}")

def test_face_analyser_gpu():
    """Probar que el face analyser usa GPU"""
    print("\n🔍 PROBANDO FACE ANALYSER CON GPU:")
    print("=" * 40)
    
    try:
        # Importar el módulo de face analyser
        import roop.face_analyser as face_analyser
        
        # Verificar que el analizador se carga
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

def main():
    print("🚀 INICIANDO PRUEBAS DE GPU FORZADO")
    print("=" * 50)
    
    # Verificar disponibilidad de GPU
    test_gpu_availability()
    
    # Probar face swapper
    test_face_swapper_gpu()
    
    # Probar face enhancer
    test_face_enhancer_gpu()
    
    # Probar face analyser
    test_face_analyser_gpu()
    
    print("\n🎉 PRUEBAS COMPLETADAS")
    print("=" * 50) 

if __name__ == "__main__":
    main() 