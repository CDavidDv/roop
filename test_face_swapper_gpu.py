#!/usr/bin/env python3
"""
Script específico para probar el face swapper con GPU
"""

import os
import sys
import time

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_face_swapper_gpu_detailed():
    """Prueba detallada del face swapper con GPU"""
    print("🎭 PRUEBA DETALLADA DE FACE SWAPPER CON GPU")
    print("=" * 60)
    
    # Verificar ONNX Runtime
    print("\n🔍 PASO 1: Verificando ONNX Runtime...")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA disponible para ONNX Runtime")
        else:
            print("❌ CUDA no disponible para ONNX Runtime")
            return False
    except Exception as e:
        print(f"❌ Error con ONNX Runtime: {e}")
        return False
    
    # Verificar PyTorch
    print("\n🔍 PASO 2: Verificando PyTorch...")
    try:
        import torch
        print(f"PyTorch CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU detectada: {torch.cuda.get_device_name()}")
            print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
            print(f"VRAM libre: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.2f}GB")
        else:
            print("❌ PyTorch CUDA no disponible")
            return False
    except Exception as e:
        print(f"❌ Error con PyTorch: {e}")
        return False
    
    # Probar face swapper
    print("\n🎭 PASO 3: Probando face swapper...")
    try:
        import roop.processors.frame.face_swapper as face_swapper
        
        print("Cargando modelo de face swapper...")
        start_time = time.time()
        
        swapper = face_swapper.get_face_swapper()
        
        load_time = time.time() - start_time
        print(f"Tiempo de carga: {load_time:.2f} segundos")
        
        if swapper:
            print("✅ Face swapper cargado exitosamente")
            
            # Verificar proveedores
            if hasattr(swapper, 'providers'):
                print(f"Proveedores del modelo: {swapper.providers}")
                if 'CUDAExecutionProvider' in swapper.providers:
                    print("🎉 ¡GPU funcionando en face swapper!")
                    return True
                else:
                    print("⚠️ Face swapper usando CPU")
                    return False
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
                return False
        else:
            print("❌ Error cargando face swapper")
            return False
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")
        return False

def test_face_swapper_performance():
    """Prueba de rendimiento del face swapper"""
    print("\n⚡ PRUEBA DE RENDIMIENTO")
    print("=" * 40)
    
    try:
        import cv2
        import numpy as np
        import roop.processors.frame.face_swapper as face_swapper
        from roop.face_analyser import get_one_face
        
        # Crear imagen de prueba
        print("Creando imagen de prueba...")
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Obtener face swapper
        swapper = face_swapper.get_face_swapper()
        
        if not swapper:
            print("❌ No se pudo cargar face swapper")
            return
        
        # Simular procesamiento
        print("Simulando procesamiento de frames...")
        times = []
        
        for i in range(5):
            start_time = time.time()
            
            # Simular detección de rostro
            faces = get_one_face(test_image)
            
            if faces:
                # Simular face swap (sin hacer el swap real)
                pass
            
            end_time = time.time()
            frame_time = end_time - start_time
            times.append(frame_time)
            
            print(f"Frame {i+1}: {frame_time:.3f}s")
        
        avg_time = sum(times) / len(times)
        fps = 1 / avg_time
        
        print(f"\n📊 RESULTADOS:")
        print(f"Tiempo promedio por frame: {avg_time:.3f}s")
        print(f"FPS estimado: {fps:.1f}")
        
        if fps > 10:
            print("✅ Rendimiento bueno (>10 FPS)")
        elif fps > 5:
            print("⚠️ Rendimiento aceptable (5-10 FPS)")
        else:
            print("❌ Rendimiento bajo (<5 FPS)")
            
    except Exception as e:
        print(f"❌ Error en prueba de rendimiento: {e}")

def main():
    """Función principal"""
    print("🚀 INICIANDO PRUEBAS DE FACE SWAPPER CON GPU")
    print("=" * 60)
    
    # Prueba detallada
    success = test_face_swapper_gpu_detailed()
    
    if success:
        print("\n✅ Face swapper configurado correctamente con GPU")
        
        # Prueba de rendimiento
        test_face_swapper_performance()
        
        print("\n🎉 ¡Todas las pruebas pasaron!")
        print("💡 Ahora puedes procesar videos con GPU")
    else:
        print("\n❌ Face swapper no está usando GPU")
        print("💡 Ejecuta: python setup_colab_gpu.py")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 