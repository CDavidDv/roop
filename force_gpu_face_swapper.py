#!/usr/bin/env python3
"""
Script para forzar el uso de GPU en el face swapper
Específicamente optimizado para Tesla T4 en Google Colab
"""

import os
import sys
import gc
import torch
import onnxruntime as ort

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'

def check_gpu_availability():
    """Verificar disponibilidad de GPU"""
    print("🔍 VERIFICACIÓN DE GPU:")
    print("=" * 50)
    
    try:
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA GPU disponible para ONNX Runtime")
        else:
            print("❌ CUDA GPU no disponible para ONNX Runtime")
    except Exception as e:
        print(f"❌ Error ONNX Runtime: {e}")
    
    try:
        print(f"PyTorch CUDA: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch GPU: {torch.cuda.get_device_name()}")
            print(f"PyTorch VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            print(f"PyTorch VRAM Total: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
    except Exception as e:
        print(f"❌ Error PyTorch: {e}")

def force_gpu_face_swapper():
    """Forzar el uso de GPU en el face swapper"""
    print("\n🎭 FORZANDO GPU EN FACE SWAPPER:")
    print("=" * 50)
    
    try:
        # Importar el módulo de face swapper
        import roop.processors.frame.face_swapper as face_swapper
        
        # Limpiar cualquier instancia previa
        face_swapper.FACE_SWAPPER = None
        gc.collect()
        
        # Verificar que el modelo se carga con GPU
        print("Cargando modelo de face swapper con GPU forzado...")
        swapper = face_swapper.get_face_swapper()
        
        if swapper:
            print("✅ Face swapper cargado exitosamente")
            # Verificar qué proveedores está usando
            if hasattr(swapper, 'providers'):
                print(f"Proveedores del modelo: {swapper.providers}")
                if 'CUDAExecutionProvider' in str(swapper.providers):
                    print("✅ GPU CUDA detectado en el modelo")
                else:
                    print("⚠️ GPU CUDA no detectado en el modelo")
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
        else:
            print("❌ Error cargando face swapper")
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")

def test_gpu_performance():
    """Probar rendimiento de GPU"""
    print("\n⚡ PROBANDO RENDIMIENTO GPU:")
    print("=" * 50)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Crear tensor de prueba en GPU
            device = torch.device('cuda')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            
            # Medir tiempo de operación
            import time
            start_time = time.time()
            z = torch.mm(x, y)
            torch.cuda.synchronize()
            end_time = time.time()
            
            print(f"✅ Operación GPU completada en {end_time - start_time:.4f} segundos")
            print(f"📊 VRAM usada: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        else:
            print("❌ GPU no disponible para PyTorch")
            
    except Exception as e:
        print(f"❌ Error probando rendimiento GPU: {e}")

def optimize_for_tesla_t4():
    """Optimizaciones específicas para Tesla T4"""
    print("\n🚀 OPTIMIZACIONES PARA TESLA T4:")
    print("=" * 50)
    
    try:
        # Configurar PyTorch para Tesla T4
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("✅ Optimizaciones PyTorch aplicadas")
            
            # Configurar memoria GPU
            torch.cuda.empty_cache()
            print("✅ Memoria GPU limpiada")
            
            # Verificar configuración
            print(f"📊 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
            print(f"📊 VRAM usada: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            
    except Exception as e:
        print(f"❌ Error aplicando optimizaciones: {e}")

def main():
    print("🚀 INICIANDO FORZADO DE GPU PARA FACE SWAPPER")
    print("=" * 60)
    print("🎯 Optimizado para Tesla T4 en Google Colab")
    print("=" * 60)
    
    # Verificar disponibilidad de GPU
    check_gpu_availability()
    
    # Aplicar optimizaciones para Tesla T4
    optimize_for_tesla_t4()
    
    # Probar rendimiento GPU
    test_gpu_performance()
    
    # Forzar GPU en face swapper
    force_gpu_face_swapper()
    
    print("\n🎉 PROCESO COMPLETADO")
    print("=" * 60)
    print("💡 Ahora puedes ejecutar el procesamiento por lotes:")
    print("python run_batch_processing.py --source imagen.jpg --videos video1.mp4 video2.mp4 --output-dir resultados")
    print("=" * 60)

if __name__ == "__main__":
    main() 