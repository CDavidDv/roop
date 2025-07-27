#!/usr/bin/env python3
"""
Script de optimización para Google Colab T4
Ejecutar antes de usar ROOP para optimizar GPU
"""

import os
import torch
import gc
import time

def optimize_colab_gpu():
    """Optimizar GPU para Colab T4"""
    print("🚀 OPTIMIZANDO GPU PARA COLAB T4")
    print("=" * 50)
    
    # Configurar variables de entorno
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
    os.environ['TF_MEMORY_ALLOCATION'] = '0.8'
    os.environ['TF_GPU_MEMORY_LIMIT'] = '12'
    
    print("✅ Variables de entorno configuradas")
    
    # Verificar GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✅ GPU detectada: {gpu_name}")
        print(f"📊 VRAM total: {vram_total:.1f}GB")
        
        # Limpiar memoria GPU
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ Memoria GPU limpiada")
        
        # Configurar optimizaciones PyTorch
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("✅ Optimizaciones PyTorch configuradas")
        
        # Mostrar memoria inicial
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"📊 Memoria inicial - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
    else:
        print("❌ GPU no disponible")
        return False
    
    # Garbage collection
    gc.collect()
    print("✅ Garbage collection completado")
    
    # Esperar un momento para estabilizar
    time.sleep(2)
    
    # Verificar memoria final
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"📊 Memoria final - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    print("✅ Optimización completada")
    return True

def check_optimization():
    """Verificar que la optimización fue exitosa"""
    print("\n🔍 VERIFICANDO OPTIMIZACIÓN:")
    print("=" * 40)
    
    try:
        import onnxruntime as ort
        
        # Verificar ONNX Runtime
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime GPU disponible")
        else:
            print("❌ ONNX Runtime GPU no disponible")
        
        # Verificar PyTorch
        if torch.cuda.is_available():
            print("✅ PyTorch GPU disponible")
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"📊 VRAM en uso: {allocated:.2f}GB")
        else:
            print("❌ PyTorch GPU no disponible")
        
        # Verificar TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✅ TensorFlow GPU disponible: {len(gpus)} dispositivos")
            else:
                print("❌ TensorFlow GPU no disponible")
        except Exception as e:
            print(f"⚠️ Error TensorFlow: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando optimización: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 OPTIMIZACIÓN GPU COLAB T4")
    print("=" * 60)
    
    # Optimizar GPU
    if optimize_colab_gpu():
        print("\n✅ Optimización completada exitosamente")
        
        # Verificar optimización
        check_optimization()
        
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Usar: python run_colab_gpu_optimized.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
        print("2. Para lotes: python run_colab_gpu_optimized.py --source imagen.jpg --target carpeta_videos --batch --output-dir resultados")
        print("3. Monitorear: python monitor_gpu.py")
        
    else:
        print("\n❌ Error en la optimización")
        print("Verifica que tienes GPU T4 asignada en Colab")

if __name__ == '__main__':
    main() 