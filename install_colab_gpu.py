#!/usr/bin/env python3
"""
Script de instalación optimizado para Google Colab con GPU T4
Instala todas las dependencias necesarias para faceswap con GPU
"""

import os
import sys
import subprocess
import platform

def check_colab_environment():
    """Verificar si estamos en Google Colab"""
    try:
        import google.colab
        print("✅ Detectado Google Colab")
        return True
    except ImportError:
        print("⚠️ No se detectó Google Colab")
        return False

def check_gpu_availability():
    """Verificar disponibilidad de GPU"""
    print("\n🔍 VERIFICANDO GPU:")
    print("=" * 40)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detectada: {gpu_name}")
            print(f"📊 VRAM total: {vram:.1f}GB")
            
            if "T4" in gpu_name:
                print("✅ GPU T4 detectada - Configuración optimizada")
            else:
                print("⚠️ GPU diferente a T4 - Configuración genérica")
            
            return True
        else:
            print("❌ GPU no disponible")
            return False
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def install_dependencies():
    """Instalar dependencias optimizadas para GPU"""
    print("\n📦 INSTALANDO DEPENDENCIAS:")
    print("=" * 40)
    
    # Lista de dependencias optimizadas para Colab T4
    dependencies = [
        "torch==2.1.0+cu118",
        "torchvision==0.16.0+cu118",
        "torchaudio==2.1.0+cu118",
        "tensorflow==2.15.0",
        "onnxruntime-gpu==1.15.1",
        "opencv-python==4.8.0.74",
        "numpy==1.26.4",
        "insightface==0.7.3",
        "gfpgan==1.3.8",
        "basicsr==1.4.2",
        "facexlib==0.3.0",
        "filterpy==1.4.5",
        "opennsfw2==0.10.2",
        "psutil==5.9.5",
        "nvidia-ml-py3==7.352.0",
        "tqdm==4.65.0",
        "pillow==10.0.0",
        "coloredlogs==15.0.1",
        "humanfriendly==10.0"
    ]
    
    # Instalar con índice extra de PyTorch
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--extra-index-url", "https://download.pytorch.org/whl/cu118"
    ] + dependencies
    
    try:
        print("⏳ Instalando dependencias...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencias instaladas exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def setup_environment_variables():
    """Configurar variables de entorno para GPU"""
    print("\n⚙️ CONFIGURANDO VARIABLES DE ENTORNO:")
    print("=" * 40)
    
    env_vars = {
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_UNIFIED_MEMORY': '1',
        'TF_MEMORY_ALLOCATION': '0.8',
        'TF_GPU_MEMORY_LIMIT': '12'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")
    
    print("✅ Variables de entorno configuradas")

def test_gpu_setup():
    """Probar configuración de GPU"""
    print("\n🧪 PROBANDO CONFIGURACIÓN GPU:")
    print("=" * 40)
    
    try:
        import torch
        import onnxruntime as ort
        
        # Probar PyTorch
        if torch.cuda.is_available():
            print("✅ PyTorch GPU disponible")
            print(f"📊 VRAM: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        else:
            print("❌ PyTorch GPU no disponible")
        
        # Probar ONNX Runtime
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("✅ ONNX Runtime GPU disponible")
        else:
            print("❌ ONNX Runtime GPU no disponible")
        
        # Probar TensorFlow
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
        print(f"❌ Error probando GPU: {e}")
        return False

def create_optimization_script():
    """Crear script de optimización para Colab"""
    print("\n📝 CREANDO SCRIPT DE OPTIMIZACIÓN:")
    print("=" * 40)
    
    script_content = '''#!/usr/bin/env python3
"""
Script de optimización para Google Colab T4
Ejecutar antes de usar ROOP para optimizar GPU
"""

import os
import torch
import gc

def optimize_colab_gpu():
    """Optimizar GPU para Colab T4"""
    print("🚀 OPTIMIZANDO GPU PARA COLAB T4")
    
    # Configurar variables de entorno
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Limpiar memoria GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ Memoria GPU limpiada")
    
    # Garbage collection
    gc.collect()
    print("✅ Garbage collection completado")
    
    print("✅ Optimización completada")

if __name__ == '__main__':
    optimize_colab_gpu()
'''
    
    try:
        with open('optimize_colab_gpu.py', 'w') as f:
            f.write(script_content)
        print("✅ Script de optimización creado: optimize_colab_gpu.py")
        return True
    except Exception as e:
        print(f"❌ Error creando script: {e}")
        return False

def main():
    """Función principal de instalación"""
    print("🚀 INSTALACIÓN OPTIMIZADA PARA GOOGLE COLAB T4")
    print("=" * 60)
    
    # Verificar entorno
    is_colab = check_colab_environment()
    
    # Verificar GPU
    gpu_available = check_gpu_availability()
    
    if not gpu_available:
        print("❌ GPU no disponible. La instalación puede continuar pero sin optimizaciones GPU.")
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Configurar variables de entorno
    setup_environment_variables()
    
    # Probar configuración
    if not test_gpu_setup():
        print("⚠️ Advertencia: Configuración GPU no óptima")
    
    # Crear script de optimización
    create_optimization_script()
    
    print("\n✅ INSTALACIÓN COMPLETADA")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Ejecutar: python optimize_colab_gpu.py")
    print("2. Usar: python run_colab_gpu_optimized.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. Para lotes: python run_colab_gpu_optimized.py --source imagen.jpg --target carpeta_videos --batch --output-dir resultados")
    
    return True

if __name__ == '__main__':
    main() 