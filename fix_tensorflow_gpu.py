#!/usr/bin/env python3
"""
Script para arreglar TensorFlow GPU en Google Colab
"""

import os
import sys
import subprocess

def check_cuda_installation():
    """Verificar instalación de CUDA"""
    print("🔍 VERIFICANDO INSTALACIÓN CUDA:")
    print("=" * 40)
    
    try:
        # Verificar nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi disponible")
            print(result.stdout)
        else:
            print("❌ nvidia-smi no disponible")
    except:
        print("❌ nvidia-smi no encontrado")
    
    try:
        # Verificar CUDA
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA disponible")
            print(result.stdout)
        else:
            print("❌ CUDA no disponible")
    except:
        print("❌ CUDA no encontrado")

def reinstall_tensorflow_gpu():
    """Reinstalar TensorFlow con soporte GPU"""
    print("\n📦 REINSTALANDO TENSORFLOW GPU:")
    print("=" * 40)
    
    try:
        # Desinstalar TensorFlow actual
        print("⏳ Desinstalando TensorFlow actual...")
        cmd1 = [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-gpu"]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar TensorFlow compatible con GPU
        print("⏳ Instalando TensorFlow 2.13.0 con GPU...")
        cmd2 = [
            sys.executable, "-m", "pip", "install",
            "tensorflow==2.13.0"
        ]
        result = subprocess.run(cmd2, check=True, capture_output=True, text=True)
        print("✅ TensorFlow GPU instalado")
        
        # Instalar dependencias adicionales
        print("⏳ Instalando dependencias GPU...")
        cmd3 = [
            sys.executable, "-m", "pip", "install",
            "nvidia-ml-py3", "pynvml"
        ]
        subprocess.run(cmd3, check=True, capture_output=True, text=True)
        print("✅ Dependencias GPU instaladas")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando TensorFlow GPU: {e}")
        return False

def install_colab_gpu_dependencies():
    """Instalar dependencias específicas de Colab"""
    print("\n🔧 INSTALANDO DEPENDENCIAS COLAB:")
    print("=" * 40)
    
    try:
        # Instalar CUDA toolkit para Colab
        print("⏳ Instalando CUDA toolkit...")
        cmd1 = [
            sys.executable, "-m", "pip", "install",
            "cudatoolkit==11.8.0"
        ]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar cuDNN
        print("⏳ Instalando cuDNN...")
        cmd2 = [
            sys.executable, "-m", "pip", "install",
            "cudnn==8.7.0.84"
        ]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        print("✅ Dependencias Colab instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def test_tensorflow_gpu():
    """Probar TensorFlow GPU"""
    print("\n🧪 PROBANDO TENSORFLOW GPU:")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Verificar dispositivos
        devices = tf.config.list_physical_devices()
        print(f"📱 Dispositivos disponibles: {len(devices)}")
        for device in devices:
            print(f"  - {device}")
        
        # Verificar GPU específicamente
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 GPU disponible: {len(gpus)} dispositivos")
            for gpu in gpus:
                print(f"  - {gpu}")
            
            # Configurar GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU configurado para crecimiento de memoria")
            
            # Probar operación en GPU
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print(f"✅ Operación GPU exitosa: {c}")
            
            return True
        else:
            print("❌ GPU no disponible")
            return False
            
    except Exception as e:
        print(f"❌ Error probando TensorFlow GPU: {e}")
        return False

def create_gpu_test_script():
    """Crear script de prueba GPU"""
    print("\n📝 CREANDO SCRIPT DE PRUEBA GPU:")
    print("=" * 40)
    
    test_script = '''#!/usr/bin/env python3
"""
Script para probar GPU con TensorFlow
"""

import os
import tensorflow as tf

# Configurar variables de entorno
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def test_gpu():
    print("🧪 PROBANDO GPU CON TENSORFLOW")
    print("=" * 40)
    
    print(f"TensorFlow version: {tf.__version__}")
    
    # Listar dispositivos
    devices = tf.config.list_physical_devices()
    print(f"Dispositivos disponibles: {len(devices)}")
    for device in devices:
        print(f"  - {device}")
    
    # Verificar GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU disponible: {len(gpus)} dispositivos")
        for gpu in gpus:
            print(f"  - {gpu}")
        
        # Configurar GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configurado para crecimiento de memoria")
        
        # Probar operación
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print(f"Operación GPU exitosa: {c}")
        
        return True
    else:
        print("GPU no disponible")
        return False

if __name__ == '__main__':
    test_gpu()
'''
    
    try:
        with open('test_gpu_tensorflow.py', 'w') as f:
            f.write(test_script)
        print("✅ Script de prueba creado: test_gpu_tensorflow.py")
        return True
    except Exception as e:
        print(f"❌ Error creando script: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO TENSORFLOW GPU EN COLAB")
    print("=" * 60)
    
    # Verificar instalación actual
    check_cuda_installation()
    
    # Reinstalar TensorFlow GPU
    if not reinstall_tensorflow_gpu():
        print("❌ Error reinstalando TensorFlow")
        return False
    
    # Instalar dependencias Colab
    install_colab_gpu_dependencies()
    
    # Crear script de prueba
    create_gpu_test_script()
    
    # Probar TensorFlow GPU
    if not test_tensorflow_gpu():
        print("❌ Error: TensorFlow GPU no funciona")
        return False
    
    print("\n✅ TENSORFLOW GPU CONFIGURADO EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Probar GPU: python test_gpu_tensorflow.py")
    print("2. Si funciona, ejecutar: python force_gpu_usage.py")
    print("3. Luego procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    
    return True

if __name__ == '__main__':
    main() 