#!/usr/bin/env python3
"""
Script completo para arreglar GPU en Google Colab
"""

import os
import sys
import subprocess

def fix_numpy_conflict():
    """Arreglar conflicto de NumPy"""
    print("🔧 ARREGLANDO CONFLICTO NUMPY:")
    print("=" * 40)
    
    try:
        # Actualizar NumPy a versión compatible
        print("⏳ Actualizando NumPy...")
        cmd1 = [sys.executable, "-m", "pip", "install", "--upgrade", "numpy>=1.25.2"]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Actualizar SciPy
        print("⏳ Actualizando SciPy...")
        cmd2 = [sys.executable, "-m", "pip", "install", "--upgrade", "scipy"]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        print("✅ NumPy y SciPy actualizados")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error actualizando NumPy: {e}")
        return False

def install_tensorflow_colab():
    """Instalar TensorFlow específico para Colab"""
    print("\n📦 INSTALANDO TENSORFLOW PARA COLAB:")
    print("=" * 40)
    
    try:
        # Desinstalar TensorFlow actual
        print("⏳ Desinstalando TensorFlow actual...")
        cmd1 = [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-gpu"]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar TensorFlow compatible con Colab
        print("⏳ Instalando TensorFlow 2.12.0...")
        cmd2 = [
            sys.executable, "-m", "pip", "install",
            "tensorflow==2.12.0"
        ]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        print("✅ TensorFlow instalado")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando TensorFlow: {e}")
        return False

def setup_colab_gpu_environment():
    """Configurar entorno GPU para Colab"""
    print("\n🔧 CONFIGURANDO ENTORNO GPU COLAB:")
    print("=" * 40)
    
    # Variables de entorno específicas para Colab
    env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '0',
        'TF_FORCE_UNIFIED_MEMORY': '1',
        'TF_MEMORY_ALLOCATION': '0.9',
        'TF_GPU_MEMORY_LIMIT': '14',
        'TF_CUDNN_USE_AUTOTUNE': '0',
        'TF_CUDNN_DETERMINISTIC': '1',
        # Variables específicas para Colab
        'LD_LIBRARY_PATH': '/usr/local/cuda/lib64:$LD_LIBRARY_PATH',
        'CUDA_HOME': '/usr/local/cuda'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")
    
    print("✅ Variables de entorno configuradas")

def test_gpu_with_colab():
    """Probar GPU con configuración Colab"""
    print("\n🧪 PROBANDO GPU CON COLAB:")
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
        print(f"❌ Error probando GPU: {e}")
        return False

def create_colab_gpu_wrapper():
    """Crear wrapper específico para Colab"""
    print("\n📝 CREANDO WRAPPER COLAB GPU:")
    print("=" * 40)
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper para ROOP con GPU en Google Colab
"""

import os
import sys
import subprocess

# CONFIGURACIÓN ESPECÍFICA PARA COLAB
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['TF_MEMORY_ALLOCATION'] = '0.9'
os.environ['TF_GPU_MEMORY_LIMIT'] = '14'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:$LD_LIBRARY_PATH'
os.environ['CUDA_HOME'] = '/usr/local/cuda'

def check_colab_gpu():
    try:
        import tensorflow as tf
        print(f"🎮 TensorFlow version: {tf.__version__}")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU detectada: {len(gpus)} dispositivos")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU configurado para Colab")
            return True
        else:
            print("❌ GPU no disponible en Colab")
            return False
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ROOP CON GPU COLAB")
    print("=" * 40)
    
    # Verificar GPU
    if not check_colab_gpu():
        print("❌ No se puede continuar sin GPU")
        return False
    
    # Obtener argumentos
    args = sys.argv[1:]
    
    # Construir comando
    cmd = [sys.executable, 'run.py'] + args
    
    print(f"🚀 Ejecutando: {' '.join(cmd)}")
    print("=" * 40)
    
    try:
        # Ejecutar ROOP
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ ROOP ejecutado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando ROOP:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

if __name__ == '__main__':
    main()
'''
    
    try:
        with open('run_roop_colab_gpu.py', 'w') as f:
            f.write(wrapper_content)
        print("✅ Wrapper Colab GPU creado: run_roop_colab_gpu.py")
        return True
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return False

def update_roop_script_colab():
    """Actualizar script principal para usar wrapper Colab"""
    print("\n📝 ACTUALIZANDO SCRIPT PRINCIPAL:")
    print("=" * 40)
    
    script_file = 'run_roop_original_gpu.py'
    
    if not os.path.exists(script_file):
        print(f"❌ Archivo {script_file} no encontrado")
        return False
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Reemplazar comando para usar wrapper Colab
        old_cmd = "sys.executable, 'run_roop_gpu_forced.py'"
        new_cmd = "sys.executable, 'run_roop_colab_gpu.py'"
        
        if old_cmd in content:
            content = content.replace(old_cmd, new_cmd)
        else:
            # Si no encuentra el comando anterior, buscar el wrapper original
            old_cmd2 = "sys.executable, 'run_roop_wrapper.py'"
            if old_cmd2 in content:
                content = content.replace(old_cmd2, new_cmd)
            else:
                print("⚠️ No se encontró comando a reemplazar")
                return False
        
        with open(script_file, 'w') as f:
            f.write(content)
        
        print("✅ Script actualizado para usar Colab GPU")
        return True
        
    except Exception as e:
        print(f"❌ Error actualizando script: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO GPU COMPLETO PARA COLAB")
    print("=" * 60)
    
    # Arreglar conflicto NumPy
    if not fix_numpy_conflict():
        print("❌ Error arreglando NumPy")
        return False
    
    # Instalar TensorFlow para Colab
    if not install_tensorflow_colab():
        print("❌ Error instalando TensorFlow")
        return False
    
    # Configurar entorno GPU
    setup_colab_gpu_environment()
    
    # Crear wrapper Colab
    if not create_colab_gpu_wrapper():
        print("❌ Error creando wrapper")
        return False
    
    # Actualizar script principal
    update_roop_script_colab()
    
    # Probar GPU
    if not test_gpu_with_colab():
        print("❌ Error: GPU no funciona en Colab")
        return False
    
    print("\n✅ GPU COLAB CONFIGURADO EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    print("2. O usar wrapper directamente: python run_roop_colab_gpu.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. Verificar uso GPU: Deberías ver 8-12 GB de VRAM en uso")
    
    return True

if __name__ == '__main__':
    main() 