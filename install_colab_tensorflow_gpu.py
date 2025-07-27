#!/usr/bin/env python3
"""
Instalación oficial de TensorFlow GPU para Google Colab
"""

import os
import sys
import subprocess

def install_colab_tensorflow():
    """Instalar TensorFlow GPU oficial para Colab"""
    print("🚀 INSTALANDO TENSORFLOW GPU OFICIAL PARA COLAB:")
    print("=" * 60)
    
    try:
        # Desinstalar TensorFlow actual
        print("⏳ Desinstalando TensorFlow actual...")
        cmd1 = [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow", "tensorflow-gpu"]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar TensorFlow GPU oficial para Colab
        print("⏳ Instalando TensorFlow GPU para Colab...")
        cmd2 = [
            sys.executable, "-m", "pip", "install",
            "tensorflow[gpu]==2.13.0"
        ]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)
        
        print("✅ TensorFlow GPU instalado")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando TensorFlow: {e}")
        return False

def install_colab_gpu_dependencies():
    """Instalar dependencias GPU específicas de Colab"""
    print("\n📦 INSTALANDO DEPENDENCIAS GPU COLAB:")
    print("=" * 40)
    
    try:
        # Instalar dependencias GPU
        dependencies = [
            "nvidia-ml-py3",
            "pynvml",
            "onnxruntime-gpu",
            "torch",
            "torchvision"
        ]
        
        for dep in dependencies:
            print(f"⏳ Instalando {dep}...")
            cmd = [sys.executable, "-m", "pip", "install", dep]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Dependencias GPU instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_colab_gpu_env():
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
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
        'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")
    
    print("✅ Variables de entorno configuradas")

def test_colab_gpu():
    """Probar GPU en Colab"""
    print("\n🧪 PROBANDO GPU COLAB:")
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

def create_colab_gpu_wrapper_final():
    """Crear wrapper final para Colab GPU"""
    print("\n📝 CREANDO WRAPPER FINAL COLAB GPU:")
    print("=" * 40)
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper final para ROOP con GPU en Google Colab
"""

import os
import sys
import subprocess

# CONFIGURACIÓN FINAL PARA COLAB GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['TF_MEMORY_ALLOCATION'] = '0.9'
os.environ['TF_GPU_MEMORY_LIMIT'] = '14'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

def check_colab_gpu_final():
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
    print("🚀 ROOP CON GPU COLAB FINAL")
    print("=" * 40)
    
    # Verificar GPU
    if not check_colab_gpu_final():
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
        with open('run_roop_colab_gpu_final.py', 'w') as f:
            f.write(wrapper_content)
        print("✅ Wrapper final creado: run_roop_colab_gpu_final.py")
        return True
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return False

def update_roop_script_final():
    """Actualizar script principal para usar wrapper final"""
    print("\n📝 ACTUALIZANDO SCRIPT PRINCIPAL FINAL:")
    print("=" * 40)
    
    script_file = 'run_roop_original_gpu.py'
    
    if not os.path.exists(script_file):
        print(f"❌ Archivo {script_file} no encontrado")
        return False
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Reemplazar comando para usar wrapper final
        old_cmd = "sys.executable, 'run_roop_colab_gpu.py'"
        new_cmd = "sys.executable, 'run_roop_colab_gpu_final.py'"
        
        if old_cmd in content:
            content = content.replace(old_cmd, new_cmd)
        else:
            # Buscar otros comandos
            old_cmds = [
                "sys.executable, 'run_roop_gpu_forced.py'",
                "sys.executable, 'run_roop_wrapper.py'"
            ]
            for old_cmd in old_cmds:
                if old_cmd in content:
                    content = content.replace(old_cmd, new_cmd)
                    break
            else:
                print("⚠️ No se encontró comando a reemplazar")
                return False
        
        with open(script_file, 'w') as f:
            f.write(content)
        
        print("✅ Script actualizado para usar wrapper final")
        return True
        
    except Exception as e:
        print(f"❌ Error actualizando script: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALACIÓN TENSORFLOW GPU OFICIAL COLAB")
    print("=" * 60)
    
    # Instalar TensorFlow GPU oficial
    if not install_colab_tensorflow():
        print("❌ Error instalando TensorFlow GPU")
        return False
    
    # Instalar dependencias GPU
    if not install_colab_gpu_dependencies():
        print("❌ Error instalando dependencias GPU")
        return False
    
    # Configurar entorno GPU
    setup_colab_gpu_env()
    
    # Crear wrapper final
    if not create_colab_gpu_wrapper_final():
        print("❌ Error creando wrapper final")
        return False
    
    # Actualizar script principal
    update_roop_script_final()
    
    # Probar GPU
    if not test_colab_gpu():
        print("❌ Error: GPU no funciona en Colab")
        return False
    
    print("\n✅ TENSORFLOW GPU COLAB INSTALADO EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    print("2. O usar wrapper directamente: python run_roop_colab_gpu_final.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. Verificar uso GPU: Deberías ver 8-12 GB de VRAM en uso")
    
    return True

if __name__ == '__main__':
    main() 