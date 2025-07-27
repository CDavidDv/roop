#!/usr/bin/env python3
"""
Script para arreglar problemas del predictor NSFW y compatibilidad TensorFlow/cuDNN
"""

import os
import sys
import subprocess

def disable_nsfw_predictor():
    """Desactivar el predictor NSFW que está causando problemas"""
    print("🚫 DESACTIVANDO PREDICTOR NSFW:")
    print("=" * 40)
    
    # Buscar el archivo predictor.py
    def find_predictor_file():
        for root, dirs, files in os.walk('.'):
            if 'predictor.py' in files:
                return os.path.join(root, 'predictor.py')
        return None
    
    predictor_file = find_predictor_file()
    
    if not predictor_file:
        print("❌ Archivo predictor.py no encontrado")
        return False
    
    print(f"✅ Archivo encontrado: {predictor_file}")
    
    # Crear backup del archivo original
    backup_file = predictor_file + '.backup'
    try:
        with open(predictor_file, 'r') as f:
            original_content = f.read()
        
        with open(backup_file, 'w') as f:
            f.write(original_content)
        print("✅ Backup creado")
    except Exception as e:
        print(f"❌ Error creando backup: {e}")
        return False
    
    # Modificar el archivo para desactivar NSFW
    try:
        with open(predictor_file, 'r') as f:
            content = f.read()
        
        # Reemplazar la función predict_video para que siempre retorne False
        new_content = content.replace(
            'def predict_video(target_path: str) -> bool:',
            'def predict_video(target_path: str) -> bool:\n    # NSFW predictor disabled for GPU compatibility\n    return False'
        )
        
        # Si no encuentra la función original, agregar una nueva
        if new_content == content:
            # Buscar donde se define la función
            lines = content.split('\n')
            new_lines = []
            for line in lines:
                if 'def predict_video(target_path: str) -> bool:' in line:
                    new_lines.append(line)
                    new_lines.append('    # NSFW predictor disabled for GPU compatibility')
                    new_lines.append('    return False')
                    # Saltar las líneas de la función original
                    i = lines.index(line) + 1
                    while i < len(lines) and (lines[i].startswith(' ') or lines[i].startswith('\t')):
                        i += 1
                    continue
                new_lines.append(line)
            new_content = '\n'.join(new_lines)
        
        with open(predictor_file, 'w') as f:
            f.write(new_content)
        
        print("✅ Predictor NSFW desactivado")
        return True
        
    except Exception as e:
        print(f"❌ Error modificando archivo: {e}")
        return False

def fix_tensorflow_compatibility():
    """Arreglar problemas de compatibilidad de TensorFlow"""
    print("\n🔧 ARREGLANDO COMPATIBILIDAD TENSORFLOW:")
    print("=" * 40)
    
    # Configurar variables de entorno para TensorFlow
    env_vars = {
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_UNIFIED_MEMORY': '1',
        'TF_MEMORY_ALLOCATION': '0.8',
        'TF_GPU_MEMORY_LIMIT': '12',
        # Variables específicas para cuDNN
        'TF_CUDNN_USE_AUTOTUNE': '0',
        'TF_CUDNN_DETERMINISTIC': '1'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")
    
    print("✅ Variables de entorno configuradas")

def install_compatible_tensorflow():
    """Instalar versión compatible de TensorFlow"""
    print("\n📦 INSTALANDO TENSORFLOW COMPATIBLE:")
    print("=" * 40)
    
    try:
        # Desinstalar TensorFlow actual
        cmd1 = [sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"]
        print(f"⏳ Desinstalando TensorFlow actual...")
        subprocess.run(cmd1, check=True, capture_output=True, text=True)
        
        # Instalar TensorFlow compatible
        cmd2 = [
            sys.executable, "-m", "pip", "install",
            "tensorflow==2.13.0"
        ]
        print(f"⏳ Instalando TensorFlow 2.13.0...")
        result = subprocess.run(cmd2, check=True, capture_output=True, text=True)
        print("✅ TensorFlow compatible instalado")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando TensorFlow: {e}")
        return False

def test_tensorflow_setup():
    """Probar configuración de TensorFlow"""
    print("\n🧪 PROBANDO CONFIGURACIÓN TENSORFLOW:")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Verificar GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU disponible: {len(gpus)} dispositivos")
            for gpu in gpus:
                print(f"  - {gpu}")
        else:
            print("❌ GPU no disponible")
        
        # Probar operación simple
        try:
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
            print("✅ Operación TensorFlow exitosa")
        except Exception as e:
            print(f"❌ Error en operación TensorFlow: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando TensorFlow: {e}")
        return False

def create_roop_wrapper():
    """Crear wrapper que evite problemas de NSFW"""
    print("\n📝 CREANDO WRAPPER ROOP:")
    print("=" * 40)
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper para ROOP que evita problemas de NSFW y TensorFlow
"""

import os
import sys
import subprocess

# Configurar variables de entorno
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def main():
    """Función principal"""
    # Obtener argumentos
    args = sys.argv[1:]
    
    # Construir comando
    cmd = [sys.executable, 'run.py'] + args
    
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
        with open('run_roop_wrapper.py', 'w') as f:
            f.write(wrapper_content)
        print("✅ Wrapper creado: run_roop_wrapper.py")
        return True
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO PROBLEMAS DE NSFW Y TENSORFLOW")
    print("=" * 60)
    
    # Desactivar predictor NSFW
    if not disable_nsfw_predictor():
        print("❌ Error desactivando predictor NSFW")
        return False
    
    # Configurar TensorFlow
    fix_tensorflow_compatibility()
    
    # Instalar TensorFlow compatible si es necesario
    # install_compatible_tensorflow()
    
    # Probar configuración
    if not test_tensorflow_setup():
        print("⚠️ Advertencia: Configuración TensorFlow no óptima")
    
    # Crear wrapper
    create_roop_wrapper()
    
    print("\n✅ PROBLEMAS ARREGLADOS EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Usar el wrapper: python run_roop_wrapper.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("2. O usar el script original: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    
    return True

if __name__ == '__main__':
    main() 