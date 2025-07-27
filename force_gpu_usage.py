#!/usr/bin/env python3
"""
Script para forzar el uso de GPU en ROOP
"""

import os
import sys
import subprocess

def force_gpu_configuration():
    """Forzar configuración de GPU"""
    print("🚀 FORZANDO CONFIGURACIÓN GPU:")
    print("=" * 40)
    
    # Variables de entorno críticas para GPU
    gpu_env_vars = {
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_CPP_MIN_LOG_LEVEL': '0',  # Mostrar todos los logs
        'TF_FORCE_UNIFIED_MEMORY': '1',
        'TF_MEMORY_ALLOCATION': '0.9',  # Usar 90% de VRAM
        'TF_GPU_MEMORY_LIMIT': '14',  # Límite alto para T4
        'TF_CUDNN_USE_AUTOTUNE': '0',
        'TF_CUDNN_DETERMINISTIC': '1',
        # Forzar uso de GPU
        'TF_ENABLE_ONEDNN_OPTS': '0',
        'TF_ENABLE_MKL_NATIVE_FORMAT': '0',
        'TF_ENABLE_AUTO_MIXED_PRECISION': '0'
    }
    
    for var, value in gpu_env_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")
    
    print("✅ Variables de entorno GPU configuradas")

def test_gpu_availability():
    """Probar disponibilidad de GPU"""
    print("\n🧪 PROBANDO GPU:")
    print("=" * 40)
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
        
        # Verificar dispositivos disponibles
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
            
            # Configurar GPU para crecimiento de memoria
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

def create_gpu_forced_wrapper():
    """Crear wrapper que fuerce uso de GPU"""
    print("\n📝 CREANDO WRAPPER CON GPU FORZADO:")
    print("=" * 40)
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper para ROOP que fuerza el uso de GPU
"""

import os
import sys
import subprocess

# FORZAR CONFIGURACIÓN GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['TF_MEMORY_ALLOCATION'] = '0.9'
os.environ['TF_GPU_MEMORY_LIMIT'] = '14'
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_MKL_NATIVE_FORMAT'] = '0'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

# Verificar GPU antes de ejecutar
def check_gpu():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"🎮 GPU detectada: {len(gpus)} dispositivos")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU configurado correctamente")
            return True
        else:
            print("❌ GPU no disponible")
            return False
    except Exception as e:
        print(f"❌ Error verificando GPU: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ROOP CON GPU FORZADO")
    print("=" * 40)
    
    # Verificar GPU
    if not check_gpu():
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
        with open('run_roop_gpu_forced.py', 'w') as f:
            f.write(wrapper_content)
        print("✅ Wrapper GPU forzado creado: run_roop_gpu_forced.py")
        return True
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return False

def update_roop_script():
    """Actualizar script principal para usar GPU forzado"""
    print("\n📝 ACTUALIZANDO SCRIPT PRINCIPAL:")
    print("=" * 40)
    
    # Buscar el archivo run_roop_original_gpu.py
    script_file = 'run_roop_original_gpu.py'
    
    if not os.path.exists(script_file):
        print(f"❌ Archivo {script_file} no encontrado")
        return False
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Reemplazar el comando para usar el wrapper GPU forzado
        old_cmd = "sys.executable, 'run_roop_wrapper.py'"
        new_cmd = "sys.executable, 'run_roop_gpu_forced.py'"
        
        if old_cmd in content:
            content = content.replace(old_cmd, new_cmd)
            
            with open(script_file, 'w') as f:
                f.write(content)
            
            print("✅ Script actualizado para usar GPU forzado")
            return True
        else:
            print("⚠️ No se encontró el comando a reemplazar")
            return False
            
    except Exception as e:
        print(f"❌ Error actualizando script: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 FORZANDO USO DE GPU EN ROOP")
    print("=" * 60)
    
    # Configurar GPU
    force_gpu_configuration()
    
    # Probar GPU
    if not test_gpu_availability():
        print("❌ Error: GPU no disponible")
        return False
    
    # Crear wrapper GPU forzado
    if not create_gpu_forced_wrapper():
        print("❌ Error creando wrapper GPU")
        return False
    
    # Actualizar script principal
    update_roop_script()
    
    print("\n✅ GPU FORZADO CONFIGURADO EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Usar el script actualizado: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    print("2. O usar el wrapper directamente: python run_roop_gpu_forced.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("3. Verificar uso GPU: Deberías ver 8-12 GB de VRAM en uso")
    
    return True

if __name__ == '__main__':
    main() 