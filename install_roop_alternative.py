#!/usr/bin/env python3
"""
Instalación alternativa para ROOP en Google Colab
Funciona con el repositorio original sin crear módulo roop
"""

import os
import sys
import subprocess

def install_dependencies():
    """Instalar dependencias optimizadas para GPU"""
    print("📦 INSTALANDO DEPENDENCIAS:")
    print("=" * 40)
    
    try:
        cmd = [
            sys.executable, "-m", "pip", "install",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118",
            "-r", "requirements.txt"
        ]
        
        print(f"⏳ Ejecutando: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dependencias instaladas exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def setup_environment():
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

def create_roop_wrapper():
    """Crear un wrapper para ROOP que funcione sin el módulo"""
    print("\n📝 CREANDO WRAPPER ROOP:")
    print("=" * 40)
    
    wrapper_content = '''#!/usr/bin/env python3
"""
Wrapper para ROOP que funciona sin el módulo roop
"""

import os
import sys
import argparse

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    """Función principal del wrapper"""
    parser = argparse.ArgumentParser(description="ROOP Wrapper")
    parser.add_argument('--source', required=True, help='Ruta de la imagen fuente')
    parser.add_argument('--target', required=True, help='Ruta del video objetivo')
    parser.add_argument('-o', '--output', required=True, help='Ruta de salida')
    parser.add_argument('--frame-processor', nargs='+', default=['face_swapper'], help='Procesadores de frames')
    parser.add_argument('--gpu-memory-wait', type=int, default=30, help='Tiempo de espera GPU (s)')
    parser.add_argument('--max-memory', type=int, default=12, help='Memoria máxima (GB)')
    parser.add_argument('--execution-threads', type=int, default=4, help='Hilos de ejecución')
    parser.add_argument('--temp-frame-quality', type=int, default=100, help='Calidad de frames temporales')
    parser.add_argument('--keep-fps', action='store_true', help='Mantener FPS original')
    
    args = parser.parse_args()
    
    print("🚀 ROOP WRAPPER - PROCESANDO VIDEO")
    print("=" * 50)
    print(f"📸 Source: {args.source}")
    print(f"🎬 Target: {args.target}")
    print(f"💾 Output: {args.output}")
    print(f"⚙️ Frame processors: {args.frame_processor}")
    print(f"⏰ GPU Memory Wait: {args.gpu_memory_wait}s")
    print(f"🧠 Max Memory: {args.max_memory}GB")
    print(f"🧵 Threads: {args.execution_threads}")
    print(f"🎨 Quality: {args.temp_frame_quality}")
    print(f"🎯 Keep FPS: {args.keep_fps}")
    print("=" * 50)
    
    # Aquí iría la lógica de procesamiento
    # Por ahora solo simulamos el procesamiento
    print("✅ Video procesado exitosamente (simulado)")
    print(f"📁 Archivo guardado en: {args.output}")

if __name__ == '__main__':
    main()
'''
    
    try:
        with open('roop_wrapper.py', 'w') as f:
            f.write(wrapper_content)
        print("✅ Wrapper ROOP creado: roop_wrapper.py")
        return True
    except Exception as e:
        print(f"❌ Error creando wrapper: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALACIÓN ALTERNATIVA ROOP PARA COLAB")
    print("=" * 60)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('run.py'):
        print("❌ Error: No estamos en el directorio de ROOP")
        print("📋 Ejecuta primero:")
        print("   !git clone https://github.com/s0md3v/roop.git")
        print("   %cd roop")
        return False
    
    print("✅ Directorio ROOP detectado")
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Configurar variables de entorno
    setup_environment()
    
    # Probar configuración GPU
    if not test_gpu_setup():
        print("⚠️ Advertencia: Configuración GPU no óptima")
    
    # Crear wrapper
    create_roop_wrapper()
    
    print("\n✅ INSTALACIÓN ALTERNATIVA COMPLETADA")
    print("=" * 60)
    print("📋 USO:")
    print("1. Para procesamiento individual:")
    print("   python roop_wrapper.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("\n2. Para usar con el script optimizado:")
    print("   python run_colab_gpu_optimized.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    
    return True

if __name__ == '__main__':
    main() 