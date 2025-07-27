#!/usr/bin/env python3
"""
Instalar todas las dependencias ROOP para Colab
"""

import os
import sys
import subprocess

def install_all_dependencies():
    """Instalar todas las dependencias"""
    print("📦 INSTALANDO TODAS LAS DEPENDENCIAS:")
    print("=" * 40)
    
    try:
        # Todas las dependencias necesarias
        dependencies = [
            "customtkinter",
            "tkinterdnd2",
            "pillow",
            "opencv-python",
            "numpy==1.24.3",
            "scipy",
            "scikit-image",
            "insightface",
            "opennsfw2",
            "onnxruntime-gpu==1.15.1",
            "tensorflow==2.12.0",
            "torch==2.0.1",
            "torchvision==0.15.2",
            "nvidia-ml-py3",
            "pynvml"
        ]
        
        for dep in dependencies:
            print(f"⏳ Instalando {dep}...")
            cmd = [sys.executable, "-m", "pip", "install", dep]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Todas las dependencias instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_colab_environment():
    """Configurar entorno para Colab"""
    print("\n🔧 CONFIGURANDO ENTORNO COLAB:")
    print("=" * 40)
    
    # Variables de entorno para Colab
    env_vars = {
        'DISPLAY': ':0',
        'PYTHONPATH': '/content/roop',
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true'
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"✅ {var} = {value}")
    
    print("✅ Variables de entorno configuradas")

def create_headless_roop():
    """Crear versión headless de ROOP"""
    print("\n📝 CREANDO ROOP HEADLESS:")
    print("=" * 40)
    
    # Crear script headless
    headless_content = '''#!/usr/bin/env python3
"""
ROOP Headless para Colab
"""

import os
import sys
import argparse

# Configurar entorno
os.environ['DISPLAY'] = ':0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    parser = argparse.ArgumentParser(description='ROOP Headless')
    parser.add_argument('--source', required=True, help='Imagen fuente')
    parser.add_argument('--target', required=True, help='Video objetivo')
    parser.add_argument('-o', '--output', required=True, help='Archivo de salida')
    parser.add_argument('--frame-processor', nargs='+', default=['face_swapper', 'face_enhancer'])
    parser.add_argument('--max-memory', type=int, default=12)
    parser.add_argument('--execution-threads', type=int, default=30)
    parser.add_argument('--temp-frame-quality', type=int, default=100)
    parser.add_argument('--keep-fps', action='store_true')
    
    args = parser.parse_args()
    
    # Importar ROOP después de configurar argumentos
    sys.path.insert(0, '.')
    from roop import core
    
    # Configurar argumentos
    core.source_path = args.source
    core.target_path = args.target
    core.output_path = args.output
    core.frame_processors = args.frame_processor
    core.max_memory = args.max_memory
    core.execution_threads = args.execution_threads
    core.temp_frame_quality = args.temp_frame_quality
    core.keep_fps = args.keep_fps
    
    # Ejecutar ROOP
    core.run()

if __name__ == '__main__':
    main()
'''
    
    try:
        with open('run_roop_headless.py', 'w') as f:
            f.write(headless_content)
        print("✅ ROOP headless creado: run_roop_headless.py")
        return True
    except Exception as e:
        print(f"❌ Error creando ROOP headless: {e}")
        return False

def test_roop_headless():
    """Probar ROOP headless"""
    print("\n🧪 PROBANDO ROOP HEADLESS:")
    print("=" * 40)
    
    try:
        # Probar importación
        import sys
        sys.path.insert(0, '.')
        
        from roop import core
        print("✅ ROOP importado exitosamente")
        
        # Probar configuración básica
        core.source_path = "/content/DanielaAS.jpg"
        core.target_path = "/content/videos_entrada/135.mp4"
        core.output_path = "/content/videos_salida/test.mp4"
        core.frame_processors = ['face_swapper', 'face_enhancer']
        
        print("✅ ROOP configurado correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error probando ROOP: {e}")
        return False

def update_roop_script_headless():
    """Actualizar script principal para usar ROOP headless"""
    print("\n📝 ACTUALIZANDO SCRIPT PRINCIPAL:")
    print("=" * 40)
    
    script_file = 'run_roop_original_gpu.py'
    
    if not os.path.exists(script_file):
        print(f"❌ Archivo {script_file} no encontrado")
        return False
    
    try:
        with open(script_file, 'r') as f:
            content = f.read()
        
        # Reemplazar comando para usar ROOP headless
        old_cmd = "sys.executable, 'run_roop_gpu_complete.py'"
        new_cmd = "sys.executable, 'run_roop_headless.py'"
        
        if old_cmd in content:
            content = content.replace(old_cmd, new_cmd)
        else:
            # Buscar otros comandos
            old_cmds = [
                "sys.executable, 'run_roop_simple_gpu.py'",
                "sys.executable, 'run_roop_pytorch_gpu.py'",
                "sys.executable, 'run_roop_colab_gpu_final.py'",
                "sys.executable, 'run_roop_colab_gpu.py'",
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
        
        print("✅ Script actualizado para usar ROOP headless")
        return True
        
    except Exception as e:
        print(f"❌ Error actualizando script: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALANDO TODAS LAS DEPENDENCIAS ROOP")
    print("=" * 60)
    
    # Instalar todas las dependencias
    if not install_all_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Configurar entorno Colab
    setup_colab_environment()
    
    # Crear ROOP headless
    if not create_headless_roop():
        print("❌ Error creando ROOP headless")
        return False
    
    # Actualizar script principal
    update_roop_script_headless()
    
    # Probar ROOP headless
    if not test_roop_headless():
        print("❌ Error: ROOP headless no funciona")
        return False
    
    print("\n✅ TODAS LAS DEPENDENCIAS INSTALADAS EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    print("2. O usar ROOP headless directamente: python run_roop_headless.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    
    return True

if __name__ == '__main__':
    main() 