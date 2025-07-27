#!/usr/bin/env python3
"""
Instalación para ROOP ultimopunto optimizado para Google Colab T4
Específico para el repositorio del usuario con optimizaciones GPU
"""

import os
import sys
import subprocess
import requests

def setup_gpu_environment():
    """Configurar variables de entorno para GPU"""
    print("⚙️ CONFIGURANDO VARIABLES DE ENTORNO:")
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

def install_dependencies():
    """Instalar dependencias optimizadas para GPU"""
    print("\n📦 INSTALANDO DEPENDENCIAS:")
    print("=" * 40)
    
    try:
        # Instalar dependencias con índice extra de PyTorch
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

def download_model():
    """Descargar el modelo de face swap si no existe"""
    print("\n📥 VERIFICANDO MODELO:")
    print("=" * 40)
    
    model_file = "inswapper_128.onnx"
    
    if os.path.exists(model_file):
        print(f"✅ Modelo ya existe: {model_file}")
        return True
    
    model_url = "https://civitai.com/api/download/models/85159"
    
    try:
        print(f"⏳ Descargando modelo desde: {model_url}")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✅ Modelo descargado: {model_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
        print("⚠️ Puedes descargarlo manualmente más tarde")
        return False

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

def optimize_gpu():
    """Optimizar GPU para Colab T4"""
    print("\n🚀 OPTIMIZANDO GPU PARA COLAB T4:")
    print("=" * 40)
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detectada: {torch.cuda.get_device_name()}")
            print(f"📊 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            
            # Configurar memoria GPU
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            print("✅ Configuración GPU completada")
        else:
            print("❌ GPU no disponible")
            return False
    except Exception as e:
        print(f"❌ Error configurando GPU: {e}")
        return False
    
    return True

def create_batch_script():
    """Crear script optimizado para procesamiento en lotes"""
    print("\n📝 CREANDO SCRIPT DE PROCESAMIENTO EN LOTE:")
    print("=" * 40)
    
    script_content = '''#!/usr/bin/env python3
"""
Script optimizado para procesamiento en lotes con ROOP ultimopunto
Optimizado para Google Colab T4
"""

import os
import sys
import subprocess
import time
import gc
from pathlib import Path

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_gpu():
    """Configurar GPU"""
    print("🚀 CONFIGURANDO GPU PARA COLAB T4")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detectada: {torch.cuda.get_device_name()}")
            torch.cuda.empty_cache()
            return True
        else:
            print("❌ GPU no disponible")
            return False
    except Exception as e:
        print(f"❌ Error configurando GPU: {e}")
        return False

def clear_memory():
    """Limpiar memoria"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    gc.collect()

def process_video(source: str, target: str, output: str, max_memory: int = 12, 
                 execution_threads: int = 30, temp_frame_quality: int = 100, keep_fps: bool = True):
    """Procesar un video"""
    print(f"🎬 Procesando: {os.path.basename(target)}")
    
    cmd = [
        sys.executable, 'run.py',
        '--source', source,
        '--target', target,
        '-o', output,
        '--frame-processor', 'face_swapper', 'face_enhancer',
        '--max-memory', str(max_memory),
        '--execution-threads', str(execution_threads),
        '--temp-frame-quality', str(temp_frame_quality)
    ]
    
    if keep_fps:
        cmd.append('--keep-fps')
    
    try:
        clear_memory()
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        clear_memory()
        print(f"✅ Completado: {os.path.basename(output)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {os.path.basename(target)}")
        print(f"STDERR: {e.stderr}")
        return False

def process_folder_batch(source: str, input_folder: str, output_dir: str, max_memory: int = 12,
                        execution_threads: int = 30, temp_frame_quality: int = 100, keep_fps: bool = True):
    """Procesar todos los videos en una carpeta"""
    print("🚀 PROCESAMIENTO DE CARPETA - ROOP ULTIMOPUNTO")
    print("=" * 60)
    print(f"📸 Source: {source}")
    print(f"📁 Carpeta de entrada: {input_folder}")
    print(f"📁 Carpeta de salida: {output_dir}")
    print("=" * 60)
    
    # Verificar que la carpeta existe
    if not os.path.exists(input_folder):
        print(f"❌ Carpeta de entrada no encontrada: {input_folder}")
        return
    
    # Buscar videos en la carpeta
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    videos = []
    
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_path = os.path.join(input_folder, file)
            videos.append(video_path)
    
    if not videos:
        print(f"❌ No se encontraron videos en: {input_folder}")
        return
    
    print(f"🎬 Videos encontrados: {len(videos)}")
    for video in videos:
        print(f"  - {os.path.basename(video)}")
    
    # Crear directorio de salida
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Directorio creado: {output_dir}")
    
    successful = 0
    failed = 0
    
    for i, video in enumerate(videos, 1):
        print(f"\\n📹 Progreso: {i}/{len(videos)}")
        
        # Generar nombre de salida
        video_name = Path(video).name
        source_name = Path(source).stem
        output_name = f"{source_name}_{video_name}"
        output_path = os.path.join(output_dir, output_name)
        
        # Esperar entre videos
        if i > 1:
            print("⏳ Esperando liberación de memoria...")
            time.sleep(3)
        
        if process_video(source, video, output_path, max_memory, execution_threads, temp_frame_quality, keep_fps):
            successful += 1
        else:
            failed += 1
    
    print(f"\\n📊 RESUMEN:")
    print(f"✅ Exitosos: {successful}")
    print(f"❌ Fallidos: {failed}")
    if successful + failed > 0:
        print(f"📈 Tasa de éxito: {successful/(successful+failed)*100:.1f}%")

if __name__ == '__main__':
    # Configurar GPU
    if not setup_gpu():
        print("❌ No se pudo configurar GPU")
        exit(1)
    
    # Ejemplo de uso - modificar según tus archivos
    source = "/content/DanielaAS.jpg"  # Cambiar por tu imagen
    input_folder = "videos_entrada"    # Carpeta con videos
    output_dir = "videos_salida"       # Carpeta de salida
    
    process_folder_batch(source, input_folder, output_dir, 
                        max_memory=12, execution_threads=30, keep_fps=True)
'''
    
    try:
        with open('run_batch_ultimopunto.py', 'w') as f:
            f.write(script_content)
        print("✅ Script creado: run_batch_ultimopunto.py")
        return True
    except Exception as e:
        print(f"❌ Error creando script: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALACIÓN ROOP ULTIMOPUNTO PARA COLAB")
    print("=" * 60)
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('run.py'):
        print("❌ Error: No estamos en el directorio de ROOP")
        print("📋 Asegúrate de ejecutar:")
        print("   !git clone --branch ultimopunto https://github.com/CDavidDv/roop.git")
        print("   %cd roop")
        return False
    
    print("✅ Directorio ROOP ultimopunto detectado")
    
    # Configurar variables de entorno
    setup_gpu_environment()
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Descargar modelo si es necesario
    download_model()
    
    # Optimizar GPU
    if not optimize_gpu():
        print("❌ Error optimizando GPU")
        return False
    
    # Probar configuración GPU
    if not test_gpu_setup():
        print("⚠️ Advertencia: Configuración GPU no óptima")
    
    # Crear script de procesamiento
    create_batch_script()
    
    print("\n✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("📋 USO:")
    print("1. Crear carpetas:")
    print("   !mkdir -p videos_entrada videos_salida")
    print("\n2. Subir videos a videos_entrada")
    print("\n3. Procesar videos:")
    print("   !python run_batch_ultimopunto.py")
    print("\n4. O usar el script original:")
    print("   !python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    
    return True

if __name__ == '__main__':
    main() 