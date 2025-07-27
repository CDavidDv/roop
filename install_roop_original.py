#!/usr/bin/env python3
"""
Instalación para ROOP original optimizado para Google Colab T4
Basado en tu código original que funcionaba
"""

import os
import sys
import subprocess
import requests

def clone_roop_repository():
    """Clonar el repositorio de ROOP"""
    print("📦 CLONANDO REPOSITORIO ROOP:")
    print("=" * 40)
    
    try:
        # Clonar el repositorio
        cmd = ["git", "clone", "https://github.com/CDavidDv/roop"]
        print(f"⏳ Ejecutando: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Repositorio clonado exitosamente")
        
        # Cambiar al directorio
        os.chdir("roop")
        print("✅ Cambiado al directorio roop")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error clonando repositorio:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def install_dependencies():
    """Instalar dependencias"""
    print("\n📦 INSTALANDO DEPENDENCIAS:")
    print("=" * 40)
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
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
    """Descargar el modelo de face swap"""
    print("\n📥 DESCARGANDO MODELO:")
    print("=" * 40)
    
    model_url = "https://civitai.com/api/download/models/85159"
    model_file = "inswapper_128.onnx"
    
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
        return False

def setup_gpu_environment():
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

def create_optimized_script():
    """Crear script optimizado para procesamiento en lotes"""
    print("\n📝 CREANDO SCRIPT OPTIMIZADO:")
    print("=" * 40)
    
    script_content = '''#!/usr/bin/env python3
"""
Script optimizado para procesamiento en lotes con ROOP
Basado en tu código original pero optimizado para GPU
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

def process_batch(source: str, videos: list, output_dir: str, max_memory: int = 12,
                 execution_threads: int = 30, temp_frame_quality: int = 100, keep_fps: bool = True):
    """Procesar lote de videos"""
    print("🚀 INICIANDO PROCESAMIENTO EN LOTE")
    print("=" * 50)
    print(f"📸 Source: {source}")
    print(f"🎬 Videos: {len(videos)}")
    print(f"📁 Output: {output_dir}")
    print("=" * 50)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    successful = 0
    failed = 0
    
    for i, video in enumerate(videos, 1):
        print(f"\\n📹 Progreso: {i}/{len(videos)}")
        
        if not os.path.exists(video):
            print(f"❌ Video no encontrado: {video}")
            failed += 1
            continue
        
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
    
    # Ejemplo de uso
    source = "/content/LilitAS.png"
    videos = [
        "/content/62.mp4", "/content/71.mp4", "/content/72.mp4",
        "/content/74.mp4", "/content/75.mp4", "/content/76.mp4",
        "/content/77.mp4", "/content/78.mp4", "/content/79.mp4"
    ]
    output_dir = "/content/resultados"
    
    process_batch(source, videos, output_dir, max_memory=12, execution_threads=30, keep_fps=True)
'''
    
    try:
        with open('run_optimized_batch.py', 'w') as f:
            f.write(script_content)
        print("✅ Script optimizado creado: run_optimized_batch.py")
        return True
    except Exception as e:
        print(f"❌ Error creando script: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALACIÓN ROOP ORIGINAL PARA COLAB")
    print("=" * 60)
    
    # Clonar repositorio
    if not clone_roop_repository():
        print("❌ Error clonando repositorio")
        return False
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Descargar modelo
    if not download_model():
        print("❌ Error descargando modelo")
        return False
    
    # Configurar GPU
    setup_gpu_environment()
    
    # Probar GPU
    if not test_gpu_setup():
        print("⚠️ Advertencia: Configuración GPU no óptima")
    
    # Crear script optimizado
    create_optimized_script()
    
    print("\n✅ INSTALACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("📋 USO:")
    print("1. Para procesamiento en lotes:")
    print("   python run_optimized_batch.py")
    print("\n2. Para procesamiento individual:")
    print("   python run.py --source imagen.jpg --target video.mp4 -o resultado.mp4")
    print("\n3. Para procesar carpeta:")
    print("   python run_roop_original_gpu.py --source imagen.jpg --input-folder videos --output-dir resultados")
    
    return True

if __name__ == '__main__':
    main() 