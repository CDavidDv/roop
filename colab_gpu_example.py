#!/usr/bin/env python3
"""
Ejemplo de uso para Google Colab con GPU optimizado
"""

# Configuración para Google Colab
import os
import subprocess
import sys

# Configurar variables de entorno para GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

def setup_colab():
    """Configurar Google Colab para ROOP"""
    print("🚀 Configurando Google Colab para ROOP...")
    
    # Clonar ROOP
    subprocess.run([
        "git", "clone", "--branch", "v3", "https://github.com/CDavidDv/roop.git"
    ], check=True)
    
    # Cambiar al directorio de ROOP
    os.chdir("roop")
    
    # Descargar el modelo de face swap
    subprocess.run([
        "wget", "https://civitai.com/api/download/models/85159", 
        "-O", "inswapper_128.onnx"
    ], check=True)
    
    print("✅ Configuración completada")

def process_videos_with_gpu():
    """Procesar videos usando GPU"""
    print("🎬 Iniciando procesamiento con GPU...")
    
    # Ejecutar el script optimizado para GPU
    cmd = [
        "python", "run_batch_gpu.py",
        "--source", "/content/DanielaAS.jpg",  # Cambiar por tu imagen fuente
        "--input-folder", "/content/videos",   # Carpeta con videos a procesar
        "--output-folder", "/content/resultados",  # Carpeta para resultados
        "--frame-processors", "face_swapper", "face_enhancer",
        "--max-memory", "12",
        "--execution-threads", "8",
        "--temp-frame-quality", "100",
        "--gpu-memory-wait", "30",
        "--keep-fps"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Procesamiento completado exitosamente!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ Error en el procesamiento:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")

if __name__ == "__main__":
    # Configurar Colab
    setup_colab()
    
    # Procesar videos
    process_videos_with_gpu() 