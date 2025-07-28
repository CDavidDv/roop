#!/usr/bin/env python3
"""
Script que verifica y usa el entorno de Python correcto en Google Colab
"""

import subprocess
import sys
import os

def check_python_environment():
    """Verificar el entorno de Python"""
    print("🔍 Verificando entorno de Python...")
    print(f"🐍 Python executable: {sys.executable}")
    print(f"🐍 Python version: {sys.version}")
    print(f"🐍 Python path: {sys.path[0]}")
    
    # Verificar si estamos en Google Colab
    try:
        import google.colab
        print("✅ Ejecutando en Google Colab")
        return True
    except ImportError:
        print("❌ No estamos en Google Colab")
        return False

def install_dependencies_in_current_env():
    """Instalar dependencias en el entorno actual"""
    print("\n📦 Instalando dependencias en el entorno actual...")
    
    packages = [
        "onnxruntime-gpu",
        "tensorflow-gpu",
        "torch",
        "torchvision",
        "opencv-python",
        "pillow",
        "numpy",
        "scipy",
        "psutil",
        "tqdm",
        "insightface",
        "basicsr",
        "facexlib",
        "gfpgan",
        "realesrgan",
        "albumentations",
        "ffmpeg-python",
        "moviepy",
        "imageio",
        "imageio-ffmpeg"
    ]
    
    for package in packages:
        print(f"Instalando {package}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--force-reinstall", "--quiet"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {package} instalado")
            else:
                print(f"❌ Error instalando {package}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error instalando {package}: {e}")

def test_onnxruntime():
    """Probar onnxruntime específicamente"""
    print("\n🔍 Probando onnxruntime...")
    
    try:
        import onnxruntime
        print(f"✅ onnxruntime disponible: {onnxruntime.__version__}")
        
        # Probar providers disponibles
        providers = onnxruntime.get_available_providers()
        print(f"✅ Providers disponibles: {providers}")
        
        return True
    except ImportError as e:
        print(f"❌ onnxruntime no disponible: {e}")
        return False

def run_roop_with_python():
    """Ejecutar ROOP con el Python correcto"""
    print("\n🎬 Ejecutando ROOP...")
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists("run_batch_gpu_simple.py"):
        print("❌ No se encuentra run_batch_gpu_simple.py")
        print("🔄 Cambiando al directorio roop...")
        if os.path.exists("roop"):
            os.chdir("roop")
        else:
            print("❌ No se encuentra el directorio roop")
            return False
    
    # Comando para ejecutar ROOP
    cmd = [
        sys.executable, "run_batch_gpu_simple.py",
        "--source", "/content/DanielaAS.jpg",
        "--input-folder", "/content/videos",
        "--output-folder", "/content/resultados",
        "--frame-processors", "face_swapper", "face_enhancer",
        "--max-memory", "12",
        "--execution-threads", "8",
        "--temp-frame-quality", "100",
        "--gpu-memory-wait", "30",
        "--keep-fps"
    ]
    
    print(f"🎯 Ejecutando: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ ROOP ejecutado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando ROOP: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 SOLUCIONADOR COMPLETO PARA GOOGLE COLAB")
    print("=" * 60)
    
    # Verificar entorno
    if not check_python_environment():
        print("❌ No estamos en Google Colab")
        return
    
    # Instalar dependencias
    install_dependencies_in_current_env()
    
    # Probar onnxruntime
    if not test_onnxruntime():
        print("❌ onnxruntime no está funcionando")
        print("🔄 Reinicia el runtime y ejecuta nuevamente")
        return
    
    # Ejecutar ROOP
    if run_roop_with_python():
        print("\n🎉 ¡PROCESAMIENTO COMPLETADO!")
        print("=" * 60)
        print("✅ Todas las dependencias funcionando")
        print("✅ ROOP ejecutado correctamente")
        print("📁 Videos guardados en: /content/resultados")
    else:
        print("\n❌ Error en el procesamiento")
        print("🔄 Verifica las rutas y archivos")

if __name__ == "__main__":
    main() 