#!/usr/bin/env python3
"""
Script de instalación actualizado para ROOP con optimizaciones GPU
"""

import os
import sys
import subprocess
import platform
import warnings

# Suprimir warnings
warnings.filterwarnings('ignore')

def check_python_version():
    """Verificar versión de Python"""
    print("🔍 Verificando versión de Python...")
    
    if sys.version_info < (3, 10):
        print(f"❌ Python {sys.version_info.major}.{sys.version_info.minor} no es compatible")
        print("💡 Se requiere Python 3.10 o superior")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} compatible")
    return True

def check_pip():
    """Verificar pip"""
    print("\n🔍 Verificando pip...")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pip disponible: {result.stdout.strip()}")
            return True
        else:
            print("❌ pip no disponible")
            return False
    except Exception as e:
        print(f"❌ Error verificando pip: {e}")
        return False

def upgrade_pip():
    """Actualizar pip"""
    print("\n🔄 Actualizando pip...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        print("✅ pip actualizado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error actualizando pip: {e}")
        return False

def install_dependencies():
    """Instalar dependencias"""
    print("\n📦 Instalando dependencias...")
    
    # Lista de dependencias base
    base_deps = [
        'numpy==1.26.4',
        'typing-extensions==4.11.0',
        'psutil==5.9.8',
        'pillow==10.2.0',
        'tqdm==4.66.1',
        'opencv-python==4.9.0.80',
        'coloredlogs==15.0.1',
        'humanfriendly==10.0',
        'sqlalchemy==2.0.31',
        'addict==2.4.0',
        'pydantic==2.8.0',
        'pydantic-core==2.20.0',
        'pandas-stubs==2.0.3.230814',
        'lmdb==1.5.1',
    ]
    
    # Instalar dependencias base
    for dep in base_deps:
        print(f"   📦 Instalando {dep}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True)
            print(f"   ✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error instalando {dep}: {e}")
            return False
    
    return True

def install_pytorch_gpu():
    """Instalar PyTorch con soporte GPU"""
    print("\n🔥 Instalando PyTorch con CUDA...")
    
    try:
        # Instalar PyTorch con CUDA 12.1
        pytorch_cmd = [
            sys.executable, '-m', 'pip', 'install',
            'torch==2.2.0+cu121',
            'torchaudio==2.2.0+cu121',
            'torchvision==0.17.0+cu121',
            '--extra-index-url', 'https://download.pytorch.org/whl/cu121'
        ]
        
        subprocess.run(pytorch_cmd, check=True, capture_output=True)
        print("✅ PyTorch con CUDA instalado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando PyTorch: {e}")
        print("🔄 Intentando instalar PyTorch CPU...")
        
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio'], 
                          check=True, capture_output=True)
            print("✅ PyTorch CPU instalado")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"❌ Error instalando PyTorch CPU: {e2}")
            return False

def install_tensorflow_gpu():
    """Instalar TensorFlow con soporte GPU"""
    print("\n🔥 Instalando TensorFlow...")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 
                       'tensorflow==2.16.1', 'tensorflow-estimator==2.16.1', 'tensorboard==2.16.1'], 
                      check=True, capture_output=True)
        print("✅ TensorFlow instalado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando TensorFlow: {e}")
        return False

def install_ai_libraries():
    """Instalar librerías de IA"""
    print("\n🤖 Instalando librerías de IA...")
    
    ai_deps = [
        'onnx==1.16.0',
        'onnxruntime-gpu==1.17.0',
        'gfpgan==1.3.8',
        'basicsr==1.4.2',
        'facexlib==0.3.0',
        'insightface==0.7.3',
        'filterpy==1.4.5',
        'opennsfw2==0.10.2',
    ]
    
    for dep in ai_deps:
        print(f"   📦 Instalando {dep}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True)
            print(f"   ✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error instalando {dep}: {e}")
            return False
    
    return True

def install_ui_libraries():
    """Instalar librerías de UI"""
    print("\n🖥️ Instalando librerías de UI...")
    
    ui_deps = [
        'customtkinter==5.2.2',
        'darkdetect==0.8.0',
        'tkinterdnd2==0.3.0',
        'tk==0.1.0',
    ]
    
    for dep in ui_deps:
        print(f"   📦 Instalando {dep}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True)
            print(f"   ✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error instalando {dep}: {e}")
            return False
    
    return True

def install_gpu_tools():
    """Instalar herramientas de GPU"""
    print("\n🔧 Instalando herramientas de GPU...")
    
    gpu_deps = [
        'nvidia-ml-py3==12.535.133',
    ]
    
    for dep in gpu_deps:
        print(f"   📦 Instalando {dep}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                          check=True, capture_output=True)
            print(f"   ✅ {dep} instalado")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Warning: {dep} no se pudo instalar: {e}")
            # No fallar la instalación por esto
    
    return True

def download_models():
    """Descargar modelos necesarios"""
    print("\n📥 Descargando modelos...")
    
    # Crear directorio para modelos
    model_dir = os.path.join(os.getcwd(), 'content', 'roop')
    os.makedirs(model_dir, exist_ok=True)
    
    # URL del modelo
    model_url = 'https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx'
    model_path = os.path.join(model_dir, 'inswapper_128.onnx')
    
    if os.path.exists(model_path):
        print("✅ Modelo ya existe")
        return True
    
    print(f"📥 Descargando modelo desde {model_url}...")
    
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print("✅ Modelo descargado")
        return True
    except Exception as e:
        print(f"❌ Error descargando modelo: {e}")
        print("💡 El modelo se descargará automáticamente en la primera ejecución")
        return False

def run_gpu_test():
    """Ejecutar test de GPU"""
    print("\n🧪 Ejecutando test de GPU...")
    
    try:
        result = subprocess.run([sys.executable, 'gpu_optimization.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error ejecutando test de GPU: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALACIÓN ACTUALIZADA DE ROOP")
    print("=" * 50)
    
    # Verificar requisitos
    if not check_python_version():
        return False
    
    if not check_pip():
        print("❌ pip no disponible. Instale pip primero.")
        return False
    
    # Actualizar pip
    if not upgrade_pip():
        print("⚠️ No se pudo actualizar pip, continuando...")
    
    # Instalar dependencias
    if not install_dependencies():
        print("❌ Error instalando dependencias base")
        return False
    
    # Instalar PyTorch
    if not install_pytorch_gpu():
        print("❌ Error instalando PyTorch")
        return False
    
    # Instalar TensorFlow
    if not install_tensorflow_gpu():
        print("❌ Error instalando TensorFlow")
        return False
    
    # Instalar librerías de IA
    if not install_ai_libraries():
        print("❌ Error instalando librerías de IA")
        return False
    
    # Instalar librerías de UI
    if not install_ui_libraries():
        print("❌ Error instalando librerías de UI")
        return False
    
    # Instalar herramientas de GPU
    install_gpu_tools()
    
    # Descargar modelos
    download_models()
    
    # Test de GPU
    print("\n" + "=" * 50)
    print("🎉 INSTALACIÓN COMPLETADA")
    print("=" * 50)
    print("✅ Todas las dependencias han sido instaladas")
    print("🔧 Ejecutando test de GPU...")
    
    run_gpu_test()
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. Ejecute: python gpu_optimization.py")
    print("2. Ejecute: python run_batch_processing.py --help")
    print("3. ¡Listo para usar ROOP!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ¡Instalación completada exitosamente!")
    else:
        print("\n❌ La instalación falló. Revise los errores arriba.")
        sys.exit(1) 