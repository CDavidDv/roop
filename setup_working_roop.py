#!/usr/bin/env python3
"""
Script para configurar el repositorio de ROOP que funciona
Usa el repositorio CDavidDv/roop que funcionaba originalmente
"""

import os
import sys
import subprocess
import shutil

def setup_working_roop():
    """Configurar el repositorio de ROOP que funciona"""
    print("🚀 CONFIGURANDO ROOP FUNCIONAL")
    print("=" * 60)
    
    # Verificar si ya estamos en el directorio correcto
    if os.path.exists('run_batch_processing.py'):
        print("✅ Ya estamos en el directorio de ROOP funcional")
        return True
    
    # Clonar el repositorio que funciona
    print("📥 CLONANDO REPOSITORIO FUNCIONAL:")
    print("=" * 40)
    
    try:
        # Remover directorio existente si existe
        if os.path.exists('roop'):
            shutil.rmtree('roop')
            print("🗑️ Directorio roop existente removido")
        
        # Clonar el repositorio que funciona
        cmd = ["git", "clone", "https://github.com/CDavidDv/roop.git"]
        print(f"⏳ Ejecutando: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Repositorio clonado exitosamente")
        
        # Cambiar al directorio
        os.chdir('roop')
        print("📁 Cambiado al directorio roop")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error clonando repositorio: {e}")
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
    """Descargar modelo de face swap"""
    print("\n📥 DESCARGANDO MODELO:")
    print("=" * 40)
    
    model_path = "inswapper_128.onnx"
    if not os.path.exists(model_path):
        try:
            cmd = [
                "wget", "https://civitai.com/api/download/models/85159", 
                "-O", model_path
            ]
            print(f"⏳ Ejecutando: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ Modelo descargado exitosamente")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error descargando modelo: {e}")
            return False
    else:
        print("✅ Modelo ya existe")
        return True

def setup_gpu():
    """Configurar GPU"""
    print("\n⚡ CONFIGURANDO GPU:")
    print("=" * 40)
    
    # Configurar variables de entorno
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
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU detectada: {torch.cuda.get_device_name()}")
            print(f"📊 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            torch.cuda.empty_cache()
            return True
        else:
            print("❌ GPU no disponible")
            return False
    except Exception as e:
        print(f"❌ Error configurando GPU: {e}")
        return False

def test_setup():
    """Probar la configuración"""
    print("\n🧪 PROBANDO CONFIGURACIÓN:")
    print("=" * 40)
    
    # Verificar archivos clave
    key_files = ['run_batch_processing.py', 'run.py', 'requirements.txt']
    for file in key_files:
        if os.path.exists(file):
            print(f"✅ {file} encontrado")
        else:
            print(f"❌ {file} no encontrado")
    
    # Verificar modelo
    if os.path.exists('inswapper_128.onnx'):
        print("✅ Modelo inswapper_128.onnx encontrado")
    else:
        print("❌ Modelo inswapper_128.onnx no encontrado")
    
    return True

def main():
    """Función principal"""
    print("🚀 CONFIGURACIÓN COMPLETA DE ROOP FUNCIONAL")
    print("=" * 60)
    
    # Configurar repositorio
    if not setup_working_roop():
        print("❌ Error configurando repositorio")
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
    if not setup_gpu():
        print("❌ Error configurando GPU")
        return False
    
    # Probar configuración
    test_setup()
    
    print("\n✅ CONFIGURACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)
    print("📋 AHORA PUEDES USAR:")
    print("\n🎬 Procesamiento en lote:")
    print("!python run_batch_processing.py \\")
    print("  --source /content/DanielaAS.jpg \\")
    print("  --videos /content/video1.mp4 /content/video2.mp4 \\")
    print("  --output-dir /content/resultados \\")
    print("  --max-memory 12 \\")
    print("  --execution-threads 30 \\")
    print("  --temp-frame-quality 100 \\")
    print("  --keep-fps")
    
    print("\n🎬 Procesamiento individual:")
    print("!python run.py \\")
    print("  --source /content/DanielaAS.jpg \\")
    print("  --target /content/video.mp4 \\")
    print("  -o /content/resultado.mp4 \\")
    print("  --frame-processor face_swapper face_enhancer \\")
    print("  --execution-provider cuda \\")
    print("  --keep-fps")
    
    return True

if __name__ == '__main__':
    main() 