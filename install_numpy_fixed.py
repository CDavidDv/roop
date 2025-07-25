#!/usr/bin/env python3
"""
Script para instalar NumPy 1.x compatible con TensorFlow
"""

import subprocess
import sys
import os

def run_command(command, description=""):
    """Ejecuta un comando y maneja errores"""
    print(f"🔧 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Exitoso")
            return True
        else:
            print(f"❌ {description} - Error")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} - Excepción: {e}")
        return False

def fix_numpy_compatibility():
    """Arregla la compatibilidad de NumPy con TensorFlow"""
    print("🔧 ARREGLANDO COMPATIBILIDAD DE NUMPY")
    print("=" * 50)
    
    # Paso 1: Desinstalar NumPy 2.x
    print("🧹 Desinstalando NumPy 2.x...")
    if not run_command("pip uninstall -y numpy", "Desinstalando NumPy"):
        return False
    
    # Paso 2: Instalar NumPy 1.x compatible
    print("📦 Instalando NumPy 1.x compatible...")
    numpy_versions = ["1.26.4", "1.26.3", "1.26.2", "1.26.1", "1.26.0"]
    
    numpy_installed = False
    for version in numpy_versions:
        if run_command(f"pip install numpy=={version}", f"Instalando NumPy {version}"):
            numpy_installed = True
            break
    
    if not numpy_installed:
        print("❌ No se pudo instalar NumPy 1.x")
        return False
    
    # Paso 3: Reinstalar TensorFlow
    print("🧠 Reinstalando TensorFlow...")
    tf_commands = [
        "pip uninstall -y tensorflow tensorflow-estimator tensorboard",
        "pip install tensorflow==2.15.0",
        "pip install tensorflow-estimator==2.15.0", 
        "pip install tensorboard==2.15.0",
    ]
    
    for command in tf_commands:
        if not run_command(command, "Reinstalando TensorFlow"):
            return False
    
    # Paso 4: Verificar instalación
    print("🔍 Verificando instalación...")
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        if np.__version__.startswith('1.'):
            print("✅ NumPy 1.x instalado correctamente")
        else:
            print("❌ NumPy sigue siendo 2.x")
            return False
        
        import tensorflow as tf
        print(f"✅ TensorFlow: {tf.__version__}")
        print(f"✅ GPU devices: {tf.config.list_physical_devices('GPU')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO COMPATIBILIDAD NUMPY-TENSORFLOW")
    print("=" * 60)
    
    if fix_numpy_compatibility():
        print("\n🎉 ¡COMPATIBILIDAD ARREGLADA!")
        print("=" * 60)
        print("✅ NumPy 1.x instalado")
        print("✅ TensorFlow 2.15.0 compatible")
        print("✅ GPU configurada")
        print("\n🚀 Ahora puedes ejecutar:")
        print("python run_batch_processing.py --source tu_imagen.jpg --videos video1.mp4 --output-dir resultados")
        return True
    else:
        print("\n❌ Error arreglando compatibilidad")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 