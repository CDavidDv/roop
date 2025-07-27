#!/usr/bin/env python3
"""
Instalar dependencias ROOP (corregido)
"""

import os
import sys
import subprocess

def install_roop_dependencies():
    """Instalar dependencias ROOP"""
    print("📦 INSTALANDO DEPENDENCIAS ROOP:")
    print("=" * 40)
    
    try:
        # Dependencias que se pueden instalar con pip
        dependencies = [
            "customtkinter",
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
        
        print("✅ Dependencias ROOP instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def test_tkinter():
    """Probar tkinter"""
    print("\n🧪 PROBANDO TKINTER:")
    print("=" * 40)
    
    try:
        import tkinter as tk
        print("✅ tkinter disponible")
        
        # Crear ventana de prueba
        root = tk.Tk()
        root.withdraw()  # Ocultar ventana
        print("✅ tkinter funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error con tkinter: {e}")
        return False

def test_customtkinter():
    """Probar customtkinter"""
    print("\n🧪 PROBANDO CUSTOMTKINTER:")
    print("=" * 40)
    
    try:
        import customtkinter as ctk
        print("✅ customtkinter disponible")
        return True
        
    except Exception as e:
        print(f"❌ Error con customtkinter: {e}")
        return False

def test_roop_import():
    """Probar importación de ROOP"""
    print("\n🧪 PROBANDO IMPORTACIÓN ROOP:")
    print("=" * 40)
    
    try:
        # Probar import de ROOP
        import sys
        sys.path.insert(0, '.')
        
        from roop import core
        print("✅ ROOP importado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error importando ROOP: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 INSTALANDO DEPENDENCIAS ROOP (CORREGIDO)")
    print("=" * 60)
    
    # Instalar dependencias ROOP
    if not install_roop_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Probar tkinter
    if not test_tkinter():
        print("❌ Error: tkinter no funciona")
        return False
    
    # Probar customtkinter
    if not test_customtkinter():
        print("❌ Error: customtkinter no funciona")
        return False
    
    # Probar importación ROOP
    if not test_roop_import():
        print("❌ Error: ROOP no funciona")
        return False
    
    print("\n✅ DEPENDENCIAS ROOP INSTALADAS EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    
    return True

if __name__ == '__main__':
    main() 