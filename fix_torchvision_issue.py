#!/usr/bin/env python3
"""
Script para arreglar el problema de torchvision
"""

import os
import sys
import subprocess

def fix_torchvision():
    """Arregla el problema de torchvision"""
    print("🔧 ARREGLANDO PROBLEMA DE TORCHVISION")
    print("=" * 50)
    
    # Desinstalar torchvision actual
    print("🔄 Desinstalando torchvision actual...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "torchvision", "-y"], 
                  capture_output=True, text=True)
    
    # Instalar versión compatible
    print("🔄 Instalando torchvision compatible...")
    subprocess.run([sys.executable, "-m", "pip", "install", "torchvision==0.15.2"], 
                  capture_output=True, text=True)
    
    print("✅ Torchvision arreglado")

def test_torchvision():
    """Prueba que torchvision funcione"""
    print("🧪 PROBANDO TORCHVISION")
    print("=" * 50)
    
    test_code = '''
try:
    import torchvision
    print(f"✅ torchvision importado: {torchvision.__version__}")
    
    # Probar importación problemática
    from torchvision.transforms import functional
    print("✅ functional importado correctamente")
    
    # Probar face_enhancer
    from roop.processors.frame.face_enhancer import get_face_enhancer
    print("✅ face_enhancer importado")
    
    print("✅ Torchvision funcionando correctamente")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
'''
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO TORCHVISION")
    print("=" * 60)
    
    # Paso 1: Arreglar torchvision
    fix_torchvision()
    
    # Paso 2: Probar
    if test_torchvision():
        print("\n🎉 ¡TORCHVISION ARREGLADO!")
        print("=" * 50)
        print("✅ Torchvision funcionando")
        print("✅ Face enhancer disponible")
        print("✅ Listo para procesar videos")
        return 0
    else:
        print("\n❌ Torchvision aún tiene problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 