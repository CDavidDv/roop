#!/usr/bin/env python3
"""
Instalar dependencias faltantes de ROOP
"""

import os
import sys
import subprocess

def install_missing_dependencies():
    """Instalar dependencias faltantes"""
    print("📦 INSTALANDO DEPENDENCIAS FALTANTES:")
    print("=" * 40)
    
    try:
        # Dependencias faltantes
        missing_deps = [
            "customtkinter",
            "tkinter",
            "tk",
            "pillow",
            "opencv-python",
            "numpy",
            "scipy",
            "scikit-image",
            "insightface",
            "opennsfw2",
            "onnxruntime-gpu",
            "tensorflow==2.12.0"
        ]
        
        for dep in missing_deps:
            print(f"⏳ Instalando {dep}...")
            cmd = [sys.executable, "-m", "pip", "install", dep]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ Dependencias instaladas")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
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
    print("🚀 INSTALANDO DEPENDENCIAS FALTANTES")
    print("=" * 60)
    
    # Instalar dependencias faltantes
    if not install_missing_dependencies():
        print("❌ Error instalando dependencias")
        return False
    
    # Probar importación
    if not test_roop_import():
        print("❌ Error: ROOP no funciona")
        return False
    
    print("\n✅ DEPENDENCIAS INSTALADAS EXITOSAMENTE")
    print("=" * 60)
    print("📋 PRÓXIMOS PASOS:")
    print("1. Procesar videos: python run_roop_original_gpu.py --source imagen.jpg --input-folder videos_entrada --output-dir videos_salida")
    
    return True

if __name__ == '__main__':
    main() 