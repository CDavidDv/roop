#!/usr/bin/env python3
"""
Descargar modelo como en el original
"""

import os
import subprocess
import sys

def download_model():
    """Descarga el modelo como en el original"""
    print("📥 DESCARGANDO MODELO ORIGINAL")
    print("=" * 40)
    
    # Crear directorio models si no existe
    os.makedirs("models", exist_ok=True)
    
    # Descargar modelo como en el original
    model_url = "https://huggingface.co/countfloyd/deepfake/resolve/main/inswapper_128.onnx"
    model_path = "models/inswapper_128.onnx"
    
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Modelo ya existe: {size:,} bytes")
        return True
    
    print(f"🔄 Descargando: {model_url}")
    
    try:
        # Usar wget como en el original
        result = subprocess.run([
            "wget", "-O", model_path, model_url
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            size = os.path.getsize(model_path)
            print(f"✅ Modelo descargado: {size:,} bytes")
            return True
        else:
            print(f"❌ Error descargando: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal"""
    if download_model():
        print("\n🎉 ¡MODELO DESCARGADO!")
        print("=" * 30)
        print("Ahora puedes ejecutar el procesamiento")
        return 0
    else:
        print("\n❌ Error descargando modelo")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 