#!/usr/bin/env python3
"""
Script para instalar modelos de InsightFace automáticamente
"""

import sys
import os
import subprocess

def install_insightface_models():
    """Instala los modelos de InsightFace"""
    print("📥 INSTALANDO MODELOS DE INSIGHTFACE")
    print("=" * 50)
    
    try:
        # 1. Verificar que insightface esté instalado
        print("1. Verificando insightface...")
        import insightface
        print("✅ InsightFace instalado")
        
        # 2. Crear directorio de modelos
        print("2. Creando directorio de modelos...")
        models_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
        os.makedirs(models_dir, exist_ok=True)
        print(f"✅ Directorio creado: {models_dir}")
        
        # 3. Descargar modelos automáticamente
        print("3. Descargando modelos...")
        import insightface.app
        
        # Crear app para descargar modelos
        app = insightface.app.FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0)
        print("✅ Modelos descargados automáticamente")
        
        # 4. Verificar archivos descargados
        print("4. Verificando archivos...")
        expected_files = [
            "1k3d68.onnx",
            "2d106det.onnx",
            "buffalo_l.zip",
            "det_10g.onnx",
            "genderage.onnx",
            "w600k_r50.onnx"
        ]
        
        for file in expected_files:
            file_path = os.path.join(models_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {file} - {size:,} bytes")
            else:
                print(f"⚠️ {file} - NO ENCONTRADO")
        
        print("\n🎉 ¡MODELOS INSTALADOS!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    if install_insightface_models():
        print("\n🚀 ¡LISTO PARA USAR!")
        print("=" * 30)
        print("Ahora puedes ejecutar:")
        print("python test_face_detection.py")
        return 0
    else:
        print("\n❌ Error instalando modelos")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 