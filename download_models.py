#!/usr/bin/env python3
"""
Script para descargar modelos de InsightFace
"""

import sys
import os
import insightface

def download_insightface_models():
    """Descarga los modelos de InsightFace"""
    print("📥 DESCARGANDO MODELOS DE INSIGHTFACE")
    print("=" * 50)
    
    try:
        # Crear directorio de modelos si no existe
        models_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
        os.makedirs(models_dir, exist_ok=True)
        print(f"📁 Directorio de modelos: {models_dir}")
        
        # Descargar modelos de detección
        print("🔄 Descargando modelos de detección...")
        app = insightface.app.FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0)
        print("✅ Modelos de detección descargados")
        
        # Verificar que los archivos existen
        model_files = [
            "1k3d68.onnx",
            "2d106det.onnx", 
            "buffalo_l.zip",
            "det_10g.onnx",
            "genderage.onnx",
            "w600k_r50.onnx"
        ]
        
        print("\n📋 Verificando archivos descargados:")
        for file in model_files:
            file_path = os.path.join(models_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"✅ {file} - {size:,} bytes")
            else:
                print(f"❌ {file} - NO ENCONTRADO")
        
        print("\n🎉 ¡MODELOS DESCARGADOS EXITOSAMENTE!")
        return True
        
    except Exception as e:
        print(f"❌ Error descargando modelos: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    if download_insightface_models():
        print("\n🚀 ¡LISTO PARA USAR!")
        print("=" * 30)
        print("Ahora puedes ejecutar:")
        print("python test_fixed_face_swap.py")
        return 0
    else:
        print("\n❌ Error descargando modelos")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 