#!/usr/bin/env python3
"""
Script simple para probar ROOP
"""

import os
import sys

def test_roop():
    """Prueba que ROOP funcione correctamente"""
    print("🧪 PROBANDO ROOP")
    print("=" * 50)
    
    # Verificar que el modelo existe
    model_path = "models/inswapper_128.onnx"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Modelo encontrado: {size:,} bytes")
    else:
        print("❌ Modelo no encontrado")
        return False
    
    # Probar importación de ROOP
    try:
        import roop.core
        print("✅ roop.core importado")
    except Exception as e:
        print(f"❌ Error importando roop.core: {e}")
        return False
    
    # Probar face_swapper
    try:
        from roop.processors.frame.face_swapper import get_face_swapper
        print("✅ face_swapper importado")
    except Exception as e:
        print(f"❌ Error importando face_swapper: {e}")
        return False
    
    print("✅ ROOP funcionando correctamente")
    return True

def main():
    """Función principal"""
    print("🚀 PROBANDO ROOP")
    print("=" * 60)
    
    if test_roop():
        print("\n🎉 ¡ROOP FUNCIONANDO!")
        print("=" * 50)
        print("✅ Modelo descargado correctamente")
        print("✅ ROOP importado correctamente")
        print("✅ Listo para procesar videos")
        print("\n🚀 Comandos disponibles:")
        print("   python run.py --target /content/1.mp4 --source /content/AriaAS.jpg -o /content/swapped.mp4 --execution-provider cuda --frame-processor face_swapper face_enhancer")
        print("   python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
        return 0
    else:
        print("\n❌ ROOP no funciona correctamente")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 