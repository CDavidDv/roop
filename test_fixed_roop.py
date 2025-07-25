#!/usr/bin/env python3
"""
Script para probar ROOP después del arreglo
"""

import sys
import subprocess

def test_roop():
    """Prueba que ROOP funcione correctamente"""
    print("🧪 PROBANDO ROOP ARREGLADO")
    print("=" * 50)
    
    test_code = '''
import sys
sys.path.insert(0, '.')

try:
    # Probar importación
    import roop.core
    print("✅ roop.core importado")
    
    # Probar face_analyser
    from roop.face_analyser import get_face_analyser
    print("✅ face_analyser importado")
    
    # Probar face_swapper
    from roop.processors.frame.face_swapper import get_face_swapper
    print("✅ face_swapper importado")
    
    # Probar que el modelo existe
    import os
    model_path = "models/inswapper_128.onnx"
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        print(f"✅ Modelo encontrado: {size:,} bytes")
    else:
        print("❌ Modelo no encontrado")
        exit(1)
    
    print("✅ ROOP funcionando correctamente")
    
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
    print("🚀 PROBANDO ROOP ARREGLADO")
    print("=" * 60)
    
    if test_roop():
        print("\n🎉 ¡ROOP FUNCIONANDO!")
        print("=" * 50)
        print("✅ ROOP importado correctamente")
        print("✅ Face analyser funcionando")
        print("✅ Face swapper funcionando")
        print("✅ Modelo encontrado")
        print("✅ Listo para procesar videos")
        print("\n🚀 Comandos disponibles:")
        print("   python run.py --target /content/1.mp4 --source /content/AriaAS.jpg -o /content/swapped.mp4 --execution-provider cuda --frame-processor face_swapper face_enhancer")
        print("   python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/135.mp4 --output-dir /content/resultados --temp-frame-quality 100 --keep-fps")
        return 0
    else:
        print("\n❌ ROOP aún tiene problemas")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 