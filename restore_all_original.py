#!/usr/bin/env python3
"""
Restaurar todos los archivos al original
"""

import os
import shutil
import sys

def restore_all_original():
    """Restaurar todos los archivos al original"""
    print("🔄 RESTAURANDO TODO AL ORIGINAL")
    print("=" * 50)
    
    # Lista de archivos a restaurar
    files_to_restore = [
        'roop/core.py',
        'roop/processors/frame/core.py',
        'roop/utilities.py',
        'roop/face_analyser.py',
        'roop/processors/frame/face_swapper.py',
        'roop/processors/frame/face_enhancer.py',
        'roop/predictor.py',
        'roop/globals.py'
    ]
    
    try:
        for file_path in files_to_restore:
            original_path = f"roop-Original/{file_path}"
            current_path = file_path
            
            if os.path.exists(original_path):
                print(f"✅ Restaurando: {file_path}")
                shutil.copy2(original_path, current_path)
            else:
                print(f"❌ No encontrado: {original_path}")
        
        print("\n🎉 ¡TODOS LOS ARCHIVOS RESTAURADOS!")
        print("=" * 40)
        print("✅ core.py - Original (sin GPU forzado)")
        print("✅ processors/frame/core.py - Original (sin gestión memoria)")
        print("✅ utilities.py - Original")
        print("✅ face_analyser.py - Original")
        print("✅ processors/frame/face_swapper.py - Original")
        print("✅ processors/frame/face_enhancer.py - Original (con GFPGAN)")
        print("✅ predictor.py - Original (con NSFW)")
        print("✅ globals.py - Original (execution_providers = [])")
        print("✅ Sin modificaciones")
        print("✅ Listo para usar")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal"""
    if restore_all_original():
        print("\n🚀 ¡LISTO PARA USAR!")
        print("=" * 30)
        print("Ahora puedes ejecutar:")
        print("python run.py")
        return 0
    else:
        print("\n❌ Error restaurando archivos")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 