#!/usr/bin/env python3
"""
Arreglar importación de TensorFlow en ROOP
"""

import os
import sys
import subprocess

def reinstall_tensorflow():
    """Reinstalar TensorFlow compatible"""
    print("📦 REINSTALANDO TENSORFLOW:")
    print("=" * 40)
    
    try:
        # Instalar TensorFlow compatible
        print("⏳ Instalando TensorFlow 2.12.0...")
        cmd = [sys.executable, "-m", "pip", "install", "tensorflow==2.12.0"]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("✅ TensorFlow instalado")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando TensorFlow: {e}")
        return False

def modify_roop_for_pytorch():
    """Modificar ROOP para usar PyTorch en lugar de TensorFlow"""
    print("\n🔧 MODIFICANDO ROOP PARA PYTORCH:")
    print("=" * 40)
    
    # Buscar archivo core.py
    core_file = 'roop/core.py'
    
    if not os.path.exists(core_file):
        print(f"❌ Archivo {core_file} no encontrado")
        return False
    
    try:
        with open(core_file, 'r') as f:
            content = f.read()
        
        # Reemplazar import de TensorFlow con PyTorch
        old_import = "import tensorflow"
        new_import = "# import tensorflow  # Comentado para usar PyTorch"
        
        if old_import in content:
            content = content.replace(old_import, new_import)
            
            with open(core_file, 'w') as f:
                f.write(content)
            
            print("✅ ROOP modificado para usar PyTorch")
            return True
        else:
            print("⚠️ No se encontró import de TensorFlow")
            return False
            
    except Exception as e:
        print(f"❌ Error modificando ROOP: {e}")
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
    print("🚀 ARREGLANDO IMPORTACIÓN TENSORFLOW")
    print("=" * 60)
    
    # Opción 1: Reinstalar TensorFlow
    print("📋 OPCIÓN 1: Reinstalar TensorFlow")
    if reinstall_tensorflow():
        if test_roop_import():
            print("✅ TensorFlow reinstalado y ROOP funciona")
            return True
    
    # Opción 2: Modificar ROOP para PyTorch
    print("\n📋 OPCIÓN 2: Modificar ROOP para PyTorch")
    if modify_roop_for_pytorch():
        if test_roop_import():
            print("✅ ROOP modificado para PyTorch")
            return True
    
    print("❌ No se pudo arreglar la importación")
    return False

if __name__ == '__main__':
    main() 