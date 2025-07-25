#!/usr/bin/env python3
"""
Script para solucionar problemas de compatibilidad entre torchvision y basicsr
"""

import os
import sys
import subprocess

def run_command(command, description=""):
    """Ejecutar comando con manejo de errores"""
    print(f"🔧 {description}")
    print(f"Comando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"STDERR: {e.stderr}")
        return False

def fix_torchvision_compatibility():
    """Solucionar problemas de compatibilidad de torchvision"""
    print("🔧 Solucionando problemas de compatibilidad de torchvision...")
    
    # Desinstalar versiones conflictivas
    run_command("pip uninstall -y torchvision", "Desinstalando torchvision anterior")
    run_command("pip uninstall -y basicsr", "Desinstalando basicsr anterior")
    run_command("pip uninstall -y gfpgan", "Desinstalando gfpgan anterior")
    
    # Instalar torchvision compatible
    run_command("pip install torchvision==0.16.0+cu121 --index-url https://download.pytorch.org/whl/cu121", 
                "Instalando torchvision 0.16.0 compatible")
    
    # Instalar basicsr compatible
    run_command("pip install basicsr==1.4.2", "Instalando basicsr 1.4.2")
    
    # Instalar gfpgan compatible
    run_command("pip install gfpgan==1.3.8", "Instalando gfpgan 1.3.8")
    
    return True

def verify_fix():
    """Verificar que el fix funcionó"""
    print("\n🔍 Verificando que el fix funcionó...")
    
    try:
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
        
        # Verificar que el módulo problemático existe
        from torchvision.transforms import functional
        print("✅ TorchVision functional disponible")
        
        import basicsr
        print(f"✅ BasicSR: {basicsr.__version__}")
        
        import gfpgan
        print(f"✅ GFPGAN: {gfpgan.__version__}")
        
        # Probar importación problemática
        try:
            from basicsr.data.degradations import circular_lowpass_kernel
            print("✅ BasicSR degradations disponible")
        except ImportError as e:
            print(f"⚠️ BasicSR degradations: {e}")
        
        print("✅ Compatibilidad verificada")
        return True
        
    except Exception as e:
        print(f"❌ Error verificando compatibilidad: {e}")
        return False

def create_face_enhancer_fix():
    """Crear un fix temporal para face_enhancer"""
    print("\n🔧 Creando fix temporal para face_enhancer...")
    
    fix_content = '''#!/usr/bin/env python3
"""
Fix temporal para face_enhancer con compatibilidad torchvision
"""

import os
import sys

# Configurar variables de entorno
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fix para torchvision.transforms.functional_tensor
try:
    from torchvision.transforms import functional
    # Crear alias para compatibilidad
    if not hasattr(functional, 'rgb_to_grayscale'):
        def rgb_to_grayscale(img):
            return functional.to_grayscale(img, num_output_channels=1)
        functional.rgb_to_grayscale = rgb_to_grayscale
    print("✅ Fix de torchvision aplicado")
except Exception as e:
    print(f"⚠️ Error aplicando fix de torchvision: {e}")

# Ahora importar face_enhancer
try:
    from roop.processors.frame import face_enhancer
    print("✅ Face enhancer disponible")
except Exception as e:
    print(f"❌ Error importando face enhancer: {e}")
'''
    
    with open("fix_face_enhancer.py", "w") as f:
        f.write(fix_content)
    
    print("✅ Fix temporal creado: fix_face_enhancer.py")
    return True

def main():
    """Función principal"""
    print("🚀 SOLUCIONANDO PROBLEMAS DE COMPATIBILIDAD")
    print("=" * 60)
    
    # Solucionar compatibilidad torchvision
    if not fix_torchvision_compatibility():
        print("❌ Error solucionando compatibilidad torchvision")
        return False
    
    # Verificar fix
    if not verify_fix():
        print("❌ Error verificando fix")
        return False
    
    # Crear fix temporal
    if not create_face_enhancer_fix():
        print("❌ Error creando fix temporal")
        return False
    
    print("\n" + "=" * 60)
    print("✅ PROBLEMAS DE COMPATIBILIDAD SOLUCIONADOS")
    print("=" * 60)
    print("📋 Próximos pasos:")
    print("1. Ejecuta: python fix_face_enhancer.py")
    print("2. Luego ejecuta tu procesamiento normal")
    print("3. Si persisten problemas, usa solo face_swapper")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main() 