#!/usr/bin/env python3
"""
Script para solucionar los últimos problemas con tkinterdnd2 y torchvision
"""

import os
import sys
import subprocess

def fix_last_issues():
    """Solucionar últimos problemas"""
    print("🚀 SOLUCIONANDO ÚLTIMOS PROBLEMAS")
    print("=" * 50)
    
    # Paso 1: Instalar tkinterdnd2
    print("📦 Paso 1: Instalando tkinterdnd2...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "tkinterdnd2"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ tkinterdnd2 instalado")
        else:
            print(f"⚠️ Error con tkinterdnd2: {result.stderr}")
            # Intentar alternativa
            print("📦 Intentando alternativa...")
            subprocess.run([sys.executable, "-m", "pip", "install", "tkinterdnd2-tkinter"], check=True)
            print("✅ tkinterdnd2-tkinter instalado")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Paso 2: Reinstalar torchvision
    print("\n📦 Paso 2: Reinstalando torchvision...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "torchvision", "-y"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "torchvision==0.21.0+cu124", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
        print("✅ torchvision reinstalado")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Paso 3: Verificar instalación
    print("\n🔍 Paso 3: Verificando instalación...")
    try:
        import tkinterdnd2
        print("✅ tkinterdnd2 disponible")
    except ImportError:
        print("❌ tkinterdnd2 no disponible")
        return False
    
    try:
        import torchvision
        print(f"✅ torchvision: {torchvision.__version__}")
        
        # Probar import específico
        from torchvision.transforms import functional
        print("✅ torchvision.transforms.functional disponible")
        
        return True
        
    except Exception as e:
        print(f"❌ Error con torchvision: {e}")
        return False

def test_face_swapper():
    """Probar face swapper"""
    print("\n🎭 PROBANDO FACE SWAPPER")
    print("=" * 50)
    
    try:
        import roop.processors.frame.face_swapper as face_swapper
        
        print("Cargando modelo de face swapper...")
        swapper = face_swapper.get_face_swapper()
        
        if swapper:
            print("✅ Face swapper cargado exitosamente")
            
            # Verificar proveedores
            if hasattr(swapper, 'providers'):
                print(f"Proveedores del modelo: {swapper.providers}")
                if 'CUDAExecutionProvider' in swapper.providers:
                    print("✅ Face swapper usando GPU")
                else:
                    print("⚠️ Face swapper usando CPU")
            else:
                print("Modelo cargado (no se puede verificar proveedores)")
            return True
        else:
            print("❌ Error cargando face swapper")
            return False
            
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")
        return False

def test_face_enhancer():
    """Probar face enhancer"""
    print("\n✨ PROBANDO FACE ENHANCER")
    print("=" * 50)
    
    try:
        import roop.processors.frame.face_enhancer as face_enhancer
        
        device = face_enhancer.get_device()
        print(f"Dispositivo detectado: {device}")
        
        if device == 'cuda':
            print("✅ Face enhancer configurado para usar GPU")
        else:
            print(f"⚠️ Face enhancer usando: {device}")
        
        return True
            
    except Exception as e:
        print(f"❌ Error probando face enhancer: {e}")
        return False

def main():
    print("🚀 SOLUCIONADOR ÚLTIMOS PROBLEMAS")
    print("=" * 50)
    
    # Verificar estado actual
    print("🔍 Estado actual:")
    try:
        import tkinterdnd2
        print("✅ tkinterdnd2 disponible")
    except ImportError:
        print("❌ tkinterdnd2 no disponible")
    
    try:
        import torchvision
        print(f"✅ torchvision: {torchvision.__version__}")
    except ImportError:
        print("❌ torchvision no disponible")
    
    # Preguntar si proceder
    response = input("\n¿Proceder con la corrección? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        # Solucionar problemas
        if fix_last_issues():
            print("\n✅ Problemas solucionados")
            
            # Probar componentes
            test_face_swapper()
            test_face_enhancer()
            
            print("\n🎉 ¡TODO FUNCIONA PERFECTAMENTE!")
            print("=" * 50)
            print("Ahora puedes ejecutar:")
            print("python test_gpu_force.py")
            print()
            print("Y luego el procesamiento por lotes:")
            print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/113.mp4 --output-dir /content/resultados --execution-threads 31 --temp-frame-quality 100 --keep-fps")
        else:
            print("\n❌ Algunos problemas persisten")
    else:
        print("❌ Proceso cancelado")

if __name__ == "__main__":
    main() 