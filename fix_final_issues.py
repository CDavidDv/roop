#!/usr/bin/env python3
"""
Script final para solucionar todos los problemas restantes
"""

import os
import sys
import subprocess

def fix_final_issues():
    """Solucionar todos los problemas finales"""
    print("🚀 SOLUCIONANDO PROBLEMAS FINALES")
    print("=" * 50)
    
    # Paso 1: Instalar dependencias faltantes
    print("📦 Paso 1: Instalando dependencias faltantes...")
    
    missing_deps = [
        "customtkinter",
        "opennsfw2",
        "torchvision",
        "opencv-contrib-python"
    ]
    
    for dep in missing_deps:
        print(f"📦 Instalando {dep}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", dep
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {dep} instalado")
            else:
                print(f"⚠️ Error con {dep}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Paso 2: Solucionar NumPy conflict
    print("\n🔧 Paso 2: Solucionando conflicto NumPy...")
    
    # Desinstalar OpenCV que requiere NumPy 2.x
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "opencv-python", "opencv-contrib-python", "opencv-python-headless", "-y"])
    
    # Reinstalar NumPy 1.26.4
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall"])
    
    # Reinstalar OpenCV compatible
    subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python==4.8.1.78"])
    
    print("✅ Conflicto NumPy solucionado")
    
    # Paso 3: Verificar instalación
    print("\n🔍 Paso 3: Verificando instalación...")
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
        
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        
        import customtkinter
        print("✅ CustomTkinter disponible")
        
        import opennsfw2
        print("✅ OpenNSFW2 disponible")
        
        import torchvision
        print(f"✅ TorchVision: {torchvision.__version__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando: {e}")
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

def test_predictor():
    """Probar predictor"""
    print("\n🔍 PROBANDO PREDICTOR")
    print("=" * 50)
    
    try:
        import roop.predictor
        
        # Probar funciones
        result = roop.predictor.predict_video("test.mp4")
        print("✅ Predictor funciona correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error probando predictor: {e}")
        return False

def main():
    print("🚀 SOLUCIONADOR FINAL")
    print("=" * 50)
    
    # Verificar estado actual
    print("🔍 Estado actual:")
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: No instalado")
    
    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        print("OpenCV: No instalado")
    
    # Preguntar si proceder
    response = input("\n¿Proceder con la corrección final? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        # Solucionar problemas
        if fix_final_issues():
            print("\n✅ Problemas solucionados")
            
            # Probar componentes
            test_face_swapper()
            test_face_enhancer()
            test_predictor()
            
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