#!/usr/bin/env python3
"""
Script de instalación automática de dependencias para ROOP
"""

import subprocess
import sys
import os

def install_package(package_name):
    """Instalar un paquete usando pip"""
    try:
        print(f"📦 Instalando {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} instalado exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package_name}: {e}")
        return False

def install_dependencies():
    """Instalar todas las dependencias necesarias"""
    
    print("🚀 INSTALANDO DEPENDENCIAS PARA ROOP")
    print("=" * 60)
    
    # Lista de dependencias necesarias
    dependencies = [
        "opencv-python",
        "pillow", 
        "onnxruntime-gpu",
        "opennsfw2",
        "insightface",
        "onnx",
        "tensorflow",
        "tensorflow-gpu",
        "albumentations",
        "scikit-image",
        "scipy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "psutil",
        "GPUtil"
    ]
    
    successful = 0
    failed = 0
    
    for package in dependencies:
        if install_package(package):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE INSTALACIÓN")
    print("=" * 60)
    print(f"✅ Paquetes instalados exitosamente: {successful}")
    print(f"❌ Paquetes fallidos: {failed}")
    
    if failed == 0:
        print("🎉 ¡Todas las dependencias instaladas correctamente!")
    else:
        print("⚠️ Algunos paquetes fallaron. Intenta instalarlos manualmente.")
    
    print("=" * 60)
    
    return failed == 0

def verify_installation():
    """Verificar que las dependencias principales estén instaladas"""
    
    print("\n🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 60)
    
    critical_packages = [
        "cv2",  # opencv-python
        "PIL",  # pillow
        "onnxruntime",  # onnxruntime-gpu
        "opennsfw2",  # opennsfw2
        "insightface",  # insightface
        "torch",  # torch
        "numpy"  # numpy
    ]
    
    all_ok = True
    
    for package in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NO INSTALADO")
            all_ok = False
    
    if all_ok:
        print("\n🎉 ¡Todas las dependencias críticas están instaladas!")
    else:
        print("\n⚠️ Algunas dependencias críticas faltan.")
    
    return all_ok

def main():
    """Función principal"""
    
    print("🎯 INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS")
    print("=" * 60)
    
    # Instalar dependencias
    install_success = install_dependencies()
    
    # Verificar instalación
    verify_success = verify_installation()
    
    if install_success and verify_success:
        print("\n✅ CONFIGURACIÓN COMPLETADA")
        print("🚀 Listo para usar ROOP con GPU")
    else:
        print("\n⚠️ CONFIGURACIÓN INCOMPLETA")
        print("💡 Revisa los errores arriba")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 