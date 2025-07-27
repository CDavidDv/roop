#!/usr/bin/env python3
"""
Instalación rápida para Google Colab
Combina todos los pasos necesarios en un solo script
"""

import os
import sys
import subprocess

def main():
    """Instalación completa para Google Colab"""
    print("🚀 INSTALACIÓN RÁPIDA PARA GOOGLE COLAB T4")
    print("=" * 60)
    
    # Paso 1: Verificar que estamos en el directorio correcto
    if not os.path.exists('run.py'):
        print("❌ Error: No estamos en el directorio de ROOP")
        print("📋 Ejecuta primero:")
        print("   !git clone https://github.com/s0md3v/roop.git")
        print("   %cd roop")
        return False
    
    print("✅ Directorio ROOP detectado")
    
    # Paso 2: Instalar ROOP y dependencias
    print("\n📦 INSTALANDO ROOP Y DEPENDENCIAS...")
    try:
        result = subprocess.run([sys.executable, "install_roop_colab.py"], 
                              check=True, capture_output=True, text=True)
        print("✅ ROOP instalado correctamente")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando ROOP: {e}")
        return False
    
    # Paso 3: Optimizar GPU
    print("\n⚡ OPTIMIZANDO GPU...")
    try:
        result = subprocess.run([sys.executable, "optimize_colab_gpu.py"], 
                              check=True, capture_output=True, text=True)
        print("✅ GPU optimizada correctamente")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error optimizando GPU: {e}")
        return False
    
    print("\n🎉 INSTALACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 60)
    print("📋 AHORA PUEDES USAR:")
    print("\n🎬 Procesamiento individual:")
    print("!python run_colab_gpu_optimized.py \\")
    print("  --source imagen.jpg \\")
    print("  --target video.mp4 \\")
    print("  -o resultado.mp4 \\")
    print("  --gpu-memory-wait 30 \\")
    print("  --keep-fps")
    
    print("\n📁 Procesamiento en lote:")
    print("!python run_colab_gpu_optimized.py \\")
    print("  --source imagen.jpg \\")
    print("  --target carpeta_videos \\")
    print("  --batch \\")
    print("  --output-dir resultados \\")
    print("  --gpu-memory-wait 30 \\")
    print("  --keep-fps")
    
    return True

if __name__ == '__main__':
    main() 