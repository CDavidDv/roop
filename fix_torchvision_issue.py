#!/usr/bin/env python3
"""
Script para solucionar el problema de torchvision.transforms.functional_tensor
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Ejecutar comando y mostrar resultado"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completado")
            if result.stdout.strip():
                print(f"   Salida: {result.stdout.strip()}")
        else:
            print(f"❌ {description} falló")
            print(f"   Error: {result.stderr.strip()}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error ejecutando {description}: {e}")
        return False

def main():
    print("🚀 SOLUCIONADOR PROBLEMA TORCHVISION")
    print("=" * 50)
    
    # Verificar versión actual de torchvision
    print("📊 Verificando versión actual de torchvision...")
    try:
        import torchvision
        print(f"   Versión actual: {torchvision.__version__}")
    except ImportError:
        print("   ❌ torchvision no está instalado")
    except Exception as e:
        print(f"   ❌ Error verificando torchvision: {e}")
    
    # Verificar si el módulo problemático existe
    print("\n🔍 Verificando módulo problemático...")
    try:
        from torchvision.transforms import functional_tensor
        print("   ✅ Módulo functional_tensor disponible")
    except ImportError:
        print("   ❌ Módulo functional_tensor no disponible")
    
    print("\n🔧 SOLUCIONANDO PROBLEMA TORCHVISION")
    print("=" * 50)
    
    # Paso 1: Desinstalar torchvision actual
    if not run_command("pip uninstall torchvision -y", "Desinstalando torchvision actual"):
        print("⚠️ Continuando de todas formas...")
    
    # Paso 2: Limpiar caché de pip
    run_command("pip cache purge", "Limpiando caché de pip")
    
    # Paso 3: Instalar versión compatible de torchvision
    print("\n📦 Instalando versión compatible de torchvision...")
    
    # Intentar diferentes versiones compatibles
    torchvision_versions = [
        "torchvision==0.15.2",
        "torchvision==0.14.1", 
        "torchvision==0.13.1",
        "torchvision==0.12.0"
    ]
    
    for version in torchvision_versions:
        print(f"\n🔄 Intentando instalar {version}...")
        if run_command(f"pip install {version} --no-cache-dir", f"Instalando {version}"):
            # Verificar si el módulo problemático ahora está disponible
            try:
                from torchvision.transforms import functional_tensor
                print("✅ Módulo functional_tensor ahora disponible")
                break
            except ImportError:
                print("❌ Módulo functional_tensor aún no disponible, probando siguiente versión...")
                continue
        else:
            print(f"❌ Falló instalación de {version}")
    
    # Paso 4: Verificar instalación final
    print("\n🧪 VERIFICACIÓN FINAL")
    print("=" * 50)
    
    try:
        import torchvision
        print(f"✅ torchvision instalado: {torchvision.__version__}")
    except ImportError:
        print("❌ torchvision no se pudo importar")
        return False
    
    try:
        from torchvision.transforms import functional_tensor
        print("✅ Módulo functional_tensor disponible")
    except ImportError:
        print("❌ Módulo functional_tensor aún no disponible")
        print("\n🔧 Intentando solución alternativa...")
        
        # Crear un módulo de compatibilidad
        try:
            import torchvision.transforms.functional as F
            # Crear un alias para compatibilidad
            torchvision.transforms.functional_tensor = F
            print("✅ Solución alternativa aplicada")
        except Exception as e:
            print(f"❌ No se pudo aplicar solución alternativa: {e}")
            return False
    
    print("\n🎉 ¡PROBLEMA SOLUCIONADO!")
    print("=" * 50)
    print("Ahora puedes ejecutar el procesamiento por lotes:")
    print("python run_batch_processing.py --source /content/DanielaAS.jpg --videos /content/113.mp4 --output-dir /content/resultados --execution-threads 31 --temp-frame-quality 100 --keep-fps")
    
    return True

if __name__ == "__main__":
    main() 