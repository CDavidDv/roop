#!/usr/bin/env python3
"""
Script de prueba para verificar la instalación y configuración del proyecto ROOP actualizado
"""

import os
import sys
import warnings
import subprocess
import importlib

# Suprimir warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Probar importaciones de librerías principales"""
    print("🔍 Probando importaciones...")
    
    required_modules = [
        'torch',
        'torchvision',
        'tensorflow',
        'onnxruntime',
        'cv2',
        'insightface',
        'numpy',
        'psutil',
        'tqdm',
        'customtkinter',
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Fallaron {len(failed_imports)} importaciones: {failed_imports}")
        return False
    else:
        print("\n✅ Todas las importaciones exitosas")
        return True

def test_gpu_availability():
    """Probar disponibilidad de GPU"""
    print("\n🔍 Probando GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ PyTorch GPU: {gpu_count} GPU(s) disponible(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"      • GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Test básico de GPU
            device = torch.device('cuda:0')
            x = torch.randn(100, 100).to(device)
            y = torch.mm(x, x)
            print("   ✅ Test de GPU PyTorch exitoso")
            return True
        else:
            print("   ❌ PyTorch GPU no disponible")
            return False
    except Exception as e:
        print(f"   ❌ Error en test de GPU: {e}")
        return False

def test_onnx_gpu():
    """Probar ONNX Runtime GPU"""
    print("\n🔍 Probando ONNX Runtime GPU...")
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"   📋 Proveedores disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("   ✅ ONNX Runtime CUDA disponible")
            return True
        else:
            print("   ❌ ONNX Runtime CUDA no disponible")
            return False
    except Exception as e:
        print(f"   ❌ Error en test de ONNX: {e}")
        return False

def test_roop_modules():
    """Probar módulos de ROOP"""
    print("\n🔍 Probando módulos de ROOP...")
    
    roop_modules = [
        'roop.core',
        'roop.globals',
        'roop.face_analyser',
        'roop.processors.frame.face_swapper',
        'roop.processors.frame.face_enhancer',
        'roop.utilities',
    ]
    
    failed_modules = []
    
    for module in roop_modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ Fallaron {len(failed_modules)} módulos: {failed_modules}")
        return False
    else:
        print("\n✅ Todos los módulos de ROOP cargan correctamente")
        return True

def test_scripts():
    """Probar scripts principales"""
    print("\n🔍 Probando scripts principales...")
    
    scripts = [
        'run_batch_processing.py',
        'gpu_optimization.py',
        'install_updated.py',
    ]
    
    failed_scripts = []
    
    for script in scripts:
        if os.path.exists(script):
            print(f"   ✅ {script} existe")
        else:
            print(f"   ❌ {script} no encontrado")
            failed_scripts.append(script)
    
    if failed_scripts:
        print(f"\n❌ Faltan {len(failed_scripts)} scripts: {failed_scripts}")
        return False
    else:
        print("\n✅ Todos los scripts principales existen")
        return True

def test_command_line():
    """Probar comando de línea de comandos"""
    print("\n🔍 Probando comando de línea de comandos...")
    
    try:
        result = subprocess.run([
            sys.executable, 'run_batch_processing.py', '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("   ✅ Comando --help funciona correctamente")
            return True
        else:
            print(f"   ❌ Error en comando --help: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Timeout en comando --help")
        return False
    except Exception as e:
        print(f"   ❌ Error ejecutando comando: {e}")
        return False

def test_environment():
    """Probar variables de entorno"""
    print("\n🔍 Probando variables de entorno...")
    
    env_vars = [
        'CUDA_VISIBLE_DEVICES',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'TF_CPP_MIN_LOG_LEVEL',
        'OMP_NUM_THREADS',
    ]
    
    for var in env_vars:
        value = os.environ.get(var, 'No configurada')
        print(f"   📋 {var} = {value}")
    
    print("   ✅ Variables de entorno verificadas")

def main():
    """Función principal"""
    print("🧪 PRUEBA DE INSTALACIÓN Y CONFIGURACIÓN")
    print("=" * 50)
    
    tests = [
        ("Importaciones", test_imports),
        ("GPU PyTorch", test_gpu_availability),
        ("ONNX Runtime GPU", test_onnx_gpu),
        ("Módulos ROOP", test_roop_modules),
        ("Scripts principales", test_scripts),
        ("Comando línea", test_command_line),
        ("Variables entorno", test_environment),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ Error en {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Resultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("\n🎉 ¡Todas las pruebas pasaron! El proyecto está listo para usar.")
        print("\n📋 Próximos pasos:")
        print("1. Ejecute: python gpu_optimization.py")
        print("2. Ejecute: python run_batch_processing.py --help")
        print("3. ¡Listo para procesar videos!")
    else:
        print(f"\n⚠️ {total - passed} prueba(s) fallaron. Revise los errores arriba.")
        print("\n💡 Sugerencias:")
        print("1. Ejecute: python install_updated.py")
        print("2. Verifique que CUDA esté instalado")
        print("3. Reinstale las dependencias si es necesario")
    
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 