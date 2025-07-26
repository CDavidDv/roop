#!/usr/bin/env python3
"""
Ejecutar ROOP original pero con GPU
"""

import sys
import os
import onnxruntime as ort

# Configurar GPU antes de importar roop
def setup_gpu():
    """Configurar GPU para ROOP"""
    print("🚀 CONFIGURANDO GPU PARA ROOP ORIGINAL")
    print("=" * 50)
    
    # Verificar proveedores disponibles
    available_providers = ort.get_available_providers()
    print(f"✅ Proveedores disponibles: {available_providers}")
    
    if 'CUDAExecutionProvider' in available_providers:
        print("✅ CUDA detectado - configurando GPU")
        # Configurar GPU en globals
        import roop.globals
        roop.globals.execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("✅ GPU configurado: CUDAExecutionProvider, CPUExecutionProvider")
        return True
    else:
        print("⚠️ CUDA no disponible - usando CPU")
        import roop.globals
        roop.globals.execution_providers = ['CPUExecutionProvider']
        print("⚠️ CPU configurado: CPUExecutionProvider")
        return False

def main():
    """Función principal"""
    # Configurar GPU antes de importar roop
    gpu_available = setup_gpu()
    
    # Importar roop después de configurar GPU
    import roop.core
    
    print("\n🎯 EJECUTANDO ROOP ORIGINAL CON GPU")
    print("=" * 40)
    
    if gpu_available:
        print("✅ Usando GPU para procesamiento")
    else:
        print("⚠️ Usando CPU (GPU no disponible)")
    
    # Ejecutar ROOP
    roop.core.run()

if __name__ == "__main__":
    main() 