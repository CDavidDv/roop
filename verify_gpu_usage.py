#!/usr/bin/env python3
"""
Script para verificar el uso de GPU durante el procesamiento de face-swapper
"""

import subprocess
import time
import threading
import os
import sys

def get_gpu_memory():
    """Obtener uso de memoria GPU"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if ',' in line:
                    used, total = map(int, line.split(','))
                    return used, total
    except Exception as e:
        print(f"Error obteniendo memoria GPU: {e}")
    return 0, 0

def get_gpu_utilization():
    """Obtener utilización de GPU"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip().isdigit():
                    return int(line.strip())
    except Exception as e:
        print(f"Error obteniendo utilización GPU: {e}")
    return 0

def monitor_gpu_during_processing():
    """Monitorear GPU durante procesamiento"""
    print("📊 MONITOREO DE GPU DURANTE PROCESAMIENTO")
    print("=" * 50)
    
    # Configurar variables de entorno
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print("Variables de entorno configuradas:")
    for var in ['CUDA_VISIBLE_DEVICES', 'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_CPP_MIN_LOG_LEVEL', 'OMP_NUM_THREADS', 'CUDA_LAUNCH_BLOCKING']:
        print(f"  {var}={os.environ.get(var, 'NO SET')}")
    
    print("\n🔍 Verificando estado inicial de GPU...")
    initial_used, initial_total = get_gpu_memory()
    initial_util = get_gpu_utilization()
    
    print(f"Memoria inicial: {initial_used}MB / {initial_total}MB")
    print(f"Utilización inicial: {initial_util}%")
    
    # Verificar onnxruntime
    print("\n🔍 Verificando ONNX Runtime...")
    try:
        import onnxruntime as ort
        print(f"Versión ONNX Runtime: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"Proveedores disponibles: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDAExecutionProvider disponible")
        else:
            print("❌ CUDAExecutionProvider no disponible")
            
    except Exception as e:
        print(f"❌ Error verificando ONNX Runtime: {e}")
    
    # Probar face swapper
    print("\n🎭 Probando face swapper...")
    try:
        import roop.processors.frame.face_swapper as face_swapper
        
        print("Cargando modelo...")
        start_time = time.time()
        swapper = face_swapper.get_face_swapper()
        load_time = time.time() - start_time
        
        print(f"✅ Modelo cargado en {load_time:.2f}s")
        
        if hasattr(swapper, 'providers'):
            print(f"Proveedores del modelo: {swapper.providers}")
            
            if any('CUDA' in provider for provider in swapper.providers):
                print("✅ GPU CUDA confirmado en uso")
            else:
                print("❌ GPU CUDA no confirmado")
        else:
            print("⚠️ No se pueden verificar proveedores")
        
        # Verificar memoria después de cargar modelo
        after_load_used, after_load_total = get_gpu_memory()
        after_load_util = get_gpu_utilization()
        
        print(f"\nMemoria después de cargar modelo: {after_load_used}MB / {after_load_total}MB")
        print(f"Utilización después de cargar modelo: {after_load_util}%")
        
        memory_increase = after_load_used - initial_used
        if memory_increase > 0:
            print(f"✅ Memoria GPU aumentó en {memory_increase}MB - GPU está siendo usado")
        else:
            print(f"❌ Memoria GPU no aumentó - posible problema")
        
    except Exception as e:
        print(f"❌ Error probando face swapper: {e}")
        import traceback
        traceback.print_exc()

def test_simple_cuda_session():
    """Probar sesión CUDA simple"""
    print("\n🧪 PROBANDO SESIÓN CUDA SIMPLE")
    print("=" * 50)
    
    try:
        import onnxruntime as ort
        import numpy as np
        from onnx import helper
        
        # Crear modelo simple
        X = helper.make_tensor_value_info('X', helper.TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info('Y', helper.TensorProto.FLOAT, [1, 3, 224, 224])
        node = helper.make_node('Identity', inputs=['X'], outputs=['Y'])
        graph = helper.make_graph([node], 'test', [X], [Y])
        model = helper.make_model(graph)
        
        print("Creando sesión con CUDA...")
        start_time = time.time()
        
        session = ort.InferenceSession(
            model.SerializeToString(),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
        creation_time = time.time() - start_time
        print(f"✅ Sesión creada en {creation_time:.2f}s")
        print(f"Proveedores aplicados: {session.get_providers()}")
        
        # Probar inferencia
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        print("Ejecutando inferencia...")
        start_time = time.time()
        
        output = session.run(['Y'], {'X': input_data})
        
        inference_time = time.time() - start_time
        print(f"✅ Inferencia completada en {inference_time:.4f}s")
        
        # Verificar memoria después de inferencia
        after_inference_used, after_inference_total = get_gpu_memory()
        after_inference_util = get_gpu_utilization()
        
        print(f"Memoria después de inferencia: {after_inference_used}MB / {after_inference_total}MB")
        print(f"Utilización después de inferencia: {after_inference_util}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando sesión CUDA: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_diagnostic_report():
    """Generar reporte de diagnóstico"""
    print("\n📋 GENERANDO REPORTE DE DIAGNÓSTICO")
    print("=" * 50)
    
    report = []
    report.append("🔧 DIAGNÓSTICO DE GPU PARA FACE-SWAPPER")
    report.append("=" * 60)
    
    # Información del sistema
    try:
        import platform
        report.append(f"Sistema: {platform.system()} {platform.release()}")
    except:
        pass
    
    # Información de GPU
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(',')
            if len(gpu_info) >= 2:
                report.append(f"GPU: {gpu_info[0].strip()}")
                report.append(f"VRAM Total: {gpu_info[1].strip()}MB")
    except:
        report.append("GPU: No se pudo obtener información")
    
    # Información de ONNX Runtime
    try:
        import onnxruntime as ort
        report.append(f"ONNX Runtime: {ort.__version__}")
        providers = ort.get_available_providers()
        report.append(f"Proveedores: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            report.append("✅ CUDA disponible")
        else:
            report.append("❌ CUDA no disponible")
    except Exception as e:
        report.append(f"❌ Error ONNX Runtime: {e}")
    
    # Estado de memoria
    used, total = get_gpu_memory()
    util = get_gpu_utilization()
    report.append(f"Memoria actual: {used}MB / {total}MB")
    report.append(f"Utilización actual: {util}%")
    
    # Guardar reporte
    with open('diagnostico_gpu.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("✅ Reporte guardado en 'diagnostico_gpu.txt'")
    
    # Mostrar reporte
    print("\n" + '\n'.join(report))

def main():
    """Función principal"""
    print("🔍 VERIFICADOR DE USO DE GPU")
    print("=" * 60)
    
    # Generar reporte de diagnóstico
    generate_diagnostic_report()
    
    # Monitorear GPU durante procesamiento
    monitor_gpu_during_processing()
    
    # Probar sesión CUDA simple
    test_simple_cuda_session()
    
    print("\n🎉 VERIFICACIÓN COMPLETADA")
    print("=" * 60)
    print("📋 RESUMEN:")
    print("- Si ves '✅ GPU CUDA confirmado en uso', el problema está resuelto")
    print("- Si ves '❌ GPU CUDA no confirmado', ejecuta: python fix_face_swapper_gpu.py")
    print("- Si la memoria GPU no aumenta, hay un problema de configuración")
    print("- El archivo 'diagnostico_gpu.txt' contiene información detallada")

if __name__ == "__main__":
    main() 