#!/usr/bin/env python3
"""
Ejemplo de uso del procesamiento por lotes con ROOP
Configuración optimizada para Google Colab con Tesla T4
"""

import subprocess
import sys

def run_batch_processing_example():
    """Ejecutar procesamiento por lotes con configuración optimizada"""
    
    print("🚀 EJEMPLO DE PROCESAMIENTO POR LOTES")
    print("=" * 60)
    print("Configuración optimizada para Google Colab Tesla T4:")
    print("• 31 hilos de ejecución")
    print("• 12GB RAM máxima")
    print("• 30s espera GPU entre procesadores")
    print("• Calidad de frames: 100")
    print("• Mantener FPS original")
    print("=" * 60)
    
    # Comando de ejemplo (modificar según tus archivos)
    cmd = [
        sys.executable, 'run_batch_processing.py',
        '--source', '/content/DanielaAS.jpg',
        '--videos', 
        '/content/113.mp4', '/content/114.mp4', '/content/115.mp4', 
        '/content/116.mp4', '/content/117.mp4', '/content/118.mp4', 
        '/content/119.mp4', '/content/120.mp4',
        '--output-dir', '/content/resultados',
        '--execution-threads', '31',
        '--temp-frame-quality', '100',
        '--max-memory', '12',
        '--gpu-memory-wait', '30',
        '--keep-fps'
    ]
    
    print("Comando a ejecutar:")
    print(" ".join(cmd))
    print("\n" + "=" * 60)
    
    # Preguntar si ejecutar
    response = input("¿Ejecutar el procesamiento por lotes? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'sí', 'si']:
        print("\n🔄 Iniciando procesamiento...")
        try:
            result = subprocess.run(cmd, check=True)
            print("\n✅ Procesamiento completado exitosamente!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error en el procesamiento: {e}")
    else:
        print("❌ Procesamiento cancelado")

def show_usage_instructions():
    """Mostrar instrucciones de uso"""
    print("\n📖 INSTRUCCIONES DE USO:")
    print("=" * 60)
    print("1. Asegúrate de tener los archivos en las rutas correctas:")
    print("   • Imagen fuente: /content/DanielaAS.jpg")
    print("   • Videos: /content/113.mp4, /content/114.mp4, etc.")
    print("   • Directorio de salida: /content/resultados")
    print()
    print("2. Ejecuta el script:")
    print("   python run_batch_example.py")
    print()
    print("3. O ejecuta directamente:")
    print("   python run_batch_processing.py \\")
    print("     --source /content/DanielaAS.jpg \\")
    print("     --videos /content/113.mp4 /content/114.mp4 /content/115.mp4 \\")
    print("     --output-dir /content/resultados \\")
    print("     --execution-threads 31 \\")
    print("     --temp-frame-quality 100 \\")
    print("     --keep-fps")
    print()
    print("4. El script mostrará el progreso en tiempo real de cada video")
    print("5. Los resultados se guardarán en /content/resultados/")
    print("=" * 60)

def main():
    print("🎭 ROOP - PROCESAMIENTO POR LOTES")
    print("Configuración optimizada para GPU Tesla T4")
    print()
    
    show_usage_instructions()
    
    # Ejecutar ejemplo
    run_batch_processing_example()

if __name__ == "__main__":
    main() 