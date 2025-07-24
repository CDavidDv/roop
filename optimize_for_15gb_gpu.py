#!/usr/bin/env python3
import os
import sys
import subprocess
import json
from typing import Dict, List

class RoopOptimizer:
    def __init__(self):
        self.gpu_vram_gb = 15  # Tu GPU tiene 15GB
        self.ram_gb = 12       # Tu RAM es de 12GB
        
    def get_optimal_settings(self) -> Dict:
        """Obtener configuraciones óptimas para 15GB VRAM"""
        return {
            # Configuraciones de memoria
            'max_memory': 8,  # Limitar RAM a 8GB para dejar espacio para VRAM
            'gpu_memory_wait': 5,  # Esperar 10s entre procesadores
            
            # Configuraciones de ejecución
            'execution_provider': 'cuda',
            'execution_threads': 8,
            
            # Configuraciones de video
            'temp_frame_format': 'jpg',  # Ahorrar espacio en disco
            'temp_frame_quality': 100,    # Calidad balanceada
            'output_video_encoder': 'h264_nvenc',  # Usar encoder NVIDIA
            'output_video_quality': 100,  # Calidad balanceada
            
            # Configuraciones de procesamiento
            'keep_fps': True,  # Mantener FPS original
            'skip_audio': False,  # Mantener audio
            'many_faces': False,  # Procesar solo la cara principal por defecto
            
            # Configuraciones de detección
            'similar_face_distance': 0.85,
            'reference_face_position': 0,
            'reference_frame_number': 0
        }
    
    def create_optimized_command(self, source_path: str, target_path: str, output_path: str, 
                                custom_args: List[str] = None) -> List[str]:
        """Crear comando optimizado para 15GB VRAM"""
        settings = self.get_optimal_settings()
        
        cmd = [
            'python', 'run.py',
            '-s', source_path,
            '-t', target_path,
            '-o', output_path,
            '--max-memory', str(settings['max_memory']),
            '--execution-provider', settings['execution_provider'],
            '--execution-threads', str(settings['execution_threads']),
            '--gpu-memory-wait', str(settings['gpu_memory_wait']),
            '--temp-frame-format', settings['temp_frame_format'],
            '--temp-frame-quality', str(settings['temp_frame_quality']),
            '--output-video-encoder', settings['output_video_encoder'],
            '--output-video-quality', str(settings['output_video_quality']),
            '--similar-face-distance', str(settings['similar_face_distance']),
            '--reference-face-position', str(settings['reference_face_position']),
            '--reference-frame-number', str(settings['reference_frame_number'])
        ]
        
        # Agregar flags booleanos
        if settings['keep_fps']:
            cmd.append('--keep-fps')
        if settings['skip_audio']:
            cmd.append('--skip-audio')
        if settings['many_faces']:
            cmd.append('--many-faces')
        
        # Agregar argumentos personalizados
        if custom_args:
            cmd.extend(custom_args)
        
        return cmd
    
    def check_system_resources(self) -> Dict:
        """Verificar recursos del sistema"""
        try:
            import psutil
            import subprocess
            
            # Información de RAM
            memory = psutil.virtual_memory()
            ram_info = {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'percent_used': memory.percent
            }
            
            # Información de GPU
            try:
                result = subprocess.run([
                    'nvidia-smi', 
                    '--query-gpu=name,memory.total,memory.free',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if lines and lines[0].strip():
                        parts = lines[0].split(', ')
                        if len(parts) >= 3:
                            gpu_info = {
                                'name': parts[0],
                                'total_memory_mb': int(parts[1]),
                                'free_memory_mb': int(parts[2]),
                                'total_memory_gb': int(parts[1]) / 1024,
                                'free_memory_gb': int(parts[2]) / 1024
                            }
                        else:
                            gpu_info = None
                    else:
                        gpu_info = None
                else:
                    gpu_info = None
            except:
                gpu_info = None
            
            return {
                'ram': ram_info,
                'gpu': gpu_info
            }
            
        except Exception as e:
            print(f"Error verificando recursos: {e}")
            return {}
    
    def print_optimization_report(self, source_path: str, target_path: str, output_path: str):
        """Imprimir reporte de optimización"""
        print("🎯 OPTIMIZACIÓN PARA GPU DE 15GB")
        print("=" * 60)
        
        # Verificar recursos
        resources = self.check_system_resources()
        
        if resources.get('gpu'):
            gpu = resources['gpu']
            print(f"✅ GPU detectada: {gpu['name']}")
            print(f"📊 VRAM total: {gpu['total_memory_gb']:.1f}GB")
            print(f"📊 VRAM libre: {gpu['free_memory_gb']:.1f}GB")
            
            if gpu['total_memory_gb'] >= 14:  # 15GB o más
                print("✅ VRAM suficiente para procesamiento optimizado")
            else:
                print("⚠️ VRAM menor a 15GB - ajustando configuraciones")
        else:
            print("❌ No se detectó GPU NVIDIA")
        
        if resources.get('ram'):
            ram = resources['ram']
            print(f"🧠 RAM total: {ram['total_gb']:.1f}GB")
            print(f"🧠 RAM disponible: {ram['available_gb']:.1f}GB")
            print(f"🧠 RAM usada: {ram['percent_used']:.1f}%")
            
            if ram['percent_used'] > 80:
                print("⚠️ RAM alta - considera cerrar otras aplicaciones")
            else:
                print("✅ RAM en buen estado")
        
        # Mostrar configuraciones óptimas
        settings = self.get_optimal_settings()
        print("\n⚙️ CONFIGURACIONES ÓPTIMAS:")
        print(f"  • RAM máxima: {settings['max_memory']}GB")
        print(f"  • Proveedor: {settings['execution_provider']}")
        print(f"  • Threads: {settings['execution_threads']}")
        print(f"  • Espera GPU: {settings['gpu_memory_wait']}s")
        print(f"  • Formato frames: {settings['temp_frame_format']}")
        print(f"  • Encoder: {settings['output_video_encoder']}")
        print(f"  • Calidad video: {settings['output_video_quality']}")
        
        print("\n📁 ARCHIVOS:")
        print(f"  • Origen: {source_path}")
        print(f"  • Destino: {target_path}")
        print(f"  • Salida: {output_path}")
        
        # Verificar archivos
        if not os.path.exists(source_path):
            print(f"❌ Error: No existe {source_path}")
            return False
        
        if not os.path.exists(target_path):
            print(f"❌ Error: No existe {target_path}")
            return False
        
        print("✅ Archivos verificados")
        return True
    
    def run_optimized_processing(self, source_path: str, target_path: str, output_path: str, 
                                custom_args: List[str] = None, monitor_gpu: bool = True):
        """Ejecutar procesamiento optimizado"""
        
        # Verificar recursos
        if not self.print_optimization_report(source_path, target_path, output_path):
            return False
        
        # Crear comando optimizado
        cmd = self.create_optimized_command(source_path, target_path, output_path, custom_args)
        
        print("\n🚀 EJECUTANDO PROCESAMIENTO OPTIMIZADO")
        print("=" * 60)
        print(f"Comando: {' '.join(cmd)}")
        print("=" * 60)
        
        # Iniciar monitoreo si se solicita
        if monitor_gpu:
            try:
                from monitor_gpu_advanced import GPUMonitor
                monitor = GPUMonitor()
                monitor.start_monitoring()
                print("📊 Monitoreo de GPU iniciado")
            except ImportError:
                print("⚠️ No se pudo iniciar monitoreo de GPU")
        
        try:
            # Ejecutar comando
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Monitorear salida
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(f"[ROOP] {output.strip()}")
            
            return_code = process.poll()
            
            if return_code == 0:
                print("\n✅ PROCESAMIENTO COMPLETADO EXITOSAMENTE!")
                print(f"📁 Archivo de salida: {output_path}")
            else:
                print(f"\n❌ PROCESAMIENTO FALLÓ (código: {return_code})")
                return False
                
        except KeyboardInterrupt:
            print("\n🛑 Procesamiento interrumpido por el usuario")
            process.terminate()
            return False
        except Exception as e:
            print(f"\n❌ Error durante el procesamiento: {e}")
            return False
        finally:
            if monitor_gpu and 'monitor' in locals():
                monitor.stop_monitoring()
        
        return True

def main():
    if len(sys.argv) < 4:
        print("🎯 OPTIMIZADOR PARA GPU DE 15GB")
        print("=" * 40)
        print("Uso: python optimize_for_15gb_gpu.py <imagen_origen> <video_destino> <archivo_salida> [args_adicionales...]")
        print("\nEjemplos:")
        print("  python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4")
        print("  python optimize_for_15gb_gpu.py cara.jpg video.mp4 resultado.mp4 --many-faces --keep-fps")
        print("\nArgumentos adicionales:")
        print("  --many-faces     # Procesar todas las caras")
        print("  --keep-fps       # Mantener FPS original")
        print("  --skip-audio     # Omitir audio")
        print("  --no-monitor     # No monitorear GPU")
        return
    
    source_path = sys.argv[1]
    target_path = sys.argv[2]
    output_path = sys.argv[3]
    custom_args = sys.argv[4:] if len(sys.argv) > 4 else []
    
    # Verificar si se debe monitorear GPU
    monitor_gpu = '--no-monitor' not in custom_args
    if '--no-monitor' in custom_args:
        custom_args.remove('--no-monitor')
    
    # Crear optimizador y ejecutar
    optimizer = RoopOptimizer()
    success = optimizer.run_optimized_processing(source_path, target_path, output_path, custom_args, monitor_gpu)
    
    if success:
        print("\n🎉 ¡Procesamiento completado con optimizaciones para 15GB VRAM!")
    else:
        print("\n💡 Consejos:")
        print("  • Verifica que tienes suficiente espacio en disco")
        print("  • Cierra otras aplicaciones que usen GPU")
        print("  • Considera usar --skip-audio si no necesitas audio")
        print("  • Si tienes problemas, prueba con --no-monitor")

if __name__ == "__main__":
    main() 