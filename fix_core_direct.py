#!/usr/bin/env python3
"""
Script para arreglar directamente el archivo core.py
"""

import os
import sys
import subprocess

def fix_core_directly():
    """Arregla directamente el archivo core.py"""
    print("🔧 ARREGLANDO CORE.PY DIRECTAMENTE")
    print("=" * 50)
    
    core_file = "roop/core.py"
    
    if not os.path.exists(core_file):
        print(f"❌ Error: {core_file} no encontrado")
        return False
    
    try:
        # Leer el archivo completo
        with open(core_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar y arreglar problemas específicos
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Problema 1: 'for gpu in gpus:' fuera del bloque try
            if 'for gpu in gpus:' in line and i > 0:
                # Verificar si está dentro del bloque try
                in_try = False
                for j in range(max(0, i-10), i):
                    if 'try:' in lines[j]:
                        in_try = True
                    elif 'except' in lines[j] or 'finally' in lines[j]:
                        in_try = False
                
                if not in_try:
                    # Está fuera del bloque try, agregar indentación
                    fixed_lines.append('            ' + line.strip())
                    print(f"🔧 Arreglado: línea {i+1} - agregada indentación")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
            
            i += 1
        
        # Escribir archivo corregido
        with open(core_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print("✅ Archivo core.py corregido")
        return True
        
    except Exception as e:
        print(f"❌ Error arreglando core.py: {e}")
        return False

def test_core_syntax():
    """Prueba la sintaxis de core.py"""
    print("🧪 PROBANDO SINTAXIS DE CORE.PY")
    print("=" * 50)
    
    test_code = """
import ast
import sys

try:
    with open('roop/core.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar sintaxis
    ast.parse(content)
    print("✅ Sintaxis correcta")
    
    # Intentar importar
    sys.path.insert(0, '.')
    from roop import core
    print("✅ Importación exitosa")
    return True
    
except SyntaxError as e:
    print(f"❌ Error de sintaxis en línea {e.lineno}: {e.msg}")
    return False
except Exception as e:
    print(f"❌ Error: {e}")
    return False
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 ARREGLANDO CORE.PY DIRECTAMENTE")
    print("=" * 60)
    
    # Paso 1: Arreglar archivo
    if not fix_core_directly():
        print("❌ No se pudo arreglar core.py")
        return 1
    
    # Paso 2: Probar sintaxis
    if not test_core_syntax():
        print("❌ core.py aún tiene problemas de sintaxis")
        return 1
    
    print("\n🎉 ¡CORE.PY ARREGLADO EXITOSAMENTE!")
    print("=" * 50)
    print("✅ Error de sintaxis corregido")
    print("✅ Archivo listo para usar")
    print("✅ Puedes ejecutar el procesamiento por lotes ahora")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 