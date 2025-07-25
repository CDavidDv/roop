#!/usr/bin/env python3
"""
Script para arreglar definitivamente el error de sintaxis en core.py
"""

import os
import sys
import subprocess

def fix_core_syntax_final():
    """Arregla definitivamente el error de sintaxis en core.py"""
    print("🔧 ARREGLANDO DEFINITIVAMENTE CORE.PY")
    print("=" * 50)
    
    core_file = "roop/core.py"
    
    if not os.path.exists(core_file):
        print(f"❌ Error: {core_file} no encontrado")
        return False
    
    try:
        # Leer el archivo completo
        with open(core_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar y arreglar el problema específico
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Problema específico: línea 111 con 'for gpu in gpus:' fuera del bloque try
            if 'for gpu in gpus:' in line and i > 0:
                # Verificar si está dentro del bloque try
                in_try = False
                for j in range(max(0, i-20), i):
                    if 'try:' in lines[j]:
                        in_try = True
                    elif 'except' in lines[j] or 'finally' in lines[j]:
                        in_try = False
                
                if not in_try:
                    # Está fuera del bloque try, agregar indentación correcta
                    fixed_lines.append('                ' + line.strip())
                    print(f"🔧 Arreglado: línea {i+1} - agregada indentación correcta")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
            
            i += 1
        
        # Escribir archivo corregido
        with open(core_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print("✅ Archivo core.py corregido definitivamente")
        return True
        
    except Exception as e:
        print(f"❌ Error arreglando core.py: {e}")
        return False

def test_core_syntax():
    """Prueba que core.py esté completamente arreglado"""
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
    print("✅ Error de sintaxis completamente arreglado")
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
    print("🚀 ARREGLANDO DEFINITIVAMENTE CORE.PY")
    print("=" * 60)
    
    # Paso 1: Arreglar archivo
    if not fix_core_syntax_final():
        print("❌ No se pudo arreglar core.py")
        return 1
    
    # Paso 2: Probar sintaxis
    if not test_core_syntax():
        print("❌ core.py aún tiene problemas de sintaxis")
        return 1
    
    print("\n🎉 ¡CORE.PY ARREGLADO DEFINITIVAMENTE!")
    print("=" * 50)
    print("✅ Error de sintaxis completamente corregido")
    print("✅ Archivo listo para usar")
    print("✅ Puedes ejecutar el procesamiento por lotes ahora")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 