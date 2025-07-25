#!/usr/bin/env python3
"""
Script para verificar la sintaxis de core.py
"""

import ast
import sys

def check_syntax():
    """Verifica la sintaxis del archivo core.py"""
    print("🔍 VERIFICANDO SINTAXIS DE CORE.PY")
    print("=" * 50)
    
    try:
        with open('roop/core.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar sintaxis
        ast.parse(content)
        print("✅ Sintaxis correcta")
        return True
        
    except SyntaxError as e:
        print(f"❌ Error de sintaxis en línea {e.lineno}: {e.msg}")
        print(f"Contexto: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def fix_core_py():
    """Arregla el archivo core.py si tiene problemas"""
    print("🔧 ARREGLANDO CORE.PY")
    print("=" * 50)
    
    try:
        with open('roop/core.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar y arreglar problemas comunes
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines, 1):
            # Verificar indentación incorrecta
            if line.strip().startswith('for gpu in gpus:') and i > 110:
                # Asegurar que esté dentro del bloque try
                if not any(lines[j].strip().startswith('try:') for j in range(max(0, i-20), i)):
                    print(f"⚠️ Línea {i}: 'for gpu in gpus:' fuera del bloque try")
                    continue
            
            fixed_lines.append(line)
        
        # Escribir archivo corregido
        with open('roop/core.py', 'w', encoding='utf-8') as f:
            f.write('\n'.join(fixed_lines))
        
        print("✅ Archivo corregido")
        return True
        
    except Exception as e:
        print(f"❌ Error arreglando archivo: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 VERIFICANDO Y ARREGLANDO CORE.PY")
    print("=" * 60)
    
    # Verificar sintaxis
    if not check_syntax():
        print("🔧 Intentando arreglar...")
        if fix_core_py():
            if check_syntax():
                print("✅ Archivo arreglado correctamente")
                return True
            else:
                print("❌ No se pudo arreglar")
                return False
        else:
            print("❌ Error arreglando archivo")
            return False
    else:
        print("✅ Archivo está bien")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 