#!/usr/bin/env python3
"""
Script para probar específicamente la sintaxis de core.py
"""

import ast
import sys

def test_core_syntax():
    """Prueba la sintaxis de core.py"""
    print("🧪 PROBANDO SINTAXIS DE CORE.PY")
    print("=" * 50)
    
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
        print(f"❌ Texto problemático: {e.text}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    if test_core_syntax():
        print("\n🎉 ¡CORE.PY ESTÁ COMPLETAMENTE ARREGLADO!")
        sys.exit(0)
    else:
        print("\n❌ CORE.PY AÚN TIENE PROBLEMAS")
        sys.exit(1) 