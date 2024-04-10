# while True print("Prueba excepciones")
# File "c:\Users\guille\Documents\Trabajo\Proyecto\excepciones.py", line 1
#     while True print("Prueba excepciones")
#                ^^^^^
# SyntaxError: invalid syntax

# 4 / 0
# ZeroDivisionError: division by zero

# a + 4
# NameError: name 'a' is not defined

# "4" + 4
# TypeError: can only concatenate str (not "int") to str

# Ejemplo 1
while True:
    try:
        x= int(input("Ingrese un número: "))
        break
    except ValueError:
        print("No es un número")    
        
# Ejemplo 2
cad=input("Ingrese una número: ")
try:
    print(10/int(cad))
except ValueError:
    print("No se puedo convertir a entero")
except ZeroDivisionError:
    print("No se puedo dividir por cero")        
else:
    print("Se ha producido otro error")
finally:
    print("Se ejecuta siempre al final")
