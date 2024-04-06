# Tipos de datos 

# Tipos númericos float, int
import math
numero_float = -5.0
numero_int = 5
# print(math.sqrt(9))
# print(divmod(7,3))

# -------------------------------------------------------------------
# Tipos númericos booleanos, True y False
num = 7
# print(num == 7) -> True
num = 11
# print(num >= 1 and num <= 10) -> False

# -------------------------------------------------------------------
# Entrada y Salida Estándar :  print || input

# Introducción a la cadena de caracteres, comparación y operaciones
# cad1 = 'prueba'
# cad2 = "prueba"
# cad3 = '''Hola
# como estás?'''
# "prueba" == "prueba" -> True

# -------------------------------------------------------------------
# Estructuras repetitivas

# Uso del While
# secreto = "1234"
# clave = input("Dime la clave: ")
# while clave != secreto:
#     print("Contraseña erronea") 
#     clave = input("Dime la clave: ")
# print("Contraseña correcta")

# Uso del For
# for var in range(1,20):
#     print(var,"",end="")

# for var in range(10,0,-1):
#     print(var,"",end="")

# -------------------------------------------------------------------
# Ejercicio tabla de multiplicar de un número:

# num = int(input("Dime el número del cual desea ver la tabla de multiplicar: "))
# if num > 0:
#     for var in range(1,11):
#         print(num,"x",var,"= ",num*var)
#         print("%2d x %2d = %2d" % (num,var,num*var))

# -------------------------------------------------------------------
# Algoritmo que muestre la tabla de multiplicar de los números 1,2,3,4 y 5

# num = 1
# while num < 6:
#     for var in range(1,11):
#         print("%2d x %2d = %2d" % (num,var,num*var))
#     num+=1
#     print()

# -------------------------------------------------------------------
