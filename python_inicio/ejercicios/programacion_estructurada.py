import math
from random import randint

def CalcularMaximo(num1, num2): # Parámetros formales
    if num1 > num2:
        return num1
    else:
        return num2

numero1 = int(input("Ingrese el primer número: "))
numero2 = int(input("Ingrese el segundo número: "))
num_maximo = CalcularMaximo(numero1, numero2) # Parámetros reales
print("El número máximo es: {}".format(num_maximo))

# Paso de parámetro por valor o por referencia
#----------------------------------------------------------------
# Objeto inmutable
# def f(a):
#     a = 5
    
# a = 1
# f(a)
# print(a) -> 1

#----------------------------------------------------------------
# Objeto mutable
# def f(lista):
#     lista.append(5)
    
# l = [1,2]
# f(l)
# print(l) -> [1,2,5]

#----------------------------------------------------------------
# LLamadas a función

def cuadrado(numero):
    return numero * numero

a = cuadrado(2)
# print(cuadrado(3)+1) -> 10
# print(cuadrado(cuadrado(4))) -> 256

#----------------------------------------------------------------
# Función recursiva

def factorial(numero):
    if numero == 0 or numero == 1:
        return 1
    else:
        return numero * factorial(numero - 1)

