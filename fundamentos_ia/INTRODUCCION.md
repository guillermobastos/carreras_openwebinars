# Introducción a la Inteligencia Artificial

---
1. Visión Artificial y Sistemas de Recomendación
Caso de Netflix y las diferentes portadas dependiendo
de las preferencias del usuario

2. Inteligencias Artificial para las Finanzas

3. Inteligencias Artificial en el Comercio
Predicción de demanda
Marketing y publicidad

4. Inteligencias Artificial en las Plataformas Digitales
Sugerencias de contenido 
Personalización de la publicidad

5. Inteligencias Artificial para Ciberseguridad
Detección de spam
Prevención de ataques informáticos

---
## Factores

1. Aumento del interés
2. Big Data
3. Mejoras en la computación
4. Mejoras significativas en la algoritmia

---
## Tipos

1. IA especializada -> tareas muy particulares y con un propósito concreto
2. IA general -> capacidad de comprender, aprender, razonar y realizar tareas de una manera equivalente a la humana
3. IA débil -> para realizar tareas específicas pero sin una "conciencia" (Siri)
4. IA fuerte -> capaces de emular la inteligencia humana y poseen conciencia (especulativo)

---
## Ética

### Tipos de sesgos
* En los Datos
* Sociales

### IA como ayuda a la investigación y al conocimiento

Salud y Medicina
Conducción autónoma
Mejoras en educación


## Control de excepciones en Python

Para el control de las diversas excepciones usaremos las cláusulas:
```python
try:
    data = []
    data.append(5)
except ErrorQueQueramos: 
```
Tambien he visto la forma de importar clases como por ejemplo:

```python
import math
from random import randint
```	

Saber diferenciar entre parámetros reales e informales
```python
def CalcularMaximo(num1, num2): # Parámetros formales
    if num1 > num2:
        return num1
    else:
        return num2

numero1 = int(input("Ingrese el primer número: "))
numero2 = int(input("Ingrese el segundo número: "))
num_maximo = CalcularMaximo(numero1, numero2) # Parámetros reales
print("El número máximo es: {}".format(num_maximo))
```

Funciones recursivas para llamarse a sí mismas:
```python
def factorial(numero):
    if numero == 0 or numero == 1:
        return 1
    else:
        return numero * factorial(numero - 1)
```


        

