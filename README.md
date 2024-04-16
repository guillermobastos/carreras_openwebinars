## Conceptos clave en Inteligencia Artificial

IA -> es la rama de la Matemática que busca simular en ur ordenador comportamiento inteligente

1. Algoritmos
Conjunto de instrucciiones destinadas a realizar un proceso

2. Big Data
tecnología empleada en torno a la gestión y explotación de cantidades de datos masivos
* Almacenamiento
* Acceso
* Gobierno
* Calidad
Características -> Volumen, Variedad, Velocidad, Veracidad

3. Aprendizaje automático (Machine learning)
Ejemplos: Desbloqueo facial, Predictor ortográfico

4. Aprendizaje profundo (Deep learning)
Este se produce a través de capas que reciben el nombre de neurona

---
## Sistemas basados en reglas

Busca realizar las tareas a partir de un conjunto finito y cerrado de reglas predefinidas

* Composición
Información papra la inferencia
Reglas basadas en conocimiento específico
Solución para la inferencia

Ventajas:
1. Control preciso de la toma de decisiones
2. Permite introducir excepciones para casos particulares
3. No requieren cantidades tan grandes de datos disponibles para aprender

Desventajas:
1. No todos los problemas son codificables en reglas fáciles
2. No son muy eficientes
3. Mantenimiento difícil

### Diagramas de flujo
* Importante plantear todos los posibles escenarios
* Es útil tener una respuesta especial para casos que se salen de la norma
* Consultar con todas las partes de conocimiento implicadas

---
## Sistemas de aprendizaje automático
Aprender -> nos referimos a que dado un objeto, el modelo es capaz de "aprender" a realizar la tarea de la manera más satisfactoria

* Aprendizaje supervisado -> Aquel que parte de dato etiquetado
* Aprendizaje no supervisado -> Aquel que parte de datos sin etiquetar, no poseen una solución prefijada
Son técnicas complementarias

Dos tipos de variables a predecir
1. Cualitativas, categóricas o discretas : calificación
2. cuantitativas, númericas o continuas : regresión

|               | Supervisado 1   | No supervisado 2                   |
|---------------|-----------------|------------------------------------|
| **Categórica**|  Clasificación  | Clústering                         | 
| **Contínua**  |  Regresión      | Análisis de Componentes Principales|


Aprendizaje por Refuerzo (Reinforicement Learning) -> aprende a realizar una serie de acciones en base a la obtención de una recompensa

---
## Aprendizaje Profundo (Deep Learning)
Elevada precisión
Buen funcionamiento
Redes neuronales
    
Selección automática de variables -> capacidad de las redes neuronales para extraer información útil del dato en bruto

Dato estructurado y no estructurado
Sobreajuste (overfitting) -> fenómeno en el mundo del Aprendizje Profundo en el que los modelos no son pcapaces de generalizar sobre datos nuevos
Problema de la explicabilidad


# Herramientas y lenguajes de programación para Inteligencia Artificial

## Lenguajes de programación para IA: Python y R
* Bibliotecas especializadas
* Amplia comunidad
* Software libre
* Compatibilidad de plataformas

---
Biblioteca o módulo -_> paquete de código publicado que contiene una serie de herramientas


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

## Numpy

### Estructura de los arrays

```python
import numpy as np
array1 = np.array([1, 2, 3],dtype='int')
rray2 = np.array([[1, 2, 3],[4, 5, 6]],dtype='int', ndmin=2)
array3 = np.array([1,2,3], dtype='complex')
array4 = np.zeros(10)
array5 = np.ones(10)
array6 = np.arange(0,10)
print(type(array1))
```

- Uso de dtype para especificar el tipo de dato
- ndim para decir el número mínimo de dimensiones
- tipo = complex para el uso de números reales
- np.zeros para rellenar un array de ceros, tantos como queramos o np.ones para la misma función pero con unos
- Rellena un array con un rango de valores

## Tipos de datos

```python
a = np.array([1, 2, 3, 4], dtype=np.int_)
bool_array = np.array([[1,0,0,1],[0,1,1,0]], dtype=np.bool_)
# [[ True False False  True]
#  [False  True  True False]]
char_array = np.array(['a', 'b', 'c', 'd'], dtype=np.chararray)
```

## Tamaños

```python
import numpy as np
from sys import getsizeof

a = np.array([1,2,3,4], dtype=np.int8)
b = np.array([1,2,3,4], dtype=np.int64)
dt = np.dtype('int32')
n = np.nan

# np.isnan(n) -> True
# dt.type is np.int32 -> True
# a.dtype is np.int8 -> False
# a.dtype.type is np.int8 -> True

print("A",getsizeof(a))
print("B",getsizeof(b))
```

Podemos incluso añadir el valor nan a un array que creemos:

```python
array = np.array([1,2,3,4,np.nan])
```

Podemos también usar una clase específica de NumPy para manejar matrices bidimensionales como es np.matrix:

```python
m = np.matrix('1 2 3; 4 5 6', dtype=np.float16)
a = np.array([[1,2,3],[4,5,6]],dtype=np.float16)
Diferenciamos la forma de crear las matrices bidimensionales
```

## Máscaras para crear una matriz con elementos ocultos(enmascarados)
Algunos ejemplos:
```python
x = np.array([1, 2, 3, -1, 4])
mask_array = ma.masked_array(x, mask=[0, 0, 0, 1, 0])  # -> [1 2 3 -- 4]
x = ma.masked_array(x, mask=[0, 0, 0, 1, 0])
x.compressed()  # -> [1 2 3 4] sin los datos enmascarados
x.mask = True  # -> enmascara todos los datos
x.mask = False  # -> desenmascara todos los datos
x = np.array([1, 2, 3, -1, 4])
ma.masked_equal(x, 4)  # Enmascara los valores 4
ma.masked_where(
    x > 4, x
)  # Enmascara los valores mayores a 4, añades tu la condción que quieras
```

## Trabajando con fechas 
```python
d = np.datetime64(
    "2000-01-01"
)  # -> Si pones una fecha incorrecta, se genera un ValueError
dh = np.datetime64("2000-01-01T14:30")
```

Array de fechas:
```python
np.array(["2020-07-01", "2020-08-01", "2020-09-01"])
np.array(["2020-07-01", "2020-08-01", "2020-09-01"], dtype="datetime64")
np.arange("2020-07", "2020-09", dtype="datetime64[W]")  # Para semanas
```

Comparaciones:
```python
np.datetime64("2020") == np.datetime64("2020-07-01")  # True
np.datetime64("2020-03-14T11") == np.datetime64("2020-03-14T11:00:00.00")  # False
```

Operaciones con fechas:
```python
np.timedelta64(4, "D")
np.timedelta64(10, "h")
```

## Operaciones con cadenas
Hemos visto las funciones de añadir .add(), multiplicar, multiply(), el dtype dependiendo del tamaño de la string.
Por lo general son funciones ya vistas en otros lenguajes pero en este caso usando la librería de numpy y en concreto el submódulo
np.char 
Un par de ejemplos serían:

```python
a = np.array(['A','B','C'],dtype = np.str_)
b = np.array(['D','E','F'],dtype = np.str_)
np.char.add(a,b)
# Output -> array(['AD', 'BE', 'CF'], dtype='<U2')
nombre = np.array(['Mi nombre es Guillermo Bastos'])
np.char.split(nombre, sep=' ')
# Output -> array([list(['Mi', 'nombre', 'es', 'Guillermo', 'Bastos'])], dtype=object)
```

## Álgebra lineal 
En este caso hemos usados funciones propias de la librería de numpy nuevamente pero para el cálculo de diferentes funciones en matrices:

```python
a = np.array([[3,2],[1,3]])
b = np.array([[2,2],[1,2]])
np.dot(a,b) # Para calcular el producto
np.inner(a,b) # Para calcular el producto interno en este caso
```
---
## Método supervisado con Scikit-learn

* Como entrenar un modelo
* Los pasos previos al entrenamiento
* Los modelos que tenemos disponibles
* La evaluación de los resultados
* Como optimizar los modelos

1. OBJETIVOS
* Ejecutar el flujo completo del entrenamiento de un modelo
* Explotar el ecosistema del scikit-learn para este proceso mejor

### Introducción

1. Razones de uso
* Librería que establece un framework para crear flujos de la creación de algoritmos de Machine Learning
* Unifica un campo muy diverso y abstrae las dificultades de cada algoritmo

### Sintaxis básico de scikit-learn
El entrenamiento de un modelo supervisado siempre sigue el mismo patrón:

my_model = SomeModel()
my_model.fit(X_train, y_train)
my_model.predict(X_test)

### Preparar los datos
* Siempre necesitamos:
X: datos de los "features" de n filas y m columnas, todos datos numéricos
y: datos del "target" de n filas y 1 columnam, todo numérico

### Antes de entrenar hay algunos pasos previos
* Todas las transformaciones que podemos hacer con sklearn se pueden hacer en Pandas
* Es común hacer la parte exploratoria en Pandas y explotar sklearn para construir el flujo completo de entrenamiento

### Dividir en Train y Test
* Ya que no queremos que influyan los datos "no vistos" en el diseño de los pasos de preparación

### Pipeline
* Secuencia de pasos que se utilizan para procesar y transformar datos, así como para entrenar y evaluar modelos de aprendizaje automático de manera sistemática y eficiente
* El pipeline de sklearn es una herramienta fundamental para mejorar el proceso de preparación.

---

### Qué es un modelo en Sklearn?
* Es la implementación del algoritmo que aprende a predecir nuetro target basándose en los features
* Cada modelo tienes sus propios hiperparámetros y manera de funcionar, pero el input y output es casi igual

### Generalized Linear Models (Modelos Lineales Generalizados)
* Los más comunes son la regresión lineal y la regresión lógica
* Útiles por su interpretabilidad

### SVM Support Vector Machines (Máquinas de Vectores de Soporte)
* Otros modelos muy clásicos que buscan separar los datos, con implementación para clasificación y regresión
* Su objetivo principal es encontrar un hiperplano en un espacio de características que separe de manera óptima las clases de datos.
* Lo más importante es optimizar el kernel y regularización

```python
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.preproccesing import StandardScaler
from sklearn.pipline import Pipeline

...

clas_svm_kernel_model = Pipeline(
    [
        ('scaler',StandardScaler()),
        ('svm',SVC(kernel='linear'))
    ]
)

clas_svm_kernel_model.fit(X_variables, train[y_variable_reg])
test['predictions_svm_kernel_clas'] = clas_svm_kernel_model.predict(test[X_variables])
```

### Modelos basados en árboles
* Populares por su flexibilidad antes muchos problemas
* La base de todo es el "CART" pero nuevos algoritmos aprovechan el "ensembling" para mejorar los resultados

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier

X_variables = ['followers', 'retweet', 'video','tweets','likes','following','media','verified','day','hour']

reg_tree_model = DecisionTreeRegressor(random_state=0)
reg_tree_model.fit(train[X_variables], train[y_variable_reg])
test['predictions_tree_reg'] = reg_tree_model.predict(test[X_variables])

clas_tree_model = DecisionTreeClassifier(random_state=0)
clas_tree_model.fit(train[X_variables], train[y_variable_reg])
test['predictions_tree_clas'] = clas_tree_model.predict(test[X_variables])

test[[y_variable_reg, 'predictions_lin_reg','predictions_tree_reg']]

```
