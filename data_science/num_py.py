import numpy as np
from sys import getsizeof

array1= np.array([1, 2, 3, 4, 5, 6], dtype='int')
array2 = np.array([[1, 2, 3],[4, 5, 6]],dtype='int', ndmin=2)
array3 = np.array([1,2,3], dtype='complex')
array4 = np.zeros(10)
array5 = np.ones((5,2))

# Tipos de datos
a = np.array([1, 2, 3, 4], dtype=np.int_)
bool_array = np.array([[1,0,0,1],[0,1,1,0]], dtype=np.bool_)
# [[ True False False  True]
#  [False  True  True False]]
char_array = np.array(['a', 'b', 'c', 'd'], dtype=np.chararray)
a = np.array([1,2,3,4], dtype=np.int8)
b = np.array([1,2,3,4], dtype=np.int64)

n = np.nan
# --------------------------------------------------------------------
# Indexado y recorrido de arrays
a = np.arange(10)
a[4]
a.shape

# Slicing
a[0:9]
a[0:9:2]
a[-2:10]
a[0:10]

a = np.array([[1,2,3],[3,4,5]],dtype=np.int8)
a[0][0] = a[0,0]
a[1,0:2] # -> [3,4]

m = np.matrix(a)
m[0,1] + m[1,0] # = 5

# Indexado booleano
a = np.arange(10)
a[a > 4]
a[a % 2 == 0]

# Recorrido array
for v in a:
    print(v)

for valor in np.nditer(a):
    print(valor)

# Máscaras para crear una matriz con elementos ocultos(enmascarados)
x = np.array([1, 2, 3, -1, 4])
mask_array = ma.masked_array(x, mask=[0, 0, 0, 1, 0])  # -> [1 2 3 -- 4]
mask_array.min()  # -> 1
x = ma.masked_array(x, mask=[0, 0, 0, 1, 0])
x.compressed()  # -> [1 2 3 4] sin los datos enmascarados
x.mask = True  # -> enmascara todos los datos
x.mask = False  # -> desenmascara todos los datos
x = np.array([1, 2, 3, -1, 4])
ma.masked_equal(x, 4)  # Enmascara los valores 4
ma.masked_where(
    x > 4, x
)  # Enmascara los valores mayores a 4, añades tu la condción que quieras

# Trabajando con fechas
d = np.datetime64(
    "2000-01-01"
)  # -> Si pones una fecha incorrecta, se genera un ValueError
dh = np.datetime64("2000-01-01T14:30")

### Array Fechas
np.array(["2020-07-01", "2020-08-01", "2020-09-01"])
np.array(["2020-07-01", "2020-08-01", "2020-09-01"], dtype="datetime64")
np.arange("2020-07", "2020-09", dtype="datetime64[W]")  # Para semanas

# Comparaciones
np.datetime64("2020") == np.datetime64("2020-07-01")  # True
np.datetime64("2020-03-14T11") == np.datetime64("2020-03-14T11:00:00.00")  # False

# Operaciones con fechas
np.timedelta64(4, "D")
np.timedelta64(10, "h")
a = np.timedelta64(8, "D")
np.timedelta64(a, "W")
# -> numpy.timedelta64(1,'W')
np.datetime64("2020-08-01") - np.datetime64("2020-07-01")  # -> numpy.datetime64(31,'D')
np.timedelta64(1, "W") + np.timedelta64(4, "D")  # -> numpy.timedelta64(11,'D')
np.busday_offset("2024-04-11", 2)
np.busday_count(np.datetime64("2024-04-11"), np.datetime64("2024-06-18"))

# Constantes
np.inf > 100000000000000  # True
np.inf + np.inf  # Output = inf
np.NINF # Output = -inf
np.inf + np.NINF  # Output = nan
np.nan # Output = nan
np.NZERO  # Output = -0.0
np.PZERO  # Output = 0.0
np.NZERO + np.PZERO  # Output = 0.0
np.pi # Output = 3.14159.....

 
def main():
    # print(array1)
    # print(array2)
    # print(array3)
    # print(array4)
    # print(array5)
    print(bool_array)
    print(True*True) # 1
    print(True*False) # 0
    print("A",getsizeof(a))
    print("B",getsizeof(b))
    print(n)
    print(np.isnan(n))
        
if __name__ == '__main__':
    main()
