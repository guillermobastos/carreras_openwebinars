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
