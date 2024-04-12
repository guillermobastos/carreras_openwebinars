import numpy as np
# Ejercicios II
# Ejercicio 1
print("Ejercicio 1")
m = np.linspace(start=0, stop=100, num=25,endpoint=False).reshape(5,5)
print(m)
print(np.diag(m)) # Diagonal
print(np.diag(m,k=1)) # Diagonal superior
print()

# ----------------------------------------------------------------
# Ejercicio 2
print("Ejercicio 2")
m2 = np.repeat(a= 5,repeats=20).reshape(4,5)
print(m2)
print(np.concatenate((m,m2),axis=0))
print()

# ----------------------------------------------------------------
# Ejercicio 3
print("Ejercicio 3")
a = np.array(['     este es un curso de openwebinars    '])
a1 = np.char.strip(a)
a2 = np.char.split(a1,sep=' ')
a3 = np.char.capitalize(a2[0])
print(a3)
print()

# ----------------------------------------------------------------
# Ejercicio 4
print("Ejercicio 4")
a = np.array([[3,2,1],[4,1,3],[2,0,1]])
b = np.array([[2,0,3],[1,1,1],[0,2,4]])
axb = np.dot(a,b)
bxa = np.dot(b,a)
# axb != bxa
print(a*2)
# np.vdot(a,b) == np.vdot(b,a)
print()

# ----------------------------------------------------------------
# Ejercicio 5
print("Ejercicio 5")
a3 = np.linalg.matrix_power(a,3)
ad = np.linalg.det(a)
print(ad)
y = np.array([3,2,5])
print(np.linalg.solve(a,y))


# ----------------------------------------------------------------

# Ejercicio III
# ----------------------------------------------------------------
# Ejercicio 1
print("Ejercicio 1")
a = np.array([2, 5, 4, 2, 49, 34, 59, 21, 45, 6, 105])
print(np.any(a % 7 == 0))
res = a[np.logical_or(a % 7 == 0 , a < 40)]
print(res)

# ----------------------------------------------------------------
# Ejercicio 2
print("Ejercicio 2")
print(np.prod(res))
m = res.reshape(3, 3)
print(m)

# ----------------------------------------------------------------
# Ejercicio 3
print("Ejercicio 3")
a = np.array([0,3,5])
d = np.divide(m,a)
np.max(d)
