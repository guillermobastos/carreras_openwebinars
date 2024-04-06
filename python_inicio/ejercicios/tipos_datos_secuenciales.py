# Cadenas
cadena = "informática"
print("a" in cadena) 
print(cadena[0])
print(cadena[2:5])
print(cadena[2:7:2]) #desde la posición 2 a la 6 de dos en dos
print(cadena[2:]) # desde la posición 2 al final
print(cadena[:5]) # desde el principio a la posición 4
print(cadena[::-1]) # cadena invertida 
print(cadena.upper())
# Objetos de tipo cadena son Inmutables

cadena.capitalize() # no cambia la cadena, el método devuelve la cadena con dicha modificación
cadena.lower() # minúscula

# Métodos de búsqueda
cadena = "bienvenido a mi aplicación"
cadena.count("a")
cadena.count("a",16) #desde la posición 16
cadena.count("a",10,16) #desde la posición 10 a la 15
cadena.find("mi") # primera posición donde encuentra dicha cadena y si no la encuentra devuelve -1

# Métodos de comprobación
cadena.startswith("b") # boolena
cadena.startswith("bien",13) # desde la posiciñon 13 boolena
cadena.endswith("ción") # terminación

# Ejercicio 
cont = 0
posicion = 0
cad = input("Introduce una serie de palabras separadas por un espacio: ")
cad = cad.strip()

posicion = cad.find(" ", posicion)
while posicion != -1:
    cont += 1
    while cad[posicion +1] == " ":
        posicion += 1
    posicion = cad.find(" ", posicion+1)
print("La frase tiene",cont+1,"palabras")
