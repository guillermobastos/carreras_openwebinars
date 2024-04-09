diccionario = {"one": 1, "two": 2, "three": 3}
# print(type(diccionario))
# print(diccionario["two"])

dict1 = {}
dict1["nombre"] = "Jose"
dict1["edad"] = 20
# print(dict1)

# len(dict1)
del diccionario["one"]
# print(diccionario)

# "nombre" in dict1
# dict2 = dict1.copy() -> dos diccionarios distintos pero con la copia del primero
# para que no hagan referencia al mismo lugar de memoria, esto estaría mal -> dict2 = dict1

# Métodos
# dict1.clear() -> limpiar diccionario
dict1 = {"one": 1, "two": 2, "three": 3}
dict2 = {"four": 4, "five": 5}
# dict1.update(dict2) -> suma
# dict1.get("one") -> no salta error si se equivoca la clave
# dict1.pop("one") --> borra y devuelve y si no existe la clave salta un error

# for clave in dict1.keys():
#     print(clave)

# for valor in dict1.values():
#     print(valor)

# for clave, valor in dict1.items():
#     print(clave,valor)

# Ejercicio

dict = {}
cadena = input("Dame una cadena: ")
for caracter in cadena:
    if caracter in dict:
        dict[caracter] += 1
    else:
        dict[caracter] = 1

for campo, valor in dict.items():
    print(campo, "-->", valor)

# ---------------------------------------------------------------

precios = {"manzana": 2, "naranja": 1.5, "plátano": 4, "piña": 3}
while True:
    fruta = input("Dime la fruta que has vendido: ")
    if fruta.lower() not in precios:
        print("La fruta no existe")
    else:
        cantidad = int(input("Dime la cantidad de frutas que has vendido: "))
        print("El precio es de %f" % (cantidad+precios[fruta]))
    opcion = input("Quieres vender otra fruta? (s/n): ")
    while opcion.lower() != "s" and  opcion.lower() != "n":
        opcion = input("Quieres vender otra fruta? (s/n): ")
    if opcion.lower() != "n":
        break
