import re

def limpiarResolucion(resolucionCompleta):
    resolucion = ""
    addNumber = False
    for char in resolucionCompleta:
        if addNumber:
            resolucion += char
        else:
            if char.isdigit() == True:
                addNumber = True
                resolucion += char
    return resolucion

def limpiarDerechos(derechos):
    derechos = [re.sub(r'[^\w]', ' ', x) for x in derechos]
    derechos = [re.sub(r'(\s)+', ' ', x.strip()) for x in derechos]
    union = list(set().union(derechos, derechos))
    return union