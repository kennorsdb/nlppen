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