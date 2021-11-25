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

def splitResolucion(numRes, expediente, sentenciasCSV, anno=None):
    idSentencia = None
    numDoc      = None
    expedienteNuevo      = None
    if numRes is not None:
        
        year = None
        numResolucionSplitted = None
        if numRes.isdigit():
            #Es numero unicamente, debo buscar donde está el año
            regExp = "((?:19[0-9][0-9])|(?:20[0-9][0-9]))"
            exp = re.compile(regExp,  re.X | re.M | re.I)
            splitNumRes = exp.split(numRes, 1)
            if(len(splitNumRes) > 2):
                if exp.match(splitNumRes[1]):
                    year = splitNumRes[1]
                    numResolucionSplitted = splitNumRes[2]
                else:
                    numResolucionSplitted = splitNumRes[1]
                    year = splitNumRes[2]
            else:
                numResolucionSplitted = splitNumRes[0]
                year = None
        else:
            regExp = "-|—"
            exp = re.compile(regExp,  re.X | re.M | re.I)
            splitNumRes = exp.split(numRes)

            #Año a la izquierda 97
            regExp = "[0-9][0-9]$"
            exp = re.compile(regExp,  re.X | re.M | re.I)
            if exp.match(splitNumRes[0]):
                year = splitNumRes[0]
                if year[0] == '0' or year[0] == '1' or year[0] == '2':
                    year = str(2000+int(year))
                else:
                    year = str(1900+int(year))
                
                numResolucionSplitted = splitNumRes[1]
            else:
                #Año a la derecha 97
                regExp = "[0-9][0-9]$"
                exp = re.compile(regExp,  re.X | re.M | re.I)
                if exp.match(splitNumRes[1]):
                    year = splitNumRes[1]
                    if year[0] == '0' or year[0] == '1' or year[0] == '2':
                        year = str(2000+int(year))
                    else:
                        year = str(1900+int(year))
                    
                    numResolucionSplitted = splitNumRes[0]
                else:
                    regExp = "((?:19[0-9][0-9])|(?:20[0-9][0-9]))"
                    exp = re.compile(regExp,  re.X | re.M | re.I)
                    if(len(splitNumRes) > 1):
                        if exp.match(splitNumRes[0]):
                            #Año a la izquierda 1998
                            year = splitNumRes[0]
                            numResolucionSplitted = splitNumRes[1]
                        else:
                            #Año a la derecha 1998
                            year = splitNumRes[1]
                            numResolucionSplitted = splitNumRes[0]

        try:
            numResolucionSplitted = int(numResolucionSplitted)
            year = int(year)
        except:
            pass
        #print("Num resolucion", numRes)
        #print("Resolucion", numResolucionSplitted, "Año", year) 
        #print("Anno", year)
        #print("Número de resolucion spliteado", numResolucionSplitted)
        if year is None and anno is not None:
            year = anno.year
        sentenciaFiltered = sentenciasCSV[sentenciasCSV.numeroDocumento == numResolucionSplitted]
        sentenciaFiltered = sentenciaFiltered[sentenciaFiltered.anno == year]
        lenFilter = len(sentenciaFiltered)
        if lenFilter == 1:
            
            expedienteNuevo = str(sentenciaFiltered.iloc[0, 2])
            idSentencia = str(sentenciaFiltered.iloc[0, 3])
            numDoc = str(sentenciaFiltered.iloc[0, 4])
            #print(idSentencia, numDoc)
            #print("Find it")
        else:
            if lenFilter > 1:
                cantidadAnterior = lenFilter
                sentenciaFiltered = sentenciaFiltered[sentenciaFiltered.expediente == expediente]
                lenFilter = len(sentenciaFiltered)
                if lenFilter == 1:
                    expedienteNuevo = str(sentenciaFiltered.iloc[0, 2])
                    idSentencia = str(sentenciaFiltered.iloc[0, 3])
                    numDoc = str(sentenciaFiltered.iloc[0, 4])
                else:
                    #Está repetido no importa cual agarre
                    if cantidadAnterior == lenFilter:
                        expedienteNuevo = str(sentenciaFiltered.iloc[0, 2])
                        idSentencia = str(sentenciaFiltered.iloc[0, 3])
                        numDoc = str(sentenciaFiltered.iloc[0, 4])
    return idSentencia, numDoc, expedienteNuevo 
