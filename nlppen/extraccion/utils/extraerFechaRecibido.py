import re 
import pandas as pd
from .Txt2Numbers import Txt2Numbers

class ExtraerFecha():
    ''' 
    Extraer fecha de recibido de la sentencia.  
    '''
    def __init__(self, nlp):
        """
        Parametros:
            nlp: nlp - Spacy 
                Procesador de nlp de spacy, se utiliza para el proceso de extraccion
        """
        self.nlp = nlp

    def asignarHora(self, hora, agregarHora, number):
        if hora == "" and agregarHora and number != None:
            hora = str(number)
        return hora

    def asignarMinutos(self, minutos, number):
        if minutos == "" and number != None:
            minutos = str(number)
        return minutos

    def asignarDia(self,dia, number):
        if dia == "" and  number != None:
            dia = str(number)
        return dia


    def txt2Date(self, txt, sentenceDate):
        """
        Convierte un conjunto de oraciones de texto a una fecha.

        Retorna:
             Python DataTime

        Parametros:
            txt: String
                Conjunto de oraciones a ser procesadas para la extraccion de la fecha
            
            sentenceDate : Pandas Date
                Fecha de extension de la sentencia [fechahora_ext]. Se utiliza cuando se menciona el año en curso o el presente año.
        """

        #Compiilar re
        reHoras = re.compile(r"\.|[Hh][OoÓó][Rr][Aa]([Ss])?")
        reMin = re.compile(r"\.|[Mm][Ii][Nn][Uu][Tt][Oo]([Ss])?")
        rehoraMin = re.compile(r"[0-9][0-9]:[0-9][0-9]")
        reCurso = re.compile(r"(año en curso)|(presente año)|(en curso)")
        meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                            "julio", "agosto", "septiembre", "setiembre",
                            "octubre", "noviembre", "diciembre"]
        #Definicion variables
        fecha = ""
        hora = ""
        minutos = ""
        dia = ""
        mes = ""
        anno = ""
        includeText = False
        nextWordIncluded = False
        agregarMinutos = True
        agregarHora = True
        convertToNumber = Txt2Numbers()

        doc = self.nlp(txt)
        #Eliminar el 1. del inicio del resultando de la sentencia
        (doc, indexNextWord) = self.__cleanDoc(doc)

        for token in doc:
            #print(token.text, token.pos_)
            if includeText:
                #Se esta procesando un numero en letras.
                if token.text == ",":
                    #Generalmente los annos aparecen antes de una, 
                    number = self.__textoAAnno(accumTokenText)
                    if anno == "":
                        anno = str(number)
                        break
                    includeText = False
                else:
                    if reHoras.search(token.text) != None:
                        #Token es horas. Por lo que hay que convertir el numero en texto que tengamos.
                        number = convertToNumber.number(accumTokenText)
                        hora = self.asignarHora(hora, agregarHora, number)
                        try:
                            nextWordHora = doc[token.i + indexNextWord].text.strip().lower()
                            if nextWordHora == "de" or nextWordHora == "del":
                                #Si la siguiente palabra es de o del va iniciar la fecha, ya no hay que incluir minutos.
                                agregarMinutos = False
                        except:
                            pass
                        #Ya no incluir palabras para accumTokenText hasta que aparezca otro numero.
                        includeText = False
                    else:
                        if reMin.search(token.text) != None and agregarMinutos:
                            number = convertToNumber.number(accumTokenText)
                            minutos = self.asignarMinutos(minutos, number)
                            #Ya no incluir palabras hasta que aparezca otro numero
                            includeText = False
                        else:
                            #Aprece la palabra de, del
                            if token.pos_ == "ADP" or token.pos_ == "DET":
                                number = convertToNumber.number(accumTokenText)
                                if number != None:
                                    dia = self.asignarDia(dia, number)
                                else:
                                    number = self.__textoAAnno(accumTokenText)
                                    if anno == "" and number != None:
                                        anno = str(number)
                                        break
                                includeText = False
                            else:
                                #Si no es ninguno de estos casos seguir acumulando palabras (esta formando el numero)
                                accumTokenText += " " + token.text
            else:
                #No se esta procesando un numero con forma de letras entonces puede ser los siguientes casos
                if token.pos_ == 'NOUN':
                    # Caso xx:xx detectado como NOUN por spacy. 
                    if rehoraMin.search(token.text.lower()) != None:
                        splitHoraMin = token.text.split(":")
                        #Revisar si es formato numero xx:xx para hora y minutos
                        if(len(splitHoraMin) == 2):
                            hora = splitHoraMin[0]
                            minutos = splitHoraMin[1]
                            agregarMinutos = False
                else:
                    if token.pos_ == 'NUM' and token.text.lower() != "hrs":
                        #Es un Numero puede ser en forma numerica o texto
                        if nextWordIncluded:
                            #Verifica si la palabra actual fue procesada previamente. Ver casos de, del , abajo
                            nextWordIncluded = False
                            continue
                        accumTokenText = token.text
                        if accumTokenText.isdigit() == False:
                            #No es numerico, puede ser letras o xx:xx
                            if hora == "":
                                splitHoraMin = accumTokenText.split(":")
                                #Revisar si es formato numero xx:xx para hora y minutos
                                if(len(splitHoraMin) == 2):
                                    hora = splitHoraMin[0]
                                    minutos = splitHoraMin[1]
                                    agregarMinutos = False
                                else:
                                     #El formato es de letras, comenzar a agregar
                                    includeText = True
                            else:
                                 #El formato es de letras, comenzar a agregar
                                includeText = True
                        #Es un numero no hay que hacer cambios
                        else:
                            #Es numerico
                            if(len(token.text) > 4):
                                #Filtrar números como el de sentencia 2015712365
                                continue
                            try:
                                if reHoras.search(doc[token.i + indexNextWord].text.strip()) != None:
                                    #Proxima palabra es horas
                                    hora = self.asignarHora(hora, agregarHora, accumTokenText)
                                    try:
                                        nextWordHora = doc[token.i + 1  + indexNextWord].text.strip().lower()
                                        if nextWordHora == "de" or nextWordHora == "del":
                                            agregarMinutos = False
                                    except:
                                        pass
                                else:
                                    if reMin.search(doc[token.i + indexNextWord].text.strip()) != None and agregarMinutos:
                                        minutos = self.asignarMinutos(minutos, accumTokenText)
                                    else:
                                        if dia == "":
                                            dia = accumTokenText
                                        else:
                                            if anno == "":
                                                anno = accumTokenText
                                                break
                            except:
                                pass
                if token.text.lower() == "de" or token.text.lower() == "del" or token.text.lower() == "con":
                    #Palabra previa era hora, entonces no hay que agregar minutos
                    previousToken = doc[token.i + indexNextWord - 2].text
                    if token.text.lower() != "con" and reHoras.search(previousToken.lower()) != None:
                        agregarMinutos = False

                    validatorMes = False
                    #nextWord = doc[token.i+1].text.strip().lower()

                    try:
                        nextWord = doc[token.i + indexNextWord].text.strip().lower()
                        nextWord2 = doc[token.i + 1 + indexNextWord].text.strip().lower()
                        
                        if nextWord2 == "y":
                            continue

                    except:
                        pass

                    #print("Token", token.text)
                    #print("NextWord", doc[token.i-1], doc[token.i], doc[token.i+1])
                    #print("")
                    for month in meses:
                        if month == nextWord:
                            validatorMes = True
                            break
                    if validatorMes:
                        number = convertToNumber.number(nextWord)
                        if mes == "" and number != None:
                            mes = str(number)
                        nextWordIncluded = True
                        agregarHora = False
                    else:
                        #Validar si viene en formato texto
                        if nextWord != "mil":
                            try:
                                if doc[token.i + indexNextWord + 1].text.strip() != "mil":
                                    #Puede ser parte de la fecha y hora, o puede ser un anno.
                                    number = convertToNumber.number(nextWord)
                                    if number != None:
                                        nextWordIncluded = True
                                        if hora == "" and agregarHora:
                                            hora = str(number)
                                        else:
                                            if minutos == "" and agregarMinutos and agregarHora:
                                                minutos = str(number)
                                            else:
                                                if dia == "":
                                                    dia = str(number)
                                                else:
                                                    if anno == "":
                                                        anno = str(number)
                                                        break
                                else:
                                    try:
                                        #Verificar si es un numero, no letras
                                        number = int(nextWord)
                                        if number != None:
                                            nextWordIncluded = True
                                            if hora == "" and agregarHora:
                                                hora = str(number)
                                            else:
                                                if minutos == "" and agregarMinutos and agregarHora:
                                                    minutos = str(number)
                                                else:
                                                    if dia == "":
                                                        dia = str(number)
                                                    else:
                                                        if anno == "":
                                                            anno = str(number)
                                                            break
                                    except:
                                        pass
                            except:
                                pass
                else:
                    if token.text.lower() == "el":

                        nextWord = None
                        try:
                            nextWord = doc[token.i + indexNextWord].text.strip().lower()
                        except:
                            pass

                        if nextWord is not None and nextWord != "mil":
                            nextWord2 = None
                            try:
                                nextWord2 = doc[token.i +indexNextWord + 1].text.strip()
                            except:
                                pass
                            if nextWord2 != "mil" and nextWord2 != None:
                                #Puede ser parte de la fecha y hora, o puede ser un anno.
                                number = convertToNumber.number(nextWord)
                                if number != None:
                                    nextWordIncluded = True
                                    if dia == "":
                                        dia = str(number)
                                    else:
                                        if anno == "":
                                            anno = str(number)
                                            break
                                else:
                                    try:
                                        #Verificar si es un numero, no letras
                                        number = int(nextWord)
                                        if number != None:
                                            nextWordIncluded = True
                                            if dia == "":
                                                dia = str(number)
                                            else:
                                                if anno == "":
                                                    anno = str(number)
                                                    break
                                    except:
                                        pass
                    else:
                        if token.pos_ == "ADJ":
                            number = convertToNumber.number(token.text)
                            if number != None:
                                if dia == "" or dia == None:
                                    dia = str(number)
                        else:
                            validatorMes = False
                            for month in meses:
                                if month == token.text:
                                    validatorMes = True
                                    break
                            if validatorMes:
                                number = convertToNumber.number(token.text)
                                if mes == "" and number != None:
                                    mes = str(number)
                                agregarHora = False
                        if nextWordIncluded:
                                nextWordIncluded = False

            if anno == "" or len(anno) != 4 :
                if reCurso.search(txt.lower()) != None and sentenceDate is not None:
                    anno = str(sentenceDate.year)


        if dia != "" and mes != "" and anno != "":
            if hora != "":
                if minutos != "":
                    fecha = dia+"/"+mes+"/"+anno+" "+hora+":"+minutos
                else:
                    fecha = dia+"/"+mes+"/"+anno+" "+hora+":"+"00"
            else:
                fecha = dia+"/"+mes+"/"+anno +" "+ "00" +":"+"00"

            try:
                return pd.Timestamp(pd.to_datetime(fecha, format='%d/%m/%Y %H:%M')).to_pydatetime()
            except:
                return None
        else:
            return None

    
    def __textoAAnno(self, txt):
        txt = re.sub(',', '', txt.lower())
        convertToNumber = Txt2Numbers()
        expre = re.compile (r'dos mil', re.M)
        dosMil = expre.search(txt.lower())
        number = None
        if dosMil != None:
            number = convertToNumber.number('dos mil')
            number = number * 100
            txt = re.sub('dos mil', '', txt.lower())
            number2 = convertToNumber.number(txt.strip())
            if number2 != None:
                number = number + number2
        else:
            expre = re.compile (r'mil novecientos', re.M)
            milNovecientos = expre.search(txt.lower())
            if milNovecientos != None:
                number = convertToNumber.number('mil novecientos')
                number = number * 100
                txt = re.sub('mil novecientos', '', txt.lower())
                number2 = convertToNumber.number(txt.strip())
                if number2 != None:
                    number = number + number2
        return number

    def __cleanDoc(self, doc):
        index = 1
        #Eliminar el 1. cn el que comienza el resultando.
        if(doc[0].pos_ == 'NUM' or doc[0].pos_ == 'SYM'):
            doc = doc[1:]
            index = 0
        return (doc, index)
