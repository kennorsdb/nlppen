import re 
import pandas as pd
from .Txt2Numbers import Txt2Numbers
from .extraerFechaRecibido import ExtraerFecha
from .misc import splitResolucion
class ExtraerFechaSentencia():
    def __init__(self, nlp=None):
        self.nlp = nlp
        self.mesesIndice = {"enero": 1,
        "febrero": 2,
        "marzo": 3,
        "abril": 4,
        "mayo": 5,
        "junio": 6,
        "julio": 7,
        "agosto": 8,
        "septiembre": 9,
        "setiembre": 9,
        "octubre": 10,
        "noviembre": 11,
        "diciembre": 12}
    
    def __convertToNumber(self, hora, minutos, dia, anno, mes):
        """
        Convierte cada parametro de texto en número. 

        Retorna:
             Retorna una tupla de los parametros de entrada en formato de numero

        Parametros:

            hora: String
                Representa la hora de la sentencia
            minutos: String
                Representa los minutos de la sentencia
            dia: String
                Representa el dia de la sentencia
            anno: String
                Representa el anno de la sentencia
            mes: String
                Representa el mes de la sentencia
        """
        convertToNumber = Txt2Numbers()
        horaNum = None
        minNum = None
        diaNum = None
        mesNum = None
        annoNum = None

        if hora.isdigit() == False:
            if hora != '':
                horaNum = convertToNumber.number(hora)
        else:
            horaNum = hora
            
        if minutos.isdigit() == False:
            if minutos != '':
                minNum = convertToNumber.number(minutos)
        else:
            minNum = minutos
            
        if dia.isdigit() == False:
            if dia != '':
                diaNum = convertToNumber.number(dia)
        else:
            diaNum = dia
            
        if anno.strip().isdigit() == False:
            if anno != '':
                ext = ExtraerFecha()
                annoNum = ext.textoAAnno(anno)
        else:
            annoNum = anno
            
        if mes.strip().isdigit() == False:
            if mes != '':
                if mes in self.mesesIndice:
                    mesNum = str(self.mesesIndice[mes])
                else:
                    mesNum = None
        else:
            mesNum = mes
        
        return (horaNum, minNum, diaNum, annoNum, mesNum)

    def __extraerID(self, citas , sentenciasCSV):
        citasSentenciasID  = {}
        for cita in citas:
            sentenciaId, numVoto, _ = splitResolucion(cita, None, sentenciasCSV, citas[cita])
            if sentenciaId is not None:
                citasSentenciasID[sentenciaId] = numVoto
        return citasSentenciasID

    def extraerCitas(self, txt, sentenciasCSV):
        """
        Extrae las citas a sentencias, incluyendo las fechas de creación de las citas citadas. 

        Retorna:
             Un diccionario donde las llaves corresponden a las sentencias, y los valores son Timestamp

        Parametros:

            txt: String
                Representa el texto de la sentencia completa.
        """
        doc = self.nlp(txt)
        #Expresiones regulares para capturar cada grupo.
        meses = "(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|septiembre|setiembre|octubre|noviembre|diciembre)"
        sentenciaRe = "(?:(?:[Ss]entencia\s*))(?:estructural)?\s*"
        numeroRe    = "(?:(?:n[uú]mero)|(?:[nN][ºo°][\.|,]?))"
        sentenciaCitadaRe ="\s*(?P<sentencia_citada>[\d]+\s?[\-\—]?[a-z]?\s?[\-\—]?[\d]+)[\.|,]?"
        horasRe = "(?:(?:\s?del?\s?las?\s?(?P<hora>[\w|\d]+)?\s?(?:(?:\:|horas?|hrs.?))?\s?)?"
        minutosRe = "(?:(?P<minutos>[\w|\s|\d]+)\s+(?:(?:horas?|hrs.?|minutos?)?))?"
        diaRe = "\s?del?\s?(?:d[íi]a)?(?P<dia>[\w|\s|\d]+)\s+de"
        mesRe = "\s+(?P<mes>"+meses+")+\s+"
        a = "(?:mil\s?novecientos\s?[\w]+\s?y?\s?[\w]+)|(?:dos\s?mil\s?[\w]+)"
        annoRe = "(?:del?)?\s?(?:año)?\s?(?P<anno>(?:(?:[\d][\d][\d][\d])|(?:"+a+")))?)?"
        
        #Union de todas las expresiones regulares.
        sentenciaCitadaRe = sentenciaRe+numeroRe+sentenciaCitadaRe+horasRe+minutosRe+diaRe+mesRe+annoRe
        
        citaSentenciaExp = re.compile(
                sentenciaCitadaRe,  re.X | re.M | re.I)
        matches = re.finditer(citaSentenciaExp, txt)
        sentenciasCitadas = {}
        for match in matches:
            
            startSentencia, endSentencia = match.span(1)
            startHora, endHora = match.span(2)
            startMinutos, endMinutos = match.span(3)
            startDia, endDia = match.span(4)
            startMes, endMes = match.span(5)
            startAnno, endAnno = match.span(6)
            
            #Limpiar sentencia
            sentencia = re.sub("\s+","", doc.text[startSentencia : endSentencia])
            hora = doc.text[startHora : endHora].strip()
            minutos = doc.text[startMinutos : endMinutos].strip()
            dia = doc.text[startDia : endDia].strip()
            mes = doc.text[startMes : endMes].strip()
            #Evitar errores como dos mil 2014
            annoNumeroLimpio = re.search("\d+", doc.text[startAnno : endAnno])
            anno = None
            if annoNumeroLimpio is not None:
                anno = annoNumeroLimpio.group()
            else:
                anno = doc.text[startAnno : endAnno]
            
            (horaNum, minNum, diaNum, annoNum, mesNum) = self.__convertToNumber(hora, minutos, dia, anno, mes)
            
            fecha = ""
            if diaNum is not None and mesNum is not None and annoNum is not None:
                if horaNum is not None:
                    if minNum is not None:
                        fecha = str(diaNum)+"/"+str(mesNum)+"/"+str(annoNum)+" "+str(horaNum)+":"+str(minNum)
                    else:
                        fecha = str(diaNum)+"/"+str(mesNum)+"/"+str(annoNum)+" "+str(horaNum)+":"+"00"
                else:
                    fecha = str(diaNum)+"/"+str(mesNum)+"/"+str(annoNum) +" "+ "00" +":"+"00"
            try:
                fechaSentencia = pd.Timestamp(pd.to_datetime(fecha, format='%d/%m/%Y %H:%M')).to_pydatetime()
            except:
                fechaSentencia = None

            if sentencia in sentenciasCitadas:
                #Verificar si ya existe la sentencia pero no tiene fecha asignada
                if sentenciasCitadas[sentencia] is None:
                    sentenciasCitadas[sentencia] = fechaSentencia
            else:
                #No existe la sentencia aún
                sentenciasCitadas[sentencia] = fechaSentencia
        sentenciasCitadasID = self.__extraerID(sentenciasCitadas, sentenciasCSV)
        return sentenciasCitadas, sentenciasCitadasID