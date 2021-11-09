
import stanza
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
import re
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pyspark.sql import Row
from pyspark.sql.types import IntegerType

from .extraccion.ProcessException import ProcessException
from .extraccion.Natural import Natural
from .extraccion.modelado import NNModel
from .extraccion.utils.Txt2Numbers import Txt2Numbers
from .extraccion.utils.Txt2Date import Txt2Date
from .extraccion.utils.extraerFechaRecibido import ExtraerFecha
from .extraccion.patrones.spacy_entities import extractEntities
from .extraccion.patrones.spacy_internationals import extractInternational
from .extraccion.patrones.spacy_derechos import extractDerechos
from .extraccion.utils.misc import limpiarResolucion, limpiarDerechos


POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',
            'X', 'SPACE']

MACRO_SOLO_RESULTANDO = "resultando"
MACRO_SOLO_CONSIDERANDO = "considerando"
MACRO_SOLO_POR_TANTO = "portanto"
MACRO_SOLO_ENCABEZADO = "encabezado"

def filtrarSecciones(txt, seccion):
    # RegExp para separar la resolución en sus partes
    if(seccion == MACRO_SOLO_POR_TANTO):
        partesExp = re.compile(r"""(?:(?P<encabezado>(?s:.*?))(?=(?i:resultando)|(?i:considerando?\s*\n)|(?i:(?:-|\n)\s*por\ tanto)))   # Match al encabezado
                            (?:(?i:resultando):?(?P<resultando>(?s:.*?))(?=(?i:(?:-|\n)\s*por\ tanto)|(?i:considerando:?\s*\n)))?
                            (?:(?i:considerando):?\s*\n(?P<considerando>(?s:.*?))(?=(?i:resultando[:;,\n])|(?i:(?:-|\n)\s*por\ tanto[:;,\n])))?
                            (?:(?i:(?:-|\n)\s*por\ tanto):?(?P<portanto>(?s:.*)))?""", re.X | re.M)
    else:
        partesExp = re.compile(r"""(?:(?P<encabezado>(?s:.*?))(?=(?i:resultando)|(?i:considerando?\s*\n)|(?i:(?:-|\n)\s*por\ tanto)))
                            (?:(?i:resultando):?(?P<resultando>(?s:.*?))(?=(?i:(?:-|\n)\s*por\ tanto)|(?i:considerando:?\s*\n)))?
                            (?:(?i:considerando):?\s*\n(?P<considerando>(?s:.*?))(?=(?i:resultando)|(?i:(?:-|\n)\s*por\ tanto)))?
                            (?:(?i:(?:-|\n)\s*por\ tanto):?(?P<portanto>(?s:.*(?=(?i:(?:-|\n)\s*por\ tanto)))))?""", re.X | re.M)
    
    
    res = partesExp.search(txt)
    if res is not None and res.group(seccion) is not None:
        return res.group(seccion)
    else:
        return ''

def solo_encabezado(txt):
    """
        Aplica un filtro utilizando una expresión regular de una sentencia y obtiene solamente el texto que corresponde
        a la sección del considerando.

        Retorna:
             Texto correspondiente a la parte del considerando.

        Parametros:

            txt: String
                Representa el texto de la sentencia completa.
    """

    return filtrarSecciones(txt, MACRO_SOLO_ENCABEZADO)

def solo_considerando(txt):
    """
        Aplica un filtro utilizando una expresión regular de una sentencia y obtiene solamente el texto que corresponde
        a la sección del considerando.

        Retorna:
             Texto correspondiente a la parte del considerando.

        Parametros:

            txt: String
                Representa el texto de la sentencia completa.
    """

    return filtrarSecciones(txt, MACRO_SOLO_CONSIDERANDO)

def solo_resultando(txt):
    """
        Aplica un filtro utilizando una expresión regular de una sentencia y obtiene solamente el texto que corresponde
        a la sección del por lo tanto.

        Retorna:
             Texto correspondiente a la parte del por lo tanto.

        Parametros:

            txt: String
                Representa el texto de la sentencia completa.
    """
    return filtrarSecciones(txt,MACRO_SOLO_RESULTANDO)


def solo_portanto(txt):
    """
        Aplica un filtro utilizando una expresión regular de una sentencia y obtiene solamente el texto que corresponde
        a la sección del por lo tanto.

        Retorna:
             Texto correspondiente a la parte del por lo tanto.

        Parametros:

            txt: String
                Representa el texto de la sentencia completa.
    """
    return filtrarSecciones(txt, MACRO_SOLO_POR_TANTO)


def filtro_oraciones(sent, filtro):
    if filtro == []:
        return True

    res = False

    for palabra in filtro:
        res = res or (re.search(palabra, sent.text) is not None)

    return res


def spark_cambios(txt, terminos=None):
    terms = []
    if terminos is not None:
        for key in terminos:
            for term in terminos[key]:
                terms += [key.replace(" ", "")]
                txt = re.sub(term, key.replace(" ", ""),
                             txt, re.I | re.M | re.X)

    return txt, terms

def spark_get_stanza(lang):
    """
        Revisa si ya existe una instancia de Stanza generada previamente. Evita generar una instancia del modelo cada vez que utilizado 

        Retorna:
             Modelo de Stanza

        Parametros:

            lang: String
                Representa el lenguaje a utilizar para crear el modelo de Stanza.
    """
    global nlpStanza
    if "nlpStanza" in globals():
        return nlpStanza
    else:
        nlpStanza = stanza.Pipeline(lang)
        return nlpStanza

def spark_get_spacy(lang):
    global nlp
    if "nlp" in globals():
        return nlp
    else:
        nlp = spacy.load(lang)
        return nlp



def spark_extraer_numero_sentencia(row, newColumns, encabezadoFunction, col="txt"):
    """
        Extrae el numero de sentencia del encabezado del texto, y los agrega
        al Row.

        Retorna:
             El mismo objeto Row de entrada, pero con los valores de las nuevas columnas.

        Parametros:
            row: Row - Spark
                El row al que va a ser aplicado el calculo de extension 

            encabezadoFunction: Function
                Funcion para filtrar el texto por el encabezado.
    """
    res = row.asDict()
    if encabezadoFunction is not None:
        #Filtrar encabezado
        txt = encabezadoFunction(res[col])
    else:
        txt = res[col]
    
    #Expresion para remover el numero de expediente.
    removeExp = re.compile(r"(exp|expediente(:)?(\s)?)(.)?(\s*)?(n|no)?(.)?(°|º)?\s?[0-9]+(-([a-z]*|[A-Z]*))?-[0-9]+", re.IGNORECASE)
    txt = removeExp.sub("", txt)
    #Expresione spara encontrar el numero de voto/resolucion
    voto = re.compile(r"(voto)\s?((n|no)?(\.)?(°|º)?)\s?[0-9]*-[0-9]*", re.IGNORECASE)
    numero = re.compile(r"(n|no)(\.)?(°|º)?\s?[0-9]*(-([a-z]*|[A-Z]*))?-[0-9]*", re.IGNORECASE)
    resolucionRe = re.compile(r"(resoluci[oó]n|res)((\.|:|\s))(\s)*(n|no)?(\.)?(°|º)?(\s)*(([0-9]+(-([a-z]*|[A-Z]*))?-[0-9]*)|[0-9]+)", re.IGNORECASE)
    extraccion = voto.search(txt)
    resolucion = ""
    #Verificar una a una las posibles expresiones regulares.
    if extraccion is not None:
        limpio = limpiarResolucion(extraccion.group(0))
        resolucion = limpio
    else:
        extraccion = resolucionRe.search(txt)
        if extraccion is not None:
            limpio = limpiarResolucion(extraccion.group(0))
            resolucion = limpio
        else:
            extraccion = numero.search(txt)
            if extraccion is not None:
                limpio = limpiarResolucion(extraccion.group(0))
                resolucion = limpio
                
    for column in newColumns:
        if (resolucion != ""):
            res[column] = resolucion
        else:
            res[column] = None
    return Row(**res)


def spark_extraer_fecha_recibido(row, newColumns, resultandoFunction, col="txt"):
    res = row.asDict()
    if resultandoFunction is not None:
        txt = resultandoFunction(res[col])
    else:
        txt = res[col]
    
    date = res['fechahora_ext']
    nlp = spark_get_spacy('es_core_news_lg')
    doc = nlp(txt)
    oraciones = [str(sent).strip() for sent in doc.sents if sent.text != "\n"]
    index = 0
    if len(oraciones) > 0:
        sentenciaConFecha = oraciones[index]
        if(len(sentenciaConFecha) < 5):
            index += 1
            sentenciaConFecha += oraciones[index]
        if sentenciaConFecha.lower().rfind("hrs.") >= 0 and len(oraciones) > 2:
            sentenciaConFecha += oraciones[index + 1]
        
        extraerFecha = ExtraerFecha(nlp)
        fecha = extraerFecha.txt2Date(sentenciaConFecha, date)
    else:
        fecha = None
    for column in newColumns:
        res[column] = fecha
    return Row(**res)

def spark_extraer_extension(row, newColumns, porLoTantoFunction, col="txt"):
    """
        Extrae la extensión de la sentencia y de la parte de corresponde al por lo tanto, y los agrega
        al Row.

        Retorna:
             El mismo objeto Row de entrada, pero con los valores de las nuevas columnas.

        Parametros:
            row: Row - Spark
                El row al que va a ser aplicado el calculo de extension 

            porLoTantoFunction: Function
                Funcion para filtrar el texto y calcular la extensión del por lo tanto.
    """

    res = row.asDict()
    porLoTantoExtension = 0
    sentenciaExtension = 0
    if porLoTantoFunction is not None:
        porLoTantoExtension = len(porLoTantoFunction(res[col]))
    sentenciaExtension = len(res[col])

    res[newColumns[0]] = sentenciaExtension
    res[newColumns[1]] = porLoTantoExtension
    return Row(**res)

def spark_extraer_plazos(row, newColumns, patterns, preprocess, col='txt'):
    """
        Extrae los plazos desde el texto, realiza la conversión a DeltaTime. Finalmente
        convierte el DeltaTime a timestamp y lo almacena en el row.

        Retorna:
             El mismo objeto Row de entrada, pero con los valores de las nuevas columnas.

        Parametros:
            row: Row - Spark
                El row al que va a ser aplicado el calculo de extension 

            porLoTantoFunction: Function
                Funcion para filtrar el texto y calcular la extensión del por lo tanto.
    """
    nlp = spark_get_spacy('es_core_news_lg')
    res = row.asDict()

    regularExpresion = re.compile(r"\.|[Hh][OoÓó][Rr][Aa]([Ss])?|[dD][iíIÍ][ÁAaa]([Ss])?|[Mm][EeÉé][Ss]([EeÉé][Ss])?|[ÁáAa][Ññ][OoÓó]([Ss])?")

    if preprocess is not None:
        txt = preprocess(res[col])
    else:
        txt = res[col]

    plazos = []
    #No procesar las que son sin lugar.
    if res['termino_ext'] in ["Con lugar", "Con lugar parcial"]:
        doc = nlp(txt)
        matcher = Matcher(nlp.vocab)
        matcher.add("Patron 1 :", patterns, greedy="FIRST")

        matches = matcher(doc)

        #Crear los objetos para convertir de texto a número y de para convertir de texto a fechas.
        convertToDate = Txt2Date()
        convertToNumber = Txt2Numbers()
        for _, start, end in matches:

            includeText = False
            plazo = ""
            stringNumber = ""
            for token in doc[start:end]:
                # Recolectar todos los token a partir del primer NUM hasta considir con
                # algunas de las palabras de: [horas, dias, meses, año]
                # eg( tres dias, cuatro meses)
                if includeText:
                    if token.pos_ == "PUNCT":
                        break
                    if regularExpresion.search(token.text) != None:
                        
                        number = convertToNumber.number(textToken)
                        if number != None: 
                            deltaTime = convertToDate.txt2Date(token.text, number)
                            # Convierte los delta time a un datatime y se obtiene el timestamp
                            plazo = pd.Timestamp(pd.to_datetime('1970-01-01') + deltaTime).to_pydatetime()
                        
                        break
                    stringNumber += " " + token.text
                else:
                    if token.pos_ == "NUM":
                        textToken = token.text 
                        if textToken.isdigit() == False:
                            stringNumber += textToken
                            includeText = True
            if plazo != "":
                plazos.append(plazo)
    for column in newColumns:
        if (plazos != []):
            res[column] = plazos
        else:
            res[column] = None
    return Row(**res)

def spark_extraer_instrumentos_internacionales(row, newColumns, preprocess, col='txt'):
    """
    Extrae las sentencias de se ordena de un texto y para cada una de ellas obtiene sus 
    entidades, y los agrega al Row.

    Retorna:
            El mismo objeto Row de entrada, pero con los valores de las nuevas columnas.

    Parametros:
            row: Row - Spark
            El row al que va a ser aplicada la busqueda de entidades

        doc: Document
            El documento procesado por Spacy, de aquí se obtenedrá el span donde se obtienen las entidades.
        TODO:Completar comentarios
    """
    res = row.asDict()
    if preprocess is not None:
        txt = preprocess(res[col])
    else:
        txt = res[col]
    
    
    instrumentos = filtrarInstrumentosInternacionales(txt)
    for column in newColumns:
        if instrumentos:
            res[column] = instrumentos
        else:
            res[column] = None

    return Row(**res)

def spark_extraer_derechos(row, newColumns, preprocess, col='txt'):
    res = row.asDict()
    if preprocess is not None:
        txt = preprocess(res[col])
    else:
        txt = res[col]
     #Aplicar para cada tipo de entidad
    doc = extractDerechos(txt)
    derechosFundamentales = [ent.ent_id_ for ent in doc.ents if ent.label_ == "Derecho Fundamental" or ent.label_ == "Derecho General"]
    derechosExtraidos = [ent.text for ent in doc.ents if ent.label_ == "Derecho Fundamental" or ent.label_ == "Derecho General"]
    unionFundamentales = list(set().union(derechosFundamentales, derechosFundamentales))
    if (unionFundamentales != []):
            res[newColumns[0]] = derechosFundamentales
    else:
            res[newColumns[0]] = None

    if (unionFundamentales != []):
            res[newColumns[1]] = derechosExtraidos
    else:
            res[newColumns[1]] = None

    return Row(**res)



def spark_extraer_derechos_sin_normalizar(row, newColumns, preprocess, col='txt'):
    """
        Extrae las sentencias de se ordena de un texto y para cada una de ellas obtiene sus 
        entidades, y los agrega al Row.

        Retorna:
             El mismo objeto Row de entrada, pero con los valores de las nuevas columnas.

        Parametros:
             row: Row - Spark
                El row al que va a ser aplicada la busqueda de entidades
        TODO:Completar comentarios
    """
    derechoPattern =           [
                         {"LOWER": "derecho"}, {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "+"},
                        ]

    derechoGeneralPattern =    [
                         {"LOWER": "derecho"}, {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "+"},
                         {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "*"},
                         {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "*"}
                        ]
    derechoFundamentalPattern = [
                         {"LOWER": "derecho"}, {"LOWER": "fundamental"}, {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "+"},
                         {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "*"},
                         {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "*"}
                        ]
    derechoHumanoPattern = [
                         {"LOWER": "derecho"}, {"LOWER": "humano"}, {"POS": {"IN":["PRON", "VERB", "DET"]}, "OP": "*"}, {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "+"},
                         {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "*"},
                         {"POS": {"IN":["ADP", "DET"]}, "OP": "*"},
                         {"LOWER": {"IN": ["y", "o"]}, "OP": "?"},
                         {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}, "OP": "*"}
                        ]

    nlp = spark_get_spacy('es_core_news_lg')
    res = row.asDict()
    if preprocess is not None:
        txt = preprocess(res[col])
    else:
        txt = res[col]

    doc = nlp(txt)
    matcher = Matcher(nlp.vocab)
    
    matcher.add("Derecho Acotado", [derechoPattern], greedy="FIRST")
    matcher.add("Derecho General", [derechoGeneralPattern], greedy="FIRST")
    matcher.add("Derecho Fundamental", [derechoFundamentalPattern], greedy="FIRST")
    matcher.add("Derecho Humano", [derechoHumanoPattern], greedy="FIRST")

    entities = {}

    for column in newColumns:
        entities[column] = []

    derechos = []
    derechosGeneral = []
    derechosFundamental = []
    derechosHumano = []

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = Span(doc, start, end, label=match_id)
        if span.label_ == "Derecho Acotado":
            derechos.append(span.text.lower())
        elif span.label_ == "Derecho General":
            derechosGeneral.append(span.text.lower())
        elif span.label_ == "Derecho Fundamental":
            derechosFundamental.append(span.text.lower())
        elif span.label_ == "Derecho Humano":
            derechosHumano.append(span.text.lower())

    derechos = limpiarDerechos(derechos)
    derechosGeneral = limpiarDerechos(derechosGeneral)
    derechosFundamental = limpiarDerechos(derechosFundamental)
    derechosHumano = limpiarDerechos(derechosHumano)

    listaDerechos = [derechos, derechosGeneral, derechosFundamental, derechosHumano]
    for i in range(0, len(newColumns)):
        if (listaDerechos[i] != []):
            res[newColumns[i]] = listaDerechos[i]
        else:
            res[newColumns[i]] = None

    return Row(**res)

    
def spark_extraer_entidades_se_ordena(row, newColumns, patterns, preprocess, col='txt', useSpacy=True):
    """
        Extrae las sentencias de se ordena de un texto y para cada una de ellas obtiene sus 
        entidades, y los agrega al Row.

        Retorna:
             El mismo objeto Row de entrada, pero con los valores de las nuevas columnas.

        Parametros:
             row: Row - Spark
                El row al que va a ser aplicada la busqueda de entidades
        TODO:Completar comentarios
    """
    nlp = spark_get_spacy('es_core_news_lg')
    res = row.asDict()
    if preprocess is not None:
        txt = preprocess(res[col])
    else:
        txt = res[col]

    doc = nlp(txt)
    matcher = Matcher(nlp.vocab)
    matcher.add("Patron 1 :", patterns, greedy="FIRST")

    entities = {}

    for column in newColumns:
        entities[column] = []

    matches = matcher(doc)
    for _, start, end in matches:
        
        textSpan =  doc[start:end]
        if useSpacy:
            getEntitiesBySpacy(doc, start, end, entities, newColumns)
            filtrarEntidadesPublicas(textSpan.text, entities, newColumns)
        else:
            getEntitiesByStanza(textSpan.text, entities, newColumns)
            filtrarEntidadesPublicas(textSpan.text, entities, newColumns)
    for column in newColumns:
        if (entities[column] != []):
            res[column] = entities[column]
        else:
            res[column] = None
    
    return Row(**res)


def filtrarEntidadesPublicas(span, entities, newColumns):
    #Aplicar para cada tipo de entidad
    doc = extractEntities(span)
    entidadesFiltradas = [ent.ent_id_ for ent in doc.ents if ent.label_ == "Entidad Pública" or ent.label_ == "Entidad Pública Acrónimo"]
    entities[newColumns[-1]] = entidadesFiltradas

def filtrarInstrumentosInternacionales(span):
    doc = extractInternational(span)
    entidadesFiltradas = [ent.ent_id_ for ent in doc.ents if ent.label_ in ["Organismo", "Organismo Acronimo", "Tratado Internacional ONU", "Tratado Internacional ONU Acronimo", "Declaracion Internacional ONU", "Resolucion Internacional ONU", "Tratado Internacional OEA", "Tratado Internacional OEA Acronimo", "Resolucion Internacional OEA", "Instrumento Internacional sobre Derechos Humanos"] ]
    return entidadesFiltradas
    
            
    
def getEntitiesBySpacy(doc, start, end, entities,newColumns):
    """
        Obtiene las entidades presentes en un texto utilizando Spacy

        Retorna:
             Un arreglo de strings, la forma del string es tipo : entidad.

        Parametros:

            doc: Document
                El documento procesado por Spacy, de aquí se obtenedrá el span donde se obtienen las entidades.
            start: int
                Inicio del span donde se obtendran las entidades.
            end: int
                Fin del span donde se obtendran las entidades.
    """
    for ent in doc[start:end].ents:
        if ent.label_ == "PER":
            entities[newColumns[0]].append(ent.text)
            continue
        if ent.label_ == "LOC":
            entities[newColumns[1]].append(ent.text)
            continue
        if ent.label_ == "ORG":
            entities[newColumns[2]].append(ent.text)
            continue
        if ent.label_ == "MISC":
            entities[newColumns[3]].append(ent.text)
            continue
        if ent.label_ == "GPE":
            entities[newColumns[4]].append(ent.text)
            continue

def getEntitiesByStanza(text, entities, newColumns):
    """
        Obtiene las entidades presentes en un texto utilizando Stanza

        Retorna:
             Un arreglo de strings, la forma del string es tipo : entidad.

        Parametros:

            text: String
                String para ser procesado por Stanza. De acá se obtienen las entidades.
    """
    nlp = spark_get_stanza("es")
    doc = nlp(text)
    for ent in doc.ents:
        if ent.type == "PER":
            entities[newColumns[0]].append(ent.text)
            continue
        if ent.type == "LOC":
            entities[newColumns[1]].append(ent.text)
            continue
        if ent.type == "ORG":
            entities[newColumns[2]].append(ent.text)
            continue
        if ent.type == "MISC":
            entities[newColumns[3]].append(ent.text)
            continue
        if ent.type == "GPE":
            entities[newColumns[4]].append(ent.text)
            continue

def spark_buscar_terminos_doc(row, terminos, col='txt', preprocess=None, keepRowEmpty=False):
    """
        Ejecuta la busqueda de terminos revisando la cantidad de ocurrencias de las expresiones regulares para un row en especifico

        Parametros:
        
        row: Row - Spark
            El row al que va a ser aplicada la busqueda de terminos
        
        terminos: Diccionario (e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]})
            El conjunto de terminos a buscar cada valor corresponde a una expresion regular compilada
        
        col: String
            Columna del row donde será aplicada la busqueda, de forma predeterminada está la columna txt correspondiente al texto de la sentencia.

    """
    import sys
    sys.path.insert(0, "/home/jovyan/Work/ej/paquetes/nlppen/")

    import nlppen

    if terminos == None or terminos == []:
        assert "No se han especificado términos"

    res = row.asDict()
    tiene_terminos = False

    if preprocess is not None:
        txt = preprocess(res[col])
    else:
        txt = res[col]
    for key in terminos: # Recorrer cada termino.
        for reg in terminos[key]: # Recorrer cada expresión regular.
            resultado = reg.findall(txt) #Busca todas las ocurrencias.
            if key not in res:
                res[key] = 0 # Crea la columna en el row
            tiene_terminos = len(resultado) != 0 or tiene_terminos
            res[key] += len(resultado)

    if keepRowEmpty:
        return Row(**res)
    else:
        if tiene_terminos: # Retorna un objeto Row
            return Row(**res)
        else:
            return None


def spark_extraer_tokens(batch, index_col='index', txt_col='txt', incluir=[], cambios={}, preprocess=None):
    nlp = spark_get_spacy('es_core_news_lg')
    limpiar_regex = re.compile(r'[^A-Za-záéíóúüïÁÉÍÓÚÜÏñÑ]')

    if incluir == []:
        incluir = POS_TAGS

    for df in batch:
        palabras = []
        for indx, row in df.iterrows():
            if row[txt_col] is not None:
                if preprocess is not None:
                    txt = process(row[txt_col])
                else:
                    txt = row[txt_col]
                txt, terms = spark_cambios(txt.lower(), cambios)
                doc = nlp(txt)
                palabras += [
                    {'index': str(row[index_col]),
                    'palabra': limpiar_regex.sub('', t.lemma_.lower()),
                    'POS': t.pos_,
                    'Dep': t.dep_}
                    for t in doc
                    if not t.is_stop and
                    t.pos_ in incluir
                    and limpiar_regex.sub('', t.lemma_.lower()) != '']

        yield pd.DataFrame(palabras)


def spark_skipgrams(batch, n=3, k=1, txt_col='txt',filtro=[], cruce='index', incluir=['PROPN', 'NOUN', 'VERB', 'ADJ'], cambios=None, preprocess=None):
    from nltk import skipgrams
    import re

    limpiar_regex = re.compile(r'[^A-Za-záéíóúüïÁÉÍÓÚÜÏ]')

    def limpiar(t):
        return limpiar_regex.sub('', t).lower()

    nlp = spark_get_spacy('es_core_news_lg')

    res = []
    for df in batch:
        for indx, row in df.iterrows():
            if row[txt_col] is not None:
                if preprocess is not None:
                    txt = preprocess(row[txt_col].lower())
                else:
                    txt = row[txt_col]
                txt, _ = spark_cambios(txt, cambios)
                doc = nlp(txt)
                # sents = [skipgrams(sent, n, k)
                #          for sent in doc.sents if filtro_oraciones(sent, filtro)]
                sents = [skipgrams(doc, n, k)]
                for sk in sents:
                    res += [{**{'cruce': row[cruce]}, **{f't{tn}': '__'.join([limpiar(t.lower_), t.pos_])
                             for tn, t in enumerate(sorted(skipgram)) if t.pos_ in incluir and limpiar(t.lemma_) != ''}}
                            for skipgram in sk]

        df = pd.DataFrame(res).dropna().reset_index().groupby(
            ['t'+str(tn) for tn in range(n)]+['cruce']).count()

        yield df.reset_index().rename(columns={'index': 'freq'})


def spark_aplicar_extraccion(res, doc, nombreProc, procesarDoc):
    try:
        res.update(procesarDoc(doc))
    except ProcessException as err:
        proc, msg = err.args
        res['error'].update({nombreProc:  msg})
    return res


def spark_aplicar_entidades(res, doc, nombreProc, procesarDoc):
    try:
        proc = procesarDoc(doc)
        entys = {f'entidades.{k}': v for k, v in proc['entidades'].items()}
        entys['texts.entidades'] = proc['texts.entidades_html']
        res.update(entys)

    except ProcessException as err:
        proc, msg = err.args
        res['error'].update({nombreProc:  msg})
    return res


def spark_extraccion_info(txt, model=None):
    natural = Natural()
    doc = {'txt': txt}
    res = {'error': {}}

    doc.update(spark_aplicar_extraccion(
        res, doc, 'secciones', natural.secciones))

    spark_aplicar_extraccion(
        res, doc, 'extraccion_expediente', natural.expedientes)
    spark_aplicar_extraccion(
        res, doc, 'extraccion_fechahora', natural.fechahora)
    spark_aplicar_extraccion(
        res, doc, 'extraccion_tipo_proceso', natural.tipo_proceso)
    spark_aplicar_extraccion(
        res, doc, 'extraccion_sentencia', natural.sentencia)
    spark_aplicar_extraccion(
        res, doc, 'extraccion_voto_salvado', natural.voto_salvado)
    spark_aplicar_extraccion(
        res, doc, 'extraccion_redactor', natural.redactor)
    spark_aplicar_extraccion(
        res, doc, 'extraccion_fechahora', natural.fechahora)
    # spark_aplicar_extraccion(res, doc, 'extraccion_magistrados', natural.extraer_magistrados)
    spark_aplicar_extraccion(
        res, doc, 'extraccion_lemma', natural.lematizacion)

    spark_aplicar_entidades(
        res, doc, 'extraccion.entidades', natural.entidades)

    if model is not None and 'texts.lemma' in res:
        spark_estimar_tema(res, model, res['texts.lemma'])
        

    return res


def cargar_modelo( model_file, encoder_file, vectorizer_file, tfidf_file):
    try:
        model
    except:
        model = NNModel(None, None)
        model.load_model(model_file, encoder_file, vectorizer_file, tfidf_file)
    
    return model


def spark_estimar_tema(res, model, lemma):
    with tf.device('/cpu:0'):
        model.predict(np.array([lemma]))
    
    est = model.estimated[0]
    probs = {k:v for k,v in zip(model.encoder.classes_.tolist(), model.predicted.tolist()[0])}

    res.update({'estimado.tema': est, 'estimado.tema.probs': probs})
    return res


def spark_extraer_dependencias(batch, filtro=[], cruce='index', excluir=None, cambios=None):
    import re

    if excluir == None:
        excluir = ['det', 'case', 'fixed', 'punct', 'cc', 'appos', 'nummod', 'mark', 'dep', 'expl:pass', 'flat', 'aux']

    limpiar_regex = re.compile(r'[^A-Za-záéíóúüïÁÉÍÓÚÜÏ]')

    def limpiar(t):
        return limpiar_regex.sub('', t).lower()

    nlp = spark_get_spacy('es_core_news_lg')

    res = []
    for df in batch:
        for indx, row in df.iterrows():
            txt = solo_considerando(row.txt.lower())
            txt, _ = spark_cambios(txt, cambios)
            doc = nlp(txt)


            res += [{'root':limpiar(token.head.lemma_), 
                     'root_POS':limpiar(token.head.pos_), 
                     'child': limpiar(token.lemma_), 
                     'child_POS':limpiar(token.pos_),
                     'dep': token.dep_,
                     'cruce': row[cruce]} 
                        for token in doc if token.dep_not in excluir
                    ]


        yield (pd.DataFrame(res)
                .reset_index()
                .groupby(['root','root_POS', 'child', 'child_POS', 'dep'])
                .count()
                .reset_index()
                )