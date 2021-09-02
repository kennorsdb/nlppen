from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.language import Language
from nltk.parse import CoreNLPParser
import spacy
import re

# java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
# -serverProperties StanfordCoreNLP-spanish.properties \
# -preload tokenize,ssplit,pos,ner,parse \
# -status_port 9003  -port 9003 -timeout 15000

@Language.component('Sentencias')
def sentencias_entities(doc):
    res = []
    #FEcha y Hora
    datetime = re.compile(r"""(?:(?:las)|(?:12s))[\s]+(?P<horas>[áéíóú\w\s]+?)[\s]+horas[,]?[\s]+(?:(?:y[\s])|)(?:(?:con[\s])|)(?:(?:(?P<minutos>[áéíóú\w\s]+?)
                                        [\s]+minutos?)|)[\s]*(?:(?:de[l]?)|)[\s]+(?:d[íi]a[\s]+)?
                                        (?P<dias>[áéíóú\w\s]*?)[\s]+de[\s]+(?P<mes>.+?)[\s]*(?:(?:de[l]?)|)(?:(?:año)|)[\s]+(?:(?:un[\s])|)
                                        (?P<annoHigh>(?:(?i:mil[\s]+novecientos)|(?i:dos\s+mil)))[\s]*(?:(?P<annoLow>[áéíóú\w\s]+?)|)[\s]*\.""", re.M | re.X)      
    
    for match in re.finditer(datetime, doc.text):
        start, end = match.span()
        res.append(doc.char_span(start, end, label="Fechahora"))

    #Recurrente
    recurrenteRegex1 = re.compile(
            r"""»?Recurrente:(?P<recurrente>[\s\w,]+)(?:(?:Agraviado)|(?:Recurrido)|\n)""", re.I | re.M | re.X)
    recurrenteRegex2 = re.compile(r"""(?:(?:amparo\sde)|(?:interpuest[oa])|(?:promovid[oa])|(?:presentad[oa])|(?:establecid[oa])|(?:plantead[oa])|(?:por\s+el)|(?:sobre\s+el)|(?:inconstitucionalidad))\s+
            (?:por\s+)?(?P<recurrente>[áéíóúÁÉÍÓÚ\-\s\w]+)(?=(?:;)|(?:,)|(?:a\s+favor)|(?:contra)|(?:en)|(?:para\sque)|(?:sobre)).*(?=resultando)""", re.I | re.M | re.X | re.S)
    
    for match in re.finditer(recurrenteRegex1, doc.text):
        start, end = match.span(1)
        res.append(doc.char_span(start, end, label="Recurrente"))
    
    for match in re.finditer(recurrenteRegex2, doc.text):
        start, end = match.span(1)
        res.append(doc.char_span(start, end, label="Recurrente"))
    
    # Recurrido
    recurridoRegex1 = re.compile(r"""»?Recurrido:(?P<recurrido>[\s\w,]+)(?:(?:Sala\sConstitucional)|\n)""", re.I | re.M | re.X)
    recurridoRegex2 = re.compile(r"""contra\s+(?P<recurrido>[áéíóúÁÉÍÓÚ\-\s\w]+)[.,;]+?.*(?=resultando)""", re.I | re.M | re.X | re.S)
    for match in re.finditer(recurridoRegex1, doc.text):
        start, end = match.span(1)
        res.append(doc.char_span(start, end, label="Recurrido"))
    
    for match in re.finditer(recurridoRegex2, doc.text):
        start, end = match.span(1)
        res.append(doc.char_span(start, end, label="Recurrido"))
    
    # Redactor
    redactorExp = re.compile(
            r"""Redacta.*magistrad[oa]\s+(?P<redactor>.+?)[;,\s\.]{2,}""",  re.X | re.M | re.I)
    for match in re.finditer(redactorExp, doc.text):
        start, end = match.span(1)
        res.append(doc.char_span(start, end, label="Redactor"))

    # Constitución
    constitucionExp = re.compile(
            r"""art[íi]culos*\s+((?:\d+(?:(?:,\s*)|(?:\s+y\s+))*)+)\s*de\s*la\s*Constituci[oó]n""",  re.X | re.M | re.I)
    for match in re.finditer(constitucionExp, doc.text):
        start, end = match.span(1)
        res.append(doc.char_span(start, end, label="Constitución"))
        
        
    # Cita Sentencia
    citaSentenciaExp = re.compile(
            r"""Sentencia\s+?(?:(?:n[uú]mero)|(?:N[ºo°]\.?))\s*(?P<sentencia_citada>[\d\-\.]+)""",  re.X | re.M | re.I)
    for match in re.finditer(citaSentenciaExp, doc.text):
        start, end = match.span(1)
        res.append(doc.char_span(start, end, label="CitaSentencia"))
        
    # Magistrados
    magistrado = re.compile(
        r"""(?<=por\stanto).+\n(?P<presidente>[^\n]+)\s+presidente\.?\s+([^\n]+?)\s{2,}(\w+(?:\ |\n)[^\n]+)\n([^\n]+?)\s{2,}(\w+(?:\ |\n)[^\n]+)\n([^\n]+?)\s{2,}(\w+(?:\ |\n)[^\n]+)\n""", re.I | re.M | re.X | re.S)
    match = magistrado.search(doc.text)
    if match is not None:
        for i, g in enumerate(match.groups()):
            start, end = match.span(i+1)
            res.append(doc.char_span(start, end, label="Magistrado"))   

    if doc.ents is not None:
        for r in res:
            if r is not None:
                try:
                    doc.ents = (*doc.ents, r)
                except ValueError as V:
                    pass
    else:
        doc.ents = [r for r in res if r is not None]
        
    return doc




def nlp_corte(text, personas=True):
    patterns = [
        # Leyes, Reglamentos y Decretos
        {"label": "Reglamento", "pattern": [{"LOWER": "reglamento"}, {"IS_STOP": True, "OP":"+"},  {"IS_TITLE": True, "OP":"*"}, {"IS_STOP": True, "OP":"*"},  {"IS_TITLE": True, "OP":"*"}]},
        {"label": "Reglamento", "pattern": [{"LOWER": "reglamento"}, {"IS_STOP": True, "OP":"+"},  {"IS_UPPER": True, "OP":"*"}, {"IS_STOP": True, "OP":"*"},  {"IS_UPPER": True, "OP":"*"}]},
        {"label": "Ley", "pattern": [{"LOWER": "ley"}, {"IS_STOP": True, "OP":"+"},  {"IS_TITLE": True, "OP":"+"}, {"IS_STOP": True, "OP":"*"},  {"IS_TITLE": True, "OP":"*"}]  },
        {"label": "Ley", "pattern": [{"LOWER": "ley"}, {"IS_STOP": True, "OP":"+"},  {"IS_UPPER": True, "OP":"+"}, {"IS_STOP": True, "OP":"*"},  {"IS_UPPER": True, "OP":"*"}]   },
        {"label": "Ley", "pattern": [{"LOWER": {"IN": ["codigo", "código"]}}, {"IS_STOP": True, "OP":"*"}, {"IS_TITLE": True, "OP":"+"}]  },
        {"label": "Ley", "pattern": [{"LOWER": {"IN": ["codigo", "código"]}}, {"IS_STOP": True, "OP":"*"}, {"IS_UPPER": True, "OP":"+"}]  },
        {"label": "Decreto", "pattern": [{"LOWER": {"IN": ["decreto"]}}, {"LOWER": {"IN": ["ejecutivo"]}, "OP":"?"}, {"LOWER": {"IN": ["n", "no", "°", "º"]}, "OP": "*"}, {"IS_PUNCT": True, "OP":"*"}, {"TEXT": {"REGEX": r"[Nn]?.*\d+\-?.*$"}} ]  },


        # Fecha y Hora
        {"label": "Fecha", "pattern": [ {"IS_DIGIT": True}, {"IS_STOP": True}, {"IS_ASCII": True}, {"IS_STOP": True},{"IS_DIGIT": True} ] },

        # Cantón
        { "label": "Cantón", "pattern" : [ { "LOWER": 'abangares'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'acosta'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'aguirre'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'alajuela'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'alajuelita'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'alvarado'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['aserrí', 'aserri'] } } ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'atenas'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'bagaces'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'barva'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['belén', 'belen' ]} } ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'buenos'}, { "LOWER": 'aires'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'cañas'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'carrillo'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'cartago'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'corredores'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'coto'}, { "LOWER": 'brus'} ] },
        { "label": "Cantón", "pattern" : [ { "LOWER": 'curridabat'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'desamparados'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'dota'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'el'}, { "LOWER": 'guarco'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['escazú', 'escazu' ] } } ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'esparza'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'flores'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'garabito'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'goicoechea'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'golfito'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'grecia'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['guácimo', 'guacimo' ] } } ] },
        { "label": "Cantón", "pattern" : [ { "LOWER": 'guatuso'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'heredia'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'hojancha'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['jiménez', 'jimenez'] } } ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'la'}, { "LOWER": 'cruz'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'la'}, { "LOWER": { "IN": ['unión', 'union'] } } ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['león', 'leon']}}, { "LOWER": { "IN": ['cortés', 'cortes' ] } } ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'liberia'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['limón', 'limon'] } } ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'los'}, { "LOWER": 'chiles'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'matina'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'montes'}, { "LOWER": 'de'}, { "LOWER": 'oca'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'montes'}, { "LOWER": 'de'}, { "LOWER": 'oro'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'mora'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'moravia'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'nandayure'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'naranjo'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'nicoya'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'oreamuno'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'orotina'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'osa'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'palmares'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['paraíso', 'paraiso'] } } ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'parrita'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['pérez', 'perez']}}, { "LOWER": { "IN": ['zeledón', 'zeledon']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['poás', 'poas']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['pococí', 'pococi']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'puntarenas'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'puriscal'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'san'}, { "LOWER": 'carlos'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'san'}, { "LOWER": 'isidro'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'san'}, { "LOWER": 'jose'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'san'}, { "LOWER": 'mateo'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'san'}, { "LOWER": 'pablo'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'san'}, { "LOWER": 'rafael'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'san'}, { "LOWER": { "IN": ['ramón', 'ramon']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'santa'}, { "LOWER": 'ana'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'santa'}, { "LOWER": { "IN": ['bárbara', 'barbara']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'santa'}, { "LOWER": 'cruz'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'santo'}, { "LOWER": 'domingo'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['sarapiquí', 'sarapiqui']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'siquirres'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'talamanca'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['tarrazú', 'tarrazu']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['tibás', 'tibas']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['tilarán', 'tilaran']}} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'turrialba'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'turrubares'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'upala'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'valverde'}, { "LOWER": 'vega'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": { "IN": ['vázquez', 'vazquez']}}, { "LOWER": 'de'}, { "LOWER": 'coronado'} ]},
        { "label": "Cantón", "pattern" : [ { "LOWER": 'zarcero}'} ]},

        # Instituciones Publicas
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "academia"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ciencias"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["auditoria", "auditoría"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "servicios"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "salud"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "autoridad"}, {"LOWER": "presupuestaria"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "autoridad"}, {"LOWER": "reguladora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "servicios"}, {"LOWER": {"IN": ["publicos", "públicos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": "central"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": {"IN": ["credito", "crédito"]}}, {"LOWER": {"IN": ["agricola", "agrícola"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "cartago"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}, {"LOWER": "planes"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pension"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}, {"LOWER": "sociedad"}, {"LOWER": "administradora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fondos"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["inversion", "inversión"]}}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}, {"LOWER": "valores"}, {"LOWER": "puesto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "bolsa"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": "hipotecario"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vivienda"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": "internacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": "nacional"}, {"LOWER": "sociedad"}, {"LOWER": "administradora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fondos"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["inversion", "inversión"]}}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": "nacional"}, {"LOWER": "valores"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": "popular"}, {"LOWER": "operadora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pensiones"}, {"LOWER": "complementarias"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "banco"}, {"LOWER": "popular"}, {"LOWER": "y"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "comunal"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "bn"}, {"LOWER": "vital"}, {"LOWER": "operadora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pensiones"}, {"LOWER": "complementarias"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "caja"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguro"}, {"LOWER": "social"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "casa"}, {"LOWER": "hogar"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "tia"}, {"LOWER": "tere"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "centro"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ciencia"}, {"LOWER": "y"}, {"LOWER": "cultura"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "museo"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "niños"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "centro"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["produccion", "producción"]}}, {"LOWER": {"IN": ["cinematografica", "cinematográfica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "centro"}, {"LOWER": "cultural"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["historico", "histórico"]}}, {"LOWER": {"IN": ["jose", "josé"]}}, {"LOWER": "figueres"}, {"LOWER": "ferrer"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "centro"}, {"LOWER": "cultural"}, {"LOWER": "herediano"}, {"LOWER": "omar"}, {"LOWER": "dengo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "centro"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "musica"}, {"LOWER": "incluye"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "orquesta"}, {"LOWER": {"IN": ["sinfonica", "sinfónica"]}}, {"LOWER": "nacional"}, {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["musica", "música"]}}, {"LOWER": "coro"}, {"LOWER": {"IN": ["sinfonico", "sinfónico"]}}, {"LOWER": "nacional"}, {"LOWER": {"IN": ["compañia", "compañía"]}}, {"LOWER": {"IN": ["lirica", "lírica"]}}, {"LOWER": "nacional"}, ]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "profesionales"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["nutricion", "nutrición"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"LOWER": "agropecuario"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "san"}, {"LOWER": "carlos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "abogados"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "bibliotecarios"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["biologos", "biólogos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "cirujanos"}, {"LOWER": "dentistas"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "contadores"}, {"LOWER": "privados"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "contadores"}, {"LOWER": {"IN": ["publicos", "públicos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "enfermeras"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["farmaceuticos", "farmacéuticos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["fisicos", "físicos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["geologos", "geólogos"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ingenieros"}, {"LOWER": {"IN": ["agronomos", "agrónomos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ingenieros"}, {"LOWER": {"IN": ["quimicos", "químicos"]}}, {"LOWER": "y"}, {"LOWER": "profesionales"}, {"LOWER": "afines"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "licenciados"}, {"LOWER": "y"}, {"LOWER": "profesores"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "letras"}, {"LOWER": "y"}, {"LOWER": {"IN": ["filosofia", "filosofía"]}}, {"LOWER": "ciencias"}, {"LOWER": "y"}, {"LOWER": "artes"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "medicos"}, {"LOWER": "veterinarios"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["medicos", "médicos"]}}, {"LOWER": "y"}, {"LOWER": "cirujanos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["microbiologos", "microbiólogos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "periodistas"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "profesionales"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["psicologia", "psicología"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "profesionales"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ciencias"}, {"LOWER": {"IN": ["economicas", "económicas"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "profesionales"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ciencias"}, {"LOWER": {"IN": ["politicas", "políticas"]}}, {"LOWER": "y"}, {"LOWER": "relaciones"}, {"LOWER": "internacionales"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "profesionales"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["informatica", "informática"]}}, {"LOWER": "y"}, {"LOWER": {"IN": ["computacion", "computación"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "profesionales"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["quiropractica", "quiropráctica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["quimicos", "químicos"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "trabajadores"}, {"LOWER": "sociales"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"LOWER": "federado"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ingenieros"}, {"LOWER": "y"}, {"LOWER": "arquitectos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"LOWER": "san"}, {"LOWER": "luis"}, {"LOWER": "gonzaga"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "colegio"}, {"LOWER": "universitario"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["limon", "limón"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "mejora"}, {"LOWER": "regulatoria"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ordenamiento"}, {"LOWER": "y"}, {"LOWER": "manejo"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "cuenca"}, {"LOWER": "alta"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "rio"}, {"LOWER": {"IN": ["reventazon", "reventazón"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "comision"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "promocion"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "competencia"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "comision"}, {"LOWER": "interinstitucional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "marinas"}, {"LOWER": "y"}, {"LOWER": "atracaderos"}, {"LOWER": {"IN": ["turisticos", "turísticos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "asuntos"}, {"LOWER": {"IN": ["indigenas", "indígenas"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "conmemoraciones"}, {"LOWER": {"IN": ["historicas", "históricas"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["energia", "energía"]}}, {"LOWER": {"IN": ["atomica", "atómica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "nomenclatura"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["préstamos", "prestamos"]}}, {"LOWER": "para"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["educación", "educacion"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["prevención", "prevencion"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "riesgos"}, {"LOWER": "y"}, {"LOWER": {"IN": ["atencion", "atención"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "emergencias"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "comision"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vacunacion"}, {"LOWER": "y"}, {"LOWER": {"IN": ["epidemiologia", "epidemiología"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "comision"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "consumidor"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "comision"}, {"LOWER": "nacional"}, {"LOWER": "para"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "defensa"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "idioma"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"LOWER": "nacional"}, {"LOWER": "para"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["gestion", "gestión"]}}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "biodiversidad"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "comision"}, {"LOWER": "reguladora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "turismo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comision", "comisión"]}}, {"LOWER": {"IN": ["tecnica", "técnica"]}}, {"LOWER": {"IN": ["filatelica", "filatélica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comite", "comité"]}}, {"LOWER": "coordinador"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "programa"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "rural"}, {"LOWER": "integrado"}, {"LOWER": "osa"}, {"LOWER": "golfito"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comites", "comités"]}}, {"LOWER": "cantonales"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "deportes"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["comites", "comités"]}}, {"LOWER": "cantonales"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "persona"}, {"LOWER": "joven"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["compañia", "compañía"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "danza"}, {"LOWER": "programa"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "teatro"}, {"LOWER": "melico"}, {"LOWER": "salazar"}, ]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["compañia", "compañía"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fuerza"}, {"LOWER": "y"}, {"LOWER": "luz"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "atencion"}, {"LOWER": "integral"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "salud"}, {"LOWER": "ocupacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguridad"}, {"LOWER": "vial"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "fondo"}, {"LOWER": "editorial"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "interinstitucional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["atencion", "atención"]}}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "madre"}, {"LOWER": "adolescente"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "concesiones"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "cooperativas"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "comunidad"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "enseñanza"}, {"LOWER": "superior"}, {"LOWER": "universitaria"}, {"LOWER": "privada"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["espectaculos", "espectáculos"]}}, {"LOWER": {"IN": ["publicos", "públicos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "investigaciones"}, {"LOWER": {"IN": ["cientificas", "científicas"]}}, {"LOWER": "y"}, {"LOWER": {"IN": ["tecnologicas", "tecnológicas"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "niñez"}, {"LOWER": "y"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "adolescencia"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "persona"}, {"LOWER": "adulta"}, {"LOWER": "mayor"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["produccion", "producción"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["politica", "política"]}}, {"LOWER": {"IN": ["publica", "pública"]}}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "persona"}, {"LOWER": "joven"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "rectores"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["rehabilitacion", "rehabilitación"]}}, {"LOWER": "y"}, {"LOWER": {"IN": ["educacion", "educación"]}}, {"LOWER": "especial"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "salarios"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["supervision", "supervisión"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "sistema"}, {"LOWER": "financiero"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "transporte"}, {"LOWER": {"IN": ["publico", "público"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vialidad"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "nacional"}, {"LOWER": "para"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "calidad"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "portuario"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "rector"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "sistema"}, {"LOWER": "banca"}, {"LOWER": "para"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": "superior"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "educacion"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": {"IN": ["tecnico", "técnico"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "asistencia"}, {"LOWER": "medico"}, {"LOWER": "social"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejo"}, {"LOWER": {"IN": ["tecnico", "técnico"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["aviacion", "aviación"]}}, {"LOWER": "civil"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejos"}, {"LOWER": "regionales"}, {"LOWER": "ambientales"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "consejos"}, {"LOWER": {"IN": ["tecnicos", "técnicos"]}}, {"LOWER": "asesores"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "estaciones"}, {"LOWER": "experimentales"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["contraloria", "contraloría"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["republica", "república"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["corporacion", "corporación"]}}, {"LOWER": "arrocera"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["corporacion", "corporación"]}}, {"LOWER": "bananera"}, {"LOWER": "nacional"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["corporacion", "corporación"]}}, {"LOWER": "ganadera"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["corporacion", "corporación"]}}, {"LOWER": {"IN": ["horticola", "hortícola"]}}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "correos"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "cuerpo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "bomberos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["defensoria", "defensoría"]}}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "habitantes"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["direccion", "dirección"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["geología", "geología"]}}, {"LOWER": "minas"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "hidrocarburos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "direccion"}, {"LOWER": "ejecutora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "proyectos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["direccion", "dirección"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["adaptacion", "adaptación"]}}, {"LOWER": "social"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["direccion", "dirección"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "social"}, {"LOWER": "y"}, {"LOWER": "asignaciones"}, {"LOWER": "familiares"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["direccion", "dirección"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "hidrocarburos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["direccion", "dirección"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["migracion", "migración"]}}, {"LOWER": "y"}, {"LOWER": {"IN": ["extranjeria", "extranjería"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["direccion", "dirección"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "archivo"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["direccion", "dirección"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "servicio"}, {"LOWER": "civil"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "direccion"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "comunidad"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ecomuseo"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "minas"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "abangares"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "editorial"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "empresa"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "servicios"}, {"LOWER": {"IN": ["publicos", "públicos"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "heredia"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ente"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["acreditacion", "acreditación"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "escuela"}, {"LOWER": {"IN": ["tecnica", "técnica"]}}, {"LOWER": {"IN": ["agricola", "agrícola"]}}, {"LOWER": "industrial"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fabrica"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "licores"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "social"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["limon", "limón"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "social"}, {"LOWER": "y"}, {"LOWER": "asignaciones"}, {"LOWER": "familiares"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "parques"}, {"LOWER": "nacionales"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["preinversion", "preinversioón"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vida"}, {"LOWER": "silvestre"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": "especial"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "servicio"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "guardacostas"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": "especifico"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["migracion", "migración"]}}, {"LOWER": "y"}, {"LOWER": {"IN": ["extranjeria.", "extranjería."]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": "forestal"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": {"IN": ["jurisdiccion", "jurisdiccion"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["transito", "tránsito"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": "nacional"}, {"LOWER": "ambiental"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "becas"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "emergencias"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["estabilizacion", "estabilización"]}}, {"LOWER": "cafetalera"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "fondo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "financiamiento"}, {"LOWER": "forestal"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "imprenta"}, {"LOWER": "nacional"}, {"LOWER": "junta"}, {"LOWER": "administrativa"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "acueductos"}, {"LOWER": "y"}, {"LOWER": "alcantarillados"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "electricidad"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ferrocarriles"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "investigacion"}, {"LOWER": "y"}, {"LOWER": "enseñanza"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["nutricion", "nutrición"]}}, {"LOWER": "y"}, {"LOWER": "salud"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pesca"}, {"LOWER": "y"}, {"LOWER": "acuacultura"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "puertos"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["pacifico", "pacífico"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "turismo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "deporte"}, {"LOWER": "y"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["recreacion", "recreación"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "costarricense"}, {"LOWER": "sobre"}, {"LOWER": "drogas"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "agrario"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "profesional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fomento"}, {"LOWER": "y"}, {"LOWER": {"IN": ["asesoria", "asesoría"]}}, {"LOWER": "municipal"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["cafe", "café"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": {"IN": ["geográfico", "geográfico"]}}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "latinoamericano"}, {"LOWER": {"IN": ["prevencion", "prevención"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "delito"}, {"LOWER": "y"}, {"LOWER": "tratamiento"}, {"LOWER": "delincuente"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": {"IN": ["meteorologico", "meteorológico"]}}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "mixto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ayuda"}, {"LOWER": "social"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "aprendizaje"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["estadisticas", "estadísticas"]}}, {"LOWER": "y"}, {"LOWER": "censos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fomento"}, {"LOWER": "cooperativo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["innovacion", "innovación"]}}, {"LOWER": "y"}, {"LOWER": "transferencia"}, {"LOWER": {"IN": ["tecnologica", "tecnólogica"]}}, {"LOWER": "agropecuaria"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "mujeres"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguros"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguros"}, {"LOWER": "bancredito"}, {"LOWER": "operadora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pensiones"}, {"LOWER": "complementarias"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguros"}, {"LOWER": "bancredito"}, {"LOWER": "sociedad"}, {"LOWER": "administradora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fondos"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["inversion", "inversión"]}}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguros"}, {"LOWER": "bancredito"}, {"LOWER": "valores"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vivienda"}, {"LOWER": "y"}, {"LOWER": "urbanismo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": "sobre"}, {"LOWER": "alcoholismo"}, {"LOWER": "y"}, {"LOWER": "farmacodependencia"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "instituto"}, {"LOWER": {"IN": ["tecnologico", "tecnológico"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"LOWER": "administradora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "servicios"}, {"LOWER": {"IN": ["eléctricos", "eléctricos"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "cartago"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"LOWER": "administrativa"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "centros"}, {"LOWER": {"IN": ["civicos", "cívicos"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"LOWER": "administrativa"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "archivo"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"LOWER": "administrativa"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "parque"}, {"LOWER": "recreativo"}, {"LOWER": "playas"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "manuel"}, {"LOWER": "antonio"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"LOWER": "administrativa"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "registro"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"LOWER": "administrativa"}, {"LOWER": "portuaria"}, {"LOWER": "y"}, {"LOWER": "desarrollo"}, {"LOWER": {"IN": ["economico", "económico"]}}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vertiente"}, {"LOWER": {"IN": ["atlantica", "atlántica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "zona"}, {"LOWER": "sur"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fomento"}, {"LOWER": {"IN": ["avicola", "avícola"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fomento"}, {"LOWER": "porcino"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fomento"}, {"LOWER": "salinero"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pensiones"}, {"LOWER": "y"}, {"LOWER": "jubilaciones"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "magisterio"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["proteccion", "protección"]}}, {"LOWER": "social"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vigilancia"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "drogas"}, {"LOWER": "y"}, {"LOWER": "estupefacientes"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "cabuya"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "junta"}, {"LOWER": "promotora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "turismo"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ciudad"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "puntarenas"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "juntas"}, {"LOWER": "administrativas"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "colegios"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "juntas"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["educacion", "educación"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "escuelas"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "juntas"}, {"LOWER": "viales"}, {"LOWER": "cantonales"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "laboratorio"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "metrologia"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "liga"}, {"LOWER": {"IN": ["agricola", "agrícola"]}}, {"LOWER": "industrial"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "caña"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "agricultura"}, {"LOWER": "y"}, {"LOWER": {"IN": ["ganaderia", "ganadería"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ciencia"}, {"LOWER": "y"}, {"LOWER": {"IN": ["tecnologia", "tecnología"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "comercio"}, {"LOWER": "exterior"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "cultura"}, {"LOWER": "juventud"}, {"LOWER": "y"}, {"LOWER": "deportes"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["economia", "economía"]}}, {"LOWER": "industria"}, {"LOWER": "y"}, {"LOWER": "comercio"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["educacion", "educación"]}}, {"LOWER": {"IN": ["publica", "pública"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["gobernacion", "gobernación"]}}, {"LOWER": "y"}, {"LOWER": {"IN": ["policia", "policía"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "hacienda"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "justicia"}, {"LOWER": "y"}, {"LOWER": "paz"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "presidencia"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "obras"}, {"LOWER": {"IN": ["publicas", "públicas"]}}, {"LOWER": "y"}, {"LOWER": "transportes"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["planificacion", "planificación"]}}, {"LOWER": "nacional"}, {"LOWER": "y"}, {"LOWER": {"IN": ["politica", "política"]}}, {"LOWER": {"IN": ["economica", "económica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "relaciones"}, {"LOWER": "exteriores"}, {"LOWER": "y"}, {"LOWER": "culto"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "salud"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguridad"}, {"LOWER": {"IN": ["publica", "pública"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "trabajo"}, {"LOWER": "y"}, {"LOWER": "seguridad"}, {"LOWER": "social"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vivienda"}, {"LOWER": "y"}, {"LOWER": "asentamientos"}, {"LOWER": "humanos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ambiente"}, {"LOWER": {"IN": ["energia", "energía"]}}, {"LOWER": "y"}, {"LOWER": "telecomunicaciones"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "museo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "arte"}, {"LOWER": "costarricense"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "museo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "arte"}, {"LOWER": "y"}, {"LOWER": "diseño"}, {"LOWER": "contemporaneo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "museo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "guanacaste"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "museo"}, {"LOWER": "dr."}, {"LOWER": "rafael"}, {"LOWER": {"IN": ["ángel", "ángel"]}}, {"LOWER": {"IN": ["calderon", "calderón"]}}, {"LOWER": "guardia"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "museo"}, {"LOWER": {"IN": ["historico", "histórico"]}}, {"LOWER": "cultural"}, {"LOWER": "juan"}, {"LOWER": {"IN": ["santamaria", "santamaría"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "museo"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "oficina"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["cooperacion", "cooperación"]}}, {"LOWER": "internacional"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "salud"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "oficina"}, {"LOWER": "ejecutora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "proyecto"}, {"LOWER": "turistico"}, {"LOWER": "golfo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "papagayo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "oficina"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "semillas"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "oficina"}, {"LOWER": "nacional"}, {"LOWER": "forestal"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "operadora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pensiones"}, {"LOWER": "complementarias"}, {"LOWER": "y"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["capitalizacion", "capitalización"]}}, {"LOWER": "laboral"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "caja"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguro"}, {"LOWER": "social"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "organo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["reglamentación", "reglamentación"]}}, {"LOWER": {"IN": ["tecnica", "técnica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "parque"}, {"LOWER": "marino"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pacifico"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "patronato"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "construcciones"}, {"LOWER": "instalaciones"}, {"LOWER": "y"}, {"LOWER": {"IN": ["adquisicion", "adquisición"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "bienes"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "patronato"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "ciegos"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "patronato"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "infancia"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "patronato"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["rehabilitacion", "rehabilitación"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "poder"}, {"LOWER": "ejecutivo"}]}, 
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "presidencia"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["republica", "república"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "poder"}, {"LOWER": "judicial"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "poder"}, {"LOWER": "legislativo"}, {"LOWER": "asamblea"}, {"LOWER": "legislativa"},]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "popular"}, {"LOWER": "valores"}, {"LOWER": "puesto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "bolsa"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["procuraduria", "procuraduría"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["republica", "república"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "programa"}, {"LOWER": "integral"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "mercadeo"}, {"LOWER": "agropecuario"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "promotora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "comercio"}, {"LOWER": "exterior"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "proyecto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": {"IN": ["agricola", "agrícola"]}}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "zona"}, {"LOWER": {"IN": ["atlantica", "atlántica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "proyecto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "agricola"}, {"LOWER": "peninsula"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "nicoya"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["radiográfica", "radiográfica"]}}, {"LOWER": "costarricense"}, {"LOWER": "s."}, {"LOWER": "a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "refinadora"}, {"LOWER": "costarricense"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["petroleo", "petróleo"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["secretaria", "secretaría"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["politica", "política"]}}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["alimentacion", "alimentación"]}}, {"LOWER": "y"}, {"LOWER": {"IN": ["nutricion", "nutrición"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "secretaria"}, {"LOWER": "ejecutiva"}, {"LOWER": "planificacion"}, {"LOWER": "sectorial"}, {"LOWER": "agropecuaria"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["secretaria", "secretaría"]}}, {"LOWER": {"IN": ["tecnica", "técnica"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "gobierno"}, {"LOWER": "digital"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": {"IN": ["secretaria", "secretaría"]}}, {"LOWER": {"IN": ["tecnica", "técnica"]}}, {"LOWER": "nacional"}, {"LOWER": "ambiental"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "servicio"}, {"LOWER": "fitosanitario"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "estado"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "servicio"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "aguas"}, {"LOWER": {"IN": ["subterraneas", "subterráneas"]}}, {"LOWER": "riego"}, {"LOWER": "y"}, {"LOWER": "avenamiento"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "servicio"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "guardacostas"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "servicio"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "salud"}, {"LOWER": "animal"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "sistema"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "emergencias"}, {"LOWER": "911"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "sistema"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["acreditacion", "acreditación"]}}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["educacion", "educación"]}}, {"LOWER": "superior"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "sistema"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "areas"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["conservacion", "conservación"]}}, {"LOWER": "incluye"}, {"LOWER": {"IN": ["direccion", "dirección"]}}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "vida"}, {"LOWER": "silvestre"}, {"LOWER": {"IN": ["administracion", "administración"]}}, {"LOWER": "forestal"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "estado"}, {"LOWER": "y"}, {"LOWER": "servicio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "parques"}, {"LOWER": "nacionales"},]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "sistema"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "radio"}, {"LOWER": "y"}, {"LOWER": {"IN": ["television", "televisión"]}}, {"LOWER": "cultural"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "sociedad"}, {"LOWER": "administradora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "fondos"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["inversion", "inversión"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "banco"}, {"LOWER": "popular"}, {"LOWER": "y"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "popular"}, {"LOWER": "s.a."}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "superintendencia"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "telecomunicaciones"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "superintendencia"}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "entidades"}, {"LOWER": "financieras"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "superintendencia"}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pensiones"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "superintendencia"}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "seguros"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "superintendencia"}, {"LOWER": "general"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "valores"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "teatro"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "teatro"}, {"LOWER": "popular"}, {"LOWER": "melico"}, {"LOWER": "salazar"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"LOWER": "administrativo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "transportes"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"LOWER": "administrativo"}, {"LOWER": "migratorio"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"LOWER": "aduanero"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"LOWER": "ambiental"}, {"LOWER": "administrativo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "carrera"}, {"LOWER": "docente"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "servicio"}, {"LOWER": "civil"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"LOWER": "fiscal"}, {"LOWER": "administrativo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"LOWER": "registral"}, {"LOWER": "administrativo"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "tribunal"}, {"LOWER": "supremo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "elecciones"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "unidad"}, {"LOWER": "coordinadora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "proyecto"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "mejoramiento"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "calidad"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["educacion", "educación"]}}, {"LOWER": "general"}, {"LOWER": {"IN": ["basica", "básica"]}}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "unidad"}, {"LOWER": "ejecutora"}, {"LOWER": "1030"}, {"LOWER": "banco"}, {"LOWER": "interamericano"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "ministerio"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "hacienda"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "unidad"}, {"LOWER": "ejecutora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["credito", "crédito"]}}, {"LOWER": "y"}, {"LOWER": "desarrollo"}, {"LOWER": {"IN": ["agricola", "agrícola"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pequeños"}, {"LOWER": "productores"}, {"IS_STOP": True, "OP":"+"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "zona"}, {"LOWER": "norte"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "unidad"}, {"LOWER": "ejecutora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "programa"}, {"LOWER": "ganadero"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "salud"}, {"LOWER": "animal"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "unidad"}, {"LOWER": "ejecutora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "programa"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["regularizacion", "regularización"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "catastro"}, {"LOWER": "y"}, {"LOWER": "registro"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "universidad"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "costa"}, {"LOWER": "rica"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "universidad"}, {"LOWER": "estatal"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "distancia"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "universidad"}, {"LOWER": "nacional"}]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "universidad"}, {"LOWER": {"IN": ["tecnica", "técnica"]}}, {"LOWER": "nacional"}, {"LOWER": "incluye"}, {"LOWER": "centro"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["investigacion", "investigación"]}}, {"LOWER": "y"}, {"LOWER": "perfeccionamiento"}, {"LOWER": "para"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["educacion", "educación"]}}, {"LOWER": {"IN": ["técnica", "técnica"]}}, {"LOWER": "centro"}, {"LOWER": "nacional"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["formacion", "formación"]}}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "formadores"}, {"LOWER": "y"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "personal"}, {"LOWER": {"IN": ["técnico", "técnico"]}}, {"LOWER": "para"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "desarrollo"}, {"LOWER": "industrial"}, {"LOWER": "colegio"}, {"LOWER": "universitario"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "alajuela"}, {"LOWER": "colegio"}, {"LOWER": "universitario"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "cartago"}, {"LOWER": "colegio"}, {"LOWER": "universitario"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "puntarenas"}, {"LOWER": "colegio"}, {"LOWER": "universitario"}, {"LOWER": "para"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "riego"}, {"LOWER": "y"}, {"LOWER": "desarrollo"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["tropico", "trópico"]}}, {"LOWER": "seco"}, {"LOWER": "escuela"}, {"LOWER": "centroamericana"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": {"IN": ["ganaderia", "ganadería"]}},]},
        {"label": "Entidad Pública", "pattern": [ {"LOWER": "vida"}, {"LOWER": "plena"}, {"LOWER": "operadora"}, {"IS_STOP": True, "OP":"+"}, {"LOWER": "pensiones"}, {"LOWER": "complementarias"}, {"LOWER": "s.a."}]}
    ]


    if personas==True:
        ner_tagger = CoreNLPParser('http://localhost:9003', tagtype='ner')

        tags = ner_tagger.tag(text.split())

        pers = []
        lastTag = ""
        for tok  in tags:
            token, tag = tok
            if tag == 'PERSON':
                if len(pers) != 0 and lastTag == 'PERSON':
                    pers[-1].append({"LOWER" : token.lower()})
                else:
                    pers.append([{"LOWER" : token.lower()}])
            lastTag = tag
        
        personas = [ {"label": "Persona", "pattern": p} for p in pers]   
        patterns = patterns+personas     
            
    global nlp
    try: 
        nlp
    except:
        nlp = spacy.load('es_core_news_lg')

        ruler = nlp.add_pipe("entity_ruler", first=True)
        ruler.add_patterns(patterns)
        nlp.add_pipe('Sentencias', first=True)
    
    doc = nlp(text, disable=["ner"])
        
    return doc
