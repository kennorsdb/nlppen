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
        { "label": "Cantón", "pattern" : [ { "LOWER": 'zarcero}'} ]}
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

        ruler1 = nlp.add_pipe("entity_ruler", first=True)
        ruler1.add_patterns(patterns)
        ruler2 = nlp.add_pipe("entity_ruler", name='inst_publicas', first=True, validate=True).from_disk("./instituciones_publicas_v2.jsonl")

    doc = nlp(text, disable=["ner"])

    return doc
