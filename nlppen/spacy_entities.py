from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.language import Language
from nltk.parse import CoreNLPParser
import spacy
import re

def nlp_test(text, personas=False):
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
        (nlp.add_pipe("entity_ruler", name='inst_publicas', first=True, validate=True)
            .from_disk("/home/jovyan/Work/ej/paquetes/nlppen/nlppen/instituciones_publicas_v2.jsonl"))

    doc = nlp(text, disable=["ner"])

    return doc
