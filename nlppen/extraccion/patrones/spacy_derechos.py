from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.language import Language
from nltk.parse import CoreNLPParser
import spacy

def extractDerechos(text):  

    global nlp

    try:
        nlp
    except:
        nlp = spacy.load('es_core_news_lg')
        nlp.remove_pipe("ner")
        nlp.add_pipe("ner", source=spacy.load('es_core_news_lg'))
        (nlp.add_pipe("entity_ruler", before="ner" , name='inst_publicas', validate=True)
            .from_disk("./nlppen/extraccion/patrones/jsonl/derechos.jsonl"))
    
    doc = nlp(text)
    return doc
