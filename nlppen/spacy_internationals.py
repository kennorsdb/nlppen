from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.language import Language
from nltk.parse import CoreNLPParser
import spacy

def extractInternational(text):  
    global nlp
    try:
        nlp
    except:
        nlp = spacy.load('es_core_news_lg')
        (nlp.add_pipe("entity_ruler", name='inst_publicas', first=True, validate=True)
            .from_disk("./nlppen/instrumentos_internacionales.jsonl"))

    doc = nlp(text, disable=["ner"])
    return doc
