import spacy
import re

import pandas as pd

from pyspark.sql import Row
from pyspark.sql.types import IntegerType

POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',
            'X', 'SPACE']


def solo_considerando(txt):
    # RegExp para separar la resolución en sus partes
    partesExp = re.compile(r"""(?:(?P<encabezado>(?s:.*?))(?=(?i:resultando)|(?i:considerando?\s*\n)|(?i:(?:-|\n)\s*por\ tanto)))
                            (?:(?i:resultando):?(?P<resultando>(?s:.*?))(?=(?i:(?:-|\n)\s*por\ tanto)|(?i:considerando:?\s*\n)))?
                            (?:(?i:considerando):?\s*\n(?P<considerando>(?s:.*?))(?=(?i:resultando)|(?i:(?:-|\n)\s*por\ tanto)))?
                            (?:(?i:(?:-|\n)\s*por\ tanto):?(?P<portanto>(?s:.*(?=(?i:(?:-|\n)\s*por\ tanto)))))?""", re.X | re.M)
    res = partesExp.search(txt)
    if res is not None and res.group('considerando') is not None:
        return res.group('considerando')
    else:
        return ''


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
                txt = re.sub(term, key.replace(" ", ""), txt, re.I | re.M | re.X)

    return txt, terms


def spark_get_spacy(lang):
    global nlp
    if "nlp" in globals():
        return nlp
    else:
        nlp = spacy.load(lang)
        return nlp


def spark_buscar_terminos_doc(row, terminos, col='txt'):
    if terminos == None or terminos == []:
        assert "No se han especificado términos"

    res = row.asDict()
    tiene_terminos = False
    for key in terminos:
        for reg in terminos[key]:
            resultado = reg.findall(row[col])
            if key not in res:
                res[key] = 0
            tiene_terminos = len(resultado) != 0 or tiene_terminos
            res[key] += len(resultado)

    if tiene_terminos:
        return Row(**res)
    else:
        return None


def spark_extraer_tokens(batch, index_col='index', txt_col='txt', incluir=[], cambios={}):
    nlp = spark_get_spacy('es_core_news_lg')
    limpiar_regex = re.compile(r'[^A-Za-záéíóúüïÁÉÍÓÚÜÏñÑ]')

    if incluir == []:
        incluir = POS_TAGS

    for df in batch:
        palabras = []
        for indx, row in df.iterrows():
            txt = solo_considerando(row[txt_col])
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


def spark_skipgrams(batch, n=3, k=1, filtro=[], cruce='index', incluir=['PROPN', 'NOUN', 'VERB', 'ADJ'], cambios=None):
    from nltk import skipgrams
    import re

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
            sents = [skipgrams(sent, n, k)
                     for sent in doc.sents if filtro_oraciones(sent, filtro)]
            for sk in sents:
                res += [{**{'cruce': row[cruce]}, **{f't{tn}': '__'.join([limpiar(t.lower_), t.pos_])
                         for tn, t in enumerate(skipgram) if t.pos_ in incluir and limpiar(t.lemma_) != ''}}
                        for skipgram in sk]

        df = pd.DataFrame(res).dropna().reset_index().groupby(
            ['t'+str(tn) for tn in range(n)]+['cruce']).count()

        yield df.reset_index().rename(columns={'index': 'freq'})



        
