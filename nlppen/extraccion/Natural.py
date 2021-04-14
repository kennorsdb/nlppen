# Para Multiprocessing
from multiprocessing import Manager
import pickle

# Para NLP
import re
from .utils.Txt2Numbers import Txt2Numbers
from datetime import datetime
from .ProcessException import ProcessException
from nltk.tag.stanford import StanfordPOSTagger
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem.snowball import SnowballStemmer
import pattern.es as lem
from cucco import Cucco
from cucco.config import Config
import ftfy
from .utils.spacy_entities import nlp_corte
import bs4

# Para Machine Learning
import os
import nltk
import numpy as np
from numpy import int32
from bson.binary import Binary
import spacy

os.environ['JAVAHOME'] = 'C:/Program Files/Java/jre1.8.0_192/bin'


class Natural:
    def __init__(self, debug=False):
        self.nlp = None
        self.debug = debug
        self.vocabulary = None
        #stagger = StanfordPOSTagger("spanish.tagger", '../lib/stanford-postagger/stanford-postagger.jar')
        self.stemmerEsp = SnowballStemmer('spanish')
        self.hunspell = None

        # Limpiar el texto de texto sobrante
        self.limpiarExp2 = re.compile(r'[\r\n]{3,}', re.M)
        self.limpiarExp3 = re.compile(r'', re.M)  # Char code inválido
        self.limpiarExp4 = re.compile(r'(\n\s\n)', re.M)
        self.limpiarExp5 = re.compile('([^\.:;])\n', re.I | re.M) # Unir párrafos *Experimental*


        # RegExp para separar la resolución en sus partes
        self.partesExp = re.compile(r"""(?:(?P<encabezado>(?s:.*?))(?=(?i:resultando)|(?i:considerando?\s*\n)|(?i:(?:-|\n)\s*por\ tanto)))   # Match al encabezado
                                (?:(?i:resultando):?(?P<resultando>(?s:.*?))(?=(?i:(?:-|\n)\s*por\ tanto)|(?i:considerando:?\s*\n)))?
                                (?:(?i:considerando):?\s*\n(?P<considerando>(?s:.*?))(?=(?i:resultando)|(?i:(?:-|\n)\s*por\ tanto)))?
                                (?:(?i:(?:-|\n)\s*por\ tanto):?(?P<portanto>(?s:.*)))?""", re.X | re.M)
        self.excepcionPartes = re.compile(
            r'^[ \tA-Z]+[A-Z]+[ \tA-Z]+$', re.M | re.U)

        # REgExp para obtener el número de expediente
        self.expediente = re.compile(r"""(?:\»?\??[\ \t]*(?i:Expediente)|(?i:Exp)|(?i:Amparo))
                                            [ \t]*:?[\ \t]*[\n]?\.?[ ]?N?(o|º|°)?\§?\.?\(?\§?(N\.\(?)?[\ \t]*
                                            (?P<expediente>[0-9]\S+)
                                            (\.?|\ +?|(?i:VOTO)|(?i:recurrente)|No)""", re.X | re.M | re.U)
        self.expediente2 = re.compile(
            r"""(?P<expediente>[0-9]{3,4}-[a-zA-Z]-[0-9]{2})""", re.X | re.M | re.U)
        self.expediente3 = re.compile(
            r"""(?P<expediente>[0-9]{2}.?[0-9]{6}.?[0-9]{4}.?[a-zA-Z]{2})""", re.X | re.M | re.U)
        self.expediente4 = re.compile(r"""[ -]""", re.X | re.M | re.U)

        # RegExp para la extracción de fecha y hora
        self.buscarfechahora = re.compile(r"""(?:(?:las)|(?:12s))[\s]+(?P<horas>[áéíóú\w\s]+?)[\s]+horas[,]?[\s]+(?:(?:y[\s])|)(?:(?:con[\s])|)(?:(?:(?P<minutos>[áéíóú\w\s]+?)
                                    [\s]+minutos?)|)[\s]*(?:(?:de[l]?)|)[\s]+(?:d[íi]a[\s]+)?
                                    (?P<dias>[áéíóú\w\s]*?)[\s]+de[\s]+(?P<mes>\w+)[\s]*(?:(?:de[l]?)|)[\s]*(?:(?:año)|)[\s]+(?:(?:un[\s])|)
                                    (?P<annoHigh>(?:(?i:mil[\s]+novecientos)|(?i:dos\s+mil)))[\s]*(?:(?P<annoLow>[áéíóú\w\s]+?)|)[\s]*\.""", re.M | re.X)

        # RegExp para la extracción del redacto
        self.redactorExp = re.compile(
            r"""Redacta.*magistrad[oa]\s+(?P<redactor>.+?)[;,\s\.]{2,}""",  re.X | re.M | re.I)

        # RegEx para lasentencia
        self.conLugarParcial = re.compile(r"""(?:con\s+lugar\s+parcial)""",  re.X | re.M | re.I)
        self.conLugar = re.compile(r"""(?:con\s+lugar)""",  re.X | re.M | re.I)
        self.sinLugar = re.compile(
            r"""(?:sin\s+lugar)|(?:No\s+ha\s+lugar)""",  re.X | re.M | re.I)
        self.rechazaPorElFondo = re.compile(
            r"""(?:rechaza\s+por\s+el\s+fondo)""",  re.X | re.M | re.I)
        self.rechazaDePlano = re.compile(
            r"""se\s+rechaza\s+de\s+plano""",  re.X | re.M | re.I)
        self.archivese = re.compile(
            r"""(?:arch[IÍií]vese)|((?:archivo))""",  re.X | re.M | re.I)
        self.correccionDisciplinaria = re.compile(
            r"""(?:correccion\s+disciplinaria)""",  re.X | re.M | re.I)
        self.estese = re.compile(
            r"""(?:est[eé]se)|(?:est[eé]nse)""",  re.X | re.M | re.I)
        
        # Actualmente se usa una única expresión regular para la setnencia
        self.sentenciaReg = re.compile(r"""(?P<Con_lugar_parcial>(?:parcialmente\s+con\s+lugar)|(?:con\s+lugar\s+parcial))
                      |(?P<Con_lugar>(?:con\s+lugar))
                      |(?P<Sin_lugar>(?:sin\s+lugar)|(?:No\s+ha\s+lugar))
                      |(?P<Rechazo_por_el_fondo>(?:rechaza\s+por\s+el\s+fondo))
                      |(?P<Rechazo_de_plano>(?:se\s+rechaza\s+de\s+plano))
                      |(?P<Archívese>(?:arch[IÍií]vese)|(?:archivo))
                      |(?P<Estese>(?:est[eé]se)|(?:est[eé]nse))
                      |(?P<Inevacuable>(?:inevacuable))""",  re.X | re.M | re.I)
        

        # RegEx para extraer el tipo desentencia
        self.habeascorpus = re.compile(
            r"""(?:h[áÁaA]beas\s+corpus)""",  re.X | re.M | re.I)
        self.recursoAmparo = re.compile(
            r"""recurso\s+de\s+amparo""",  re.X | re.M | re.I)
        self.accionIncost = re.compile(
            r"""acci[oOóÓ]n\s+de\sinconstitucionalidad""",  re.X | re.M | re.I)
        self.consultaFacultativa = re.compile(
            r"""consulta\s+facultativa""",  re.X | re.M | re.I)
        self.consultaLegislativa = re.compile(
            r"""consulta\s+legislativa""",  re.X | re.M | re.I)
        self.gestionInejecucion = re.compile(
            r"""gesti[óÓoO]n\s+de\s+inejecuci[óÓoO]n""",  re.X | re.M | re.I)
        self.consultaPreceptiva = re.compile(
            r"""consulta\s+legislativa\s+preceptiva""",  re.X | re.M | re.I)
        self.consultaPrescriptiva = re.compile(
            r"""consulta\s+legislativa\s+prescriptiva""",  re.X | re.M | re.I)
        self.libertadAgraviado = re.compile(
            r"""libertad\s+del\s+agraviado""", re.M | re.I)
        self.apremioCorporal = re.compile(
            r"""apremio\s+corporal""", re.M | re.I)
        self.desestimientos = re.compile(r"""desestimiento""", re.M | re.I)
        self.suspensionTramites = re.compile(
            r"""suspensi[oOóÓ]n\s+de\s+tr[áaÁA]mites""", re.M | re.I)
        self.suspensionActoImpugnado = re.compile(
            r"""suspensi[oOóÓ]n\s+del\s+acto\s+impugnado""", re.M | re.I)
        self.ejecucionActoImpugnado = re.compile(
            r"""ejecuci[oOóÓ]n\s+del\s+acto\s+impugnado""", re.M | re.I)
        self.archivoExpediente = re.compile(
            r"""archivo\s+del\s+expediente""", re.M | re.I)

        # RegEx para limpiar espacios en blanco
        self.limpiarspaciosRegex = re.compile("\s+", re.M)

        # RegExp Voto Salvado      
        self.votoSalvadoReg1 = re.compile(r'salvan?\s+el\s+voto', re.I | re.M | re.X)
        self.votoSalvadoReg2 = re.compile(r'salv[oó]\s+el\s+voto', re.I | re.M | re.X)
        self.votoSalvadoReg3 = re.compile(r'voto\s+salvado', re.I | re.M | re.X)
        
    def preprocesar(self, doc):
        try:
            # Se revisa si existe el texto en BS4, si no se usa tika
            if 'bs4' in doc['texts'] and doc['texts']['bs4'] != '':
                raw = doc['texts']['bs4']
            elif 'tika' in doc['texts'] and doc['texts']['tika'] != '':
                raw = doc['texts']['tika']
            else:
                raise ProcessException('Preprocesar', 'No se se extrajo el texto')

            # limpia el texto
            raw = self.limpiarExp2.sub(r"\n", raw)
            raw = self.limpiarExp4.sub(r"\n", raw)
            raw = self.limpiarExp3.sub("", raw)
            raw = raw.replace(u'\xa0', u' ')
            raw = re.sub(r'\s+\n', r'\n', raw)
            raw = self.limpiarExp5.sub('\g<1> ',raw) # Univer párrafos.
            return {'txt': raw}

        except Exception as err:
            raise ProcessException('Preprocesar', str(err))

    def secciones(self, doc):
        # Se separa el documento en secciones
        result = self.partesExp.match(doc["txt"])

        # Excepción si no se encuentra la estructura básica
        if result:
            return {'secciones': result.groupdict()}
        else:
            raise ProcessException(
                "secciones", "No se encuentran las secciones en el documento")

    def expedientes(self, doc):
        expediente = self.expediente2.search(doc["txt"])
        if not expediente:
            expediente = self.expediente3.search(doc["txt"])
        if not expediente:
            expediente = self.expediente.search(doc["txt"])

        # Si no se encuentra en txt, se busca en la extracción de tika
        if not expediente:
            if 'texts' in doc and 'tika' in doc['texts'] and doc["texts"]['tika'] is not None:
                expediente = self.expediente2.search(doc["texts"]['tika'])
                if not expediente:
                    expediente = self.expediente3.search(doc["texts"]['tika'])
                if not expediente:
                    expediente = self.expediente.search(doc["texts"]['tika'])        

        if not expediente:
            raise ProcessException("extraccion_expediente", "No se ubica número de expediente")

        # Se limpia y se normaliza el número de expediente
        result = self.expediente4.sub("", expediente.groupdict()['expediente'])

        return {'extraccion.expediente': result}

    def fechahora(self, doc):
        txt2num = Txt2Numbers()
        result = self.buscarfechahora.search(doc['txt'])

        if not result:
            raise ProcessException(
                "extraccion_fechahora", "No se reconoce fecha y hora")

        try:
            data = result.groupdict()
            if self.debug:
                print(data)

            year = txt2num.number(
                data['annoHigh']) * 100 + (txt2num.number(data['annoLow']) if data['annoLow'] else 0)
            d = datetime(year=year, month=txt2num.number(data['mes']), day=(txt2num.number(data['dias']if data['dias'] else 0)),
                            hour=txt2num.number(data['horas']), minute=(txt2num.number(data['minutos']) if data['minutos'] else 0))
            if self.debug:
                print(year)

            return {'extraccion.fechahora': d}

        except Exception as err:
            raise ProcessException("extraccion_fechahora", str(err))

    def redactor(self, doc):
        result = self.redactorExp.search(doc['txt'])
        if self.debug:
            print(result.groupdict())

        if not result:
            raise ProcessException(
                "extraccion_redactor", "No se encuentra el redactor")

        return { "extraccion.redactor" : result.groupdict()['redactor'] }

    def sentencia(self, doc):
        res = None
        try:
            if "secciones" in doc and 'portanto' in doc["secciones"]and doc["secciones"]["portanto"] is not None:

                match = [r.groupdict() for r in self.sentenciaReg.finditer(doc["secciones"]['portanto'])]
                if len(match) != 0:
                    #Se extrae el nombre de los grupos encontrados
                    sentencias_keys = [ d.replace('_',' ') for r in match for d in r.keys() if r[d] is not None]
                    res = { 'extraccion.sentencia': sentencias_keys[0], 'extraccion.sentencias_varias': sentencias_keys  }
                else:
                    raise ProcessException(
                        "extraccion_sentencia", "No se encuentra la sentencia en el portanto")
            else:
                raise ProcessException("extraccion_sentencia", 
                        "No se ha identificado el portanto con anterioridad en el documento")
        except ProcessException as err:
            raise err
        except KeyError as err:
            raise ProcessException(
                "extraccion_sentencia", 'El documento no tiene el campo: ' + str(err))
        except Exception as err:
            raise ProcessException("extraccion_sentencia", str(err))
        else:
            return {'extraccion.sentencia': res}

    def tipo_proceso(self, doc):
        res = {}
        if "secciones" in doc and 'encabezado' in doc["secciones"]and doc["secciones"]["encabezado"] is not None:
            if self.habeascorpus.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Hábeas Corpus"}
            elif self.accionIncost.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Acción de Inconstitucionalidad"}
            elif self.recursoAmparo.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Recurso de Amparo"}
            elif self.consultaFacultativa.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Consulta Facultativa"}
            elif self.self.consultaPreceptiva.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Consulta Preceptiva"}
            elif self.self.consultaLegislativa.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Consulta Legislativa"}
            elif self.consultaPrescriptiva.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Consulta Prescriptiva"}
            elif self.libertadAgraviado.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Libertad del Agraviado"}
            elif self.apremioCorporal.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Arraigo Corporal"}
            elif self.desestimientos.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Desestimientos"}
            elif self.suspensionTramites.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Suspención de Trámites"}
            elif self.suspensionActoImpugnado.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Suspención del Acto Impugnado"}
            elif self.ejecucionActoImpugnado.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Ejecución del Acto Impugnado"}
            elif self.archivoExpediente.search(doc["secciones"]["encabezado"] ) is not None:
                res = {'extraccion.proceso': "Archivo del Expediente"}
            else:
                raise ProcessException(
                    "extraccion_tipo_proceso", "No se encuentra el tipo de proceso")
        else:
            raise ProcessException("extraccion_tipo_proceso", "No existe la sección de encabezado")

        return res
    

    def voto_salvado(self, doc):
        if "secciones" in doc and 'portanto' in doc["secciones"]and doc["secciones"]["portanto"] is not None:
            if self.votoSalvadoReg1.search(doc["secciones"]["portanto"] ) is not None:
                res = {'extraccion.voto_salvado': True}
            elif self.votoSalvadoReg2.search(doc["secciones"]["portanto"] ) is not None:
                res = {'extraccion.voto_salvado': True}
            elif self.votoSalvadoReg3.search(doc["secciones"]["portanto"] ) is not None:
                res = {'extraccion.voto_salvado': True}
            else:
                res = {'extraccion.voto_salvado': False}
        else:
            raise ProcessException("extraccion_voto_salvado", "No existe la sección de 'por tanto'")

        return res
    

    def lematizacion(self, doc):
        # if self.hunspell is None:
        #     self.hunspell = Hunspell('es_ANY', hunspell_data_dir='./hunspell-es/')
        # Mojibake
        mojTxt = ftfy.fix_encoding(doc['txt'])

        # # Autocorrector
        # def spelling(w):
        #     correct =  self.hunspell.suggest(w)
        #     return correct[0] if len(correct) > 0 else w

        # Se eliminan los números
        onlyLetters = re.compile("[^A-zÁÉÍÓÚáéíóú]").sub(' ', mojTxt)

        if self.debug:
            print(onlyLetters)

        # Normalización
        cucco = Cucco(config=Config(language='es'))
        normsCucco = ['remove_stop_words',
                        'replace_punctuation', 'remove_extra_whitespaces']
        normTxt1 = cucco.normalize(onlyLetters, normsCucco)

        if self.debug:
            print(normTxt1)

        # Lematización
        # lemasTxt = lem.Sentence(lem.parse(onlyLetters, lemmata=True)).lemmata
        if self.debug:
            print(lemasTxt)

        # Normalización
        cucco = Cucco(config=Config(language='es'))
        normsCucco = ['remove_stop_words']
        normTxt2 = cucco.normalize(normTxt1, normsCucco)

        if self.debug:
            print(normTxt2)

        # Stemming
        stem = [self.stemmerEsp.stem(word) for word in word_tokenize(normTxt2)]
        
        if self.debug:
            print(stem)

        return {'texts.lemma': ' '.join(stem)}


    def entidades(self, doc):

        options = {
            'ents': None,
                'colors' : {
                'CANTÓN' : 'linear-gradient(to top, #0ba360 0%, #3cba92 100%)',
                'FECHAHORA' : 'linear-gradient( to bottom, #FDEB71 10%, #F8D800 100%)',
                'FECHA' : 'linear-gradient( to bottom, #ABDCFF 10%, #0396FF 100%)',
                'RECURRIDO' : 'linear-gradient( to bottom, #FF96F9 10%, #C32BAC 100%);',
                'RECURRENTE' : 'linear-gradient( to bottom, #CE9FFC 10%, #7367F0 100%)',
                'ENTIDAD PÚBLICA' : 'linear-gradient( to bottom, #90F7EC 10%, #32CCBC 100%)',
                'REDACTOR' : 'linear-gradient( to bottom, #81FBB8 10%, #28C76F 100%)',
                'CONSTITUCIÓN' : 'linear-gradient( to bottom, #E2B0FF 10%, #9F44D3 100%)',
                'MAGISTRADO' : 'linear-gradient( to bottom, #FCCF31 10%, #F55555 100%)',
                'REGLAMENTO' : 'linear-gradient( to bottom, #FFF720 10%, #3CD500 100%)',
                'LEY' : 'linear-gradient( to top, #FD6E6A 10%, #FFC600 100%)',
                'DECRETO' : 'linear-gradient( to bottom, #3C8CE7 10%, #00EAFF 100%)'
                } 
        }

        entidades = {}
        spcy =  nlp_corte(doc['txt'], personas=False)
        for ent in spcy.ents:
            if ent.label_ not in entidades:
                entidades[ent.label_] = [ent.text]
            else:
                entidades[ent.label_].append(ent.text)

        html = spacy.displacy.render(spcy, style="ent", options=options, page=True, jupyter=False)    

        return {'entidades':  entidades, 'texts.entidades_html': html}

    def fecha_Clasf(self, doc):
        # rexp = re.compile(r'(?P<mes>[0-9]+)\/(?P<dia>[0-9]+)\/(?P<anno>[0-9]+)')
        # result = rexp.search(doc['fecha'])
        # return result.groupdict() 

        result = doc['expediente'].replace('-','')
        return {'expediente': result}
    
    
    def buscarTerminos(self, doc):
        if self.terminos == None or self.terminos == []:
            assert "No se han especificado términos"

        res = {}
        for key in self.terminos:
            for term in self.terminos[key]:
                reg = re.compile(term,  re.X | re.M | re.I)
                resultado = reg.findall(doc['txt'])
                if key not in res:
                    res[key] = 0
                res[key] += len(resultado)

        return {'keywords' : res}
    
    
    def extraer_magistrados(self, doc, debug=False): 
        if 'texts' not in doc or 'html' not in doc['texts']:
            raise ProcessException("extraccion_magistrado_bs4", "No existe la fuente html en el documento")
            
        s = bs4.BeautifulSoup(doc['texts']['html'])

        st = ''
        for item in s.find_all('p'):
            st = st + item.text.replace('\n',' ') + '\n'

        ps = {'magistrados': []}
        ptanto_lst = s.find_all(string=re.compile('\s*Por\s+tanto\:?\s*',  re.I ))

        if debug:
            print()
            print('Se encuentra el Por tanto =>')

        # Si existe un por tanto, buscamos a partir de este elemento
        if len(ptanto_lst) > 0:
            ptanto = ptanto_lst[-1].parent
        else:
            ptanto = s.find('body')

        #····························································································    
        # Estrategia 1: Tablas al final del documento    
        # Se prueba buscando las tablas que hay en el documento
        if ptanto is not None:
            for st in ptanto.find_all('table'):
                for row in st.find_all_next('tr'):
                    for p in row.find_all('p'):
                        strip = p.text.strip()
                        if strip != '' and re.search('(?:\s*president.*)', strip, re.I) is None :
                            ps['magistrados'].append(p.text.strip().replace('\n',''))

        #····························································································
        #Estrategia 2: Mediante la palabra 'presidente'
        if len(ps['magistrados']) == 0 and ptanto is not None:
            # Si no hay por Tanto, nos atrevemos a buscar en todo el documento

            presidente = [x for x in ptanto.find_all_next(text=re.compile('.*pr?esident[ea].*',re.I))]
            
            if len(presidente) > 0 :  
                mag = presidente[-1]

                person1 = mag.find_previous(string=re.compile('[^\sA-z]+'))

                if person1 is not None:
                    parents1 = list(mag.parents)[::-1] 
                    parents2 = list(person1.parents)[::-1]  

                    if len(parents1) != 0 or len(parents2) != 0:

                        # Buscamos el ancestro común más cercano
                        while parents1[0] == parents2[0]:
                            parents1.pop(0)
                            parents2.pop(0)
                            if len(parents1) == 0 or len(parents2) == 0:
                                raise ProcessException("extraccion_magistrado_bs4", "Error: No se encontró a los magistrados firmantes")

                        # Se extrae el texto de todos los "hermanos xml" 
                        ps['magistrados'] = [person1.string] 
                        for s in parents1[0].next_siblings:
                            if debug:
                                print()
                                print(repr(s))

                            if (not isinstance(s, bs4.NavigableString) and 
                                re.match(r'^[A-záéíóúÁÉÍÓÚÑñ\. \t\s]+$', s.get_text().strip()) is not None):

                                if debug:    
                                    print(s.get_text()+'\n')

                                ps['magistrados'].append(s.text.strip().replace('\xa0',' '))

                                lst = [re.split('\t+| {2,}' ,d) for d in ps['magistrados']]
                                ps['magistrados'] = [t.strip() for sublst in lst for t in sublst]

                    
        if len(ps['magistrados']) == 0:
            raise ProcessException("extraccion_magistrado_bs4", "Error: No se encontró a los magistrados firmantes") 

        return {'extraccion.magistrados': ps['magistrados']}

    
    