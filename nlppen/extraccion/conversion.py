import os
import re
import bs4
import tika
from tika import parser
from pathlib import Path 
from tqdm.notebook import tqdm 
from pathlib import Path 
from joblib import Parallel, delayed
from sentencias.mongodb import MongoTxA, AllDataQuery

def procesar_archivos(PATH='.', collection='Digesto', db="SalaC"):
        """
        Extrae el texto de todos los archivos de forma recursiva y lo guarda en la base de datos de MongoDB. Se excluyen 
        """
        db = MongoTxA(AllDataQuery(), collection=collection, db=db)
        tika.initVM()
        excludeExt = ['.emz',  '.wmf', '.wmz', '.gi~', '.tmp', '.db','.lnk','.thmx', '.bk','.xps', '.rf', '.xml', '.rcv', '.gif', '.jpg', '.dpj','.rt_', '.png', '.mso', '.xml' ]
        # Se borra la colección
        dir_path = Path(PATH)
        work_path = os.getcwd()
        delPointRegExp = re.compile (r'\.') # Char code inválido

        archivos = []
        result = [file for file in list(dir_path.rglob("*.*")) if file.suffix.lower() not in excludeExt ]
        for file in tqdm(result):
            dict = {'error': {}}
            dict['anno'] = file.parts[3]
            dict['archivo'] = file.parts[-1]
            dict['path'] = str(file)
            dict['extension'] = file.suffix.lower()
            if len(file.parts) > 4:
                dict['categoria'] =  file.parts[4]
            try:
                with open(work_path + "/" + str(file),encoding="latin-1") as f:
                    no_extracted = f.read()
                    f.seek(0)
                    s = bs4.BeautifulSoup(f)
                    bs_text = ''
                    for item in s.find_all(lambda tag: (tag.name == 'p' and tag.find_parent(['p','span']) == None) 
                                           or (tag.name == 'span' and tag.find_parent(['p','span']) == None)):
                        bs_text = bs_text + item.text.replace('\n',' ') + '\n'
                        
                    dict['texts'] = {
                        'bs4': bs_text.replace(u'\xa0',''),
                        'html': no_extracted
                        }
            except Exception as err:
                dict['texts'] = {'bs4': None,'html': None}
                dict['error']['conversion_bs4'] = str(err)

            try:
                parsed =  parser.from_file(work_path + "/" + str(file))

                dict['texts'] = {
                    'tika': parsed['content'],
                    'bs4': bs_text.replace(u'\xa0',''),
                    'html': no_extracted
                    }

                dict['metadata'] = {}
                for key in parsed['metadata']:
                    newKey = delPointRegExp.sub('-', key)
                    dict['metadata'][newKey] = parsed['metadata'][key]

                dict['status'] = parsed['status']
            except Exception as err:
                dict['texts'] = {'tika': None}
                dict['error']['conversion_tika'] = str(err)
                print ("Unexpected error:", err)

            archivos.append(dict) 
        if archivos != []:
            db.insert_many(archivos) # La actualización a la base de datos se realiza por año


def procesar_archivos_parallel(PATH, collection='Digesto', db="SalaC", workers = 24):
    dirlist = [str(f) for f in Path(PATH).iterdir() if f.is_dir()]
    Parallel(n_jobs=workers)(delayed(procesar_archivos)(PATH=dirl, collection=collection, db=db) for dirl in dirlist)