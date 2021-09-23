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
    Extrae el texto de todos los archivos de forma recursiva y 
    lo guarda en la base de datos de MongoDB. Se excluyen 
    """
    db = MongoTxA(AllDataQuery(), collection=collection, db=db)
    tika.initVM()
    excludeExt = ['.emz',  '.wmf', '.wmz', '.gi~', '.tmp',
                  '.db', '.lnk', '.thmx', '.bk', '.xps',
                  '.rf', '.xml', '.rcv', '.gif',
                  '.jpg', '.dpj', '.rt_', '.png', '.mso',
                  '.xml']
    # Se borra la colección
    dir_path = Path(PATH)
    work_path = os.getcwd()
    delPointRegExp = re.compile(r'\.') # Char code inválido

    archivos = []
    result = [file for file in list(dir_path.rglob("*.*")) if file.suffix.lower() not in excludeExt ]
    for file in tqdm(result):
        dct = {'error': {}}
        dct['anno'] = file.parts[4]
        dct['archivo'] = file.parts[-1]
        dct['path'] = str(file)
        dct['extension'] = file.suffix.lower()

        if len(file.parts) > 4:
            dct['categoria'] = file.parts[3]


        try:
            with open(work_path + "/" + str(file), 'rb') as f:
                # Se valida que el archivo es utf-8
                file_data = f.read()
                valid_utf8 = True
                encoding = 'utf-8'
                try:
                    file_data.decode('utf-8')
                except UnicodeDecodeError:
                    valid_utf8 = False
                    encoding = 'latin1'

                no_extracted = (f.read().decode(encoding, "backslashreplace"))
                f.seek(0)
                s = bs4.BeautifulSoup(f, "lxml", from_encoding=encoding)
                bs_text = ''
                for item in s.find_all(lambda tag: (tag.name == 'p' and tag.find_parent(['p','span']) == None) 
                                       or (tag.name == 'span' and tag.find_parent(['p','span']) == None)):
                    bs_text = bs_text + item.text.replace('\n', ' ') + '\n'

                dct['texts'] = {
                    'bs4': bs_text.replace(u'\xa0',''),
                    'html': no_extracted
                    }
        except Exception as err:
            dct['texts'] = {'bs4': None,'html': None}
            dct['error']['conversion_bs4'] = str(err)

        try:
            parsed = parser.from_file(work_path + "/" + str(file))

            dct['texts'] = {
                'tika': parsed['content'],
                'bs4': bs_text.replace(u'\xa0',''),
                'html': no_extracted
                }

            dct['metadata'] = {}
            for key in parsed['metadata']:
                newKey = delPointRegExp.sub('-', key)
                dct['metadata'][newKey] = parsed['metadata'][key]

            dct['status'] = parsed['status']
        except Exception as err:
            dct['texts'] = {'tika': None}
            dct['error']['conversion_tika'] = str(err)

        archivos.append(dct)

    if archivos != []:
        # La actualización a la base de datos se realiza por año
        db.insert_many(archivos)



def procesar_archivos_parallel(PATH, collection='Digesto', db="SalaC", workers = 24):
    dirlist = [str(f) for f in Path(PATH).iterdir() if f.is_dir()]
    Parallel(n_jobs=workers)(delayed(procesar_archivos)(PATH=dirl, collection=collection, db=db) for dirl in dirlist)