import re
from copy import deepcopy
import json
import os

from pyspark.sql.functions import desc

from .spark_udfs import *


class Analisis:

    sdf = None
    wdf = None

    def __init__(self, terminos, spark, parquet_path='../../datasets/complete', cambios_path='./cambios.json', datasets_path='./datasets'):
        self.terminos = terminos
        self.spark = spark
        self.parquet_path = parquet_path
        self.datasets_path = datasets_path

        if os.path.exists(cambios_path):
            with open(cambios_path) as f:
                self.cambios_config = json.load(f)
        else:
            self.cambios_config = {}

        if not os.path.exists(datasets_path):
            os.makedirs(datasets_path)

    def filtrar_sentencias(self, parquet_file='terminos.parquet'):
        parquet_path = self.datasets_path + '/' + parquet_file
        if os.path.exists(parquet_path):
            self.sdf = self.spark.read.parquet(parquet_path)
        else:
            self.__busqueda_terminos()

            if self.sdf is not None:
                self.sdf.write.parquet(parquet_path)

        return self.sdf

    def tabla_resumen(self):
        terminos_cols = [col.replace(' ', '_') for col in self.terminos.keys()]
        ldf = self.sdf.select(*terminos_cols).toPandas()

        total_terminos = ldf.sum(axis=0)
        docs_encontrados = ldf.apply(lambda x: x != 0).sum(axis=0)
        df = pd.DataFrame([total_terminos, docs_encontrados]).T
        df.columns = ['Total de términos encontrados',
                      'Documentos encontrados']
        df.index = [term.replace('_', ' ') for term in df.index]
        return df

    def tabla_coocurrencia(self):
        terminos_cols = [col.replace(' ', '_') for col in self.terminos.keys()]
        ldf = self.sdf.select(*terminos_cols).toPandas()

        ldf[ldf != 0] = 1
        df = ldf.T.dot(ldf)
        df.columns = [term.replace('_', ' ') for term in df.columns]
        df.index = [term.replace('_', ' ') for term in df.index]
        return df

    def __busqueda_terminos(self):
        sdf = self.spark.read.parquet(self.parquet_path)
        schema = deepcopy(sdf.schema)

        term_regex = {}
        for cat, lst in self.terminos.items():
            col_name = cat.replace(' ', '_')
            term_regex[col_name] = [re.compile(r'\s+' + t.replace(' ', r'[\s\.\,\-\)\;\:\]]+'),  re.X | re.M | re.I)
                                    for t in lst]
            schema.add(col_name, 'integer', True)

        self.sdf = (sdf.rdd
                    .map(lambda row: spark_buscar_terminos_doc(row, term_regex))
                    .filter(lambda d: d is not None)
                    .toDF(schema=schema)
                    .persist()
                    )
        return self.sdf

    def sub_busqueda(self, terminos_sub, actualizar_sdf=False):
        schema = deepcopy(self.sdf.schema)
        term_regex = {}

        for cat, lst in terminos_sub.items():
            col_name = cat.replace(' ', '_')
            term_regex[col_name] = [re.compile(r'\s+' + t.replace(' ', r'[\s\.\,\-\)\;\:\]]+'))
                                    for t in lst]
            schema.add(col_name, 'integer', True)

        self.subbusqueda = (self.sdf.rdd
                            .map(lambda row: spark_buscar_terminos_doc(row, term_regex))
                            .filter(lambda d: d is not None)
                            .toDF(schema=schema)
                            .persist()
                            )

        if actualizar_sdf:
            self.terminos = {**self.terminos, **terminos_sub}
            self.sdf = self.subbusqueda

        return self.subbusqueda

    def frecuencias(self, parquet_file='frecuencias.parquet', **kargs):
        parquet_path = self.datasets_path + '/' + parquet_file
        if os.path.exists(parquet_path):
            self.wdf = self.spark.read.parquet(parquet_path)
        else:
            self.__frecuencias(**kargs)

            if self.wdf is not None:
                self.wdf.write.parquet(parquet_path)

        return self.wdf

    def __frecuencias(self, index_col='index', txt_col='txt', incluir=['PROPN', 'NOUN', 'VERB', 'ADJ'], cambios={}):
        if cambios == {}:
            cambios = {**deepcopy(self.terminos), **
                       deepcopy(self.cambios_config)}

        self.wdf = (self.sdf.mapInPandas(lambda df: spark_extraer_tokens(df,
                                                                         index_col=index_col,
                                                                         incluir=incluir,
                                                                         cambios=cambios),
                                         schema='index string, palabra string, POS string, Dep string')
                    .groupby(['palabra', 'index'])
                    .count()
                    .sort(desc('count'))
                    .persist())

        return self.wdf

    def skipgrams(self, parquet_prefix='skgrams', filtro=[], cruce='index', n=3, k=1, incluir=['NOUN', 'VERB', 'ADJ'], cambios=None):
        print('procesando skipgrams')
        rel_parquet_path = f'{self.datasets_path}/{parquet_prefix}_{cruce}_n{n}_k{k}_{"_".join(incluir)}_rel.parquet'
        att_parquet_path = f'{self.datasets_path}/{parquet_prefix}_{cruce}_n{n}_k{k}_{"_".join(incluir)}_att.parquet'
        if os.path.exists(rel_parquet_path):
            relaciones = pd.read_parquet(rel_parquet_path)
            attributes = pd.read_parquet(att_parquet_path)
        else:
            relaciones, attributes = self.procesar_skipgrams(filtro=filtro,
                                                             cruce=cruce,
                                                             n=n, k=k,
                                                             incluir=incluir,
                                                             cambios=cambios)
            print('Escribiendo Parquets')
            relaciones.to_parquet(rel_parquet_path)
            attributes.to_parquet(att_parquet_path)

        return relaciones, attributes

    def procesar_skipgrams(self, filtro=[], cruce='index', n=3, k=1, incluir=['NOUN', 'VERB', 'ADJ'], cambios=None):
        if cambios == None:
            cambios = {**deepcopy(self.terminos), **
                       deepcopy(self.cambios_config)}

        columnas = [f't{n}' for n in range(n)]

        print('Procesando skipgrams')
        schema = ' string, '.join(columnas)+' string, cruce string, freq int'
        print(self.sdf.mapInPandas(lambda d: spark_skipgrams(d, incluir=incluir,
              cruce=cruce, filtro=filtro, k=k, n=n, cambios=cambios), schema=schema).count())

        res = (self.sdf.mapInPandas(lambda d: spark_skipgrams(d, incluir=incluir, cruce=cruce, filtro=filtro, k=k, n=n, cambios=cambios), schema=schema)
               .groupby(*(columnas+['cruce'])).sum()
               .withColumnRenamed("sum(freq)", "freq")
               .persist())

        print('Creando df de relaciones')
        relaciones_cols = [nom for col in columnas for nom in [
            col, f'POS_{col}']] + ['cruce', 'freq']
        relaciones = (res.rdd.map(lambda r: [t for col in columnas for t in r[col].split('__')]+[r['cruce'], r['freq']])
                      .toDF(relaciones_cols)
                      .toPandas()
                      )

        print('Creando df de atributos')
        attributes = (res.rdd.flatMap(lambda r: [r[col].split('__')+[r['freq']] for col in columnas])
                      .toDF(['token', 'POS', 'freq'])
                      .groupby(['token', 'POS']).sum()
                      .withColumnRenamed("sum(freq)", "freq")
                      .toPandas()
                      .set_index('token'))

        res.unpersist()

        return relaciones, attributes

    def procesar_dependencias(self, cruce='index', excluir=None, cambios=None):
        if cambios == None:
            cambios = {**deepcopy(self.terminos), **
                       deepcopy(self.cambios_config)}

        schema = ''

        