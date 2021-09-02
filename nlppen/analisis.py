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


    def frecuencias(self, parquet_file='frecuencias.parquet', **kargs):
        parquet_path = self.datasets_path + '/' + parquet_file
        if os.path.exists(parquet_path):
            self.wdf = self.spark.read.parquet(parquet_path)
        else:
            self.__frecuencias(**kargs)

            if self.wdf is not None:
                self.wdf.write.parquet(parquet_path)

        return self.wdf

    def __frecuencias(self, index_col='index', txt_col='txt', incluir=['PROPN', 'NOUN', 'VERB', 'ADJ'], cambios={}, preprocess=None):
        if cambios == {}:
            cambios = {**deepcopy(self.terminos), **
                       deepcopy(self.cambios_config)}

        self.wdf = (self.sdf.mapInPandas(lambda df: spark_extraer_tokens(df,
                                                                         index_col=index_col,
                                                                         txt_col=txt_col,
                                                                         incluir=incluir,
                                                                         cambios=cambios,
                                                                         preprocess=preprocess),
                                         schema='index string, palabra string, POS string, Dep string')
                    .groupby(['palabra', 'index'])
                    .count()
                    .sort(desc('count'))
                    .persist())

        return self.wdf

    def skipgrams(self, parquet_prefix='skgrams', cruce='index', n=3, k=1, incluir=['PROPN', 'NOUN', 'VERB', 'ADJ'], cambios=None,  **kargs):
        print('procesando skipgrams')
        rel_parquet_path = f'{self.datasets_path}/{parquet_prefix}_{cruce}_n{n}_k{k}_{"_".join(incluir)}_rel.parquet'
        att_parquet_path = f'{self.datasets_path}/{parquet_prefix}_{cruce}_n{n}_k{k}_{"_".join(incluir)}_att.parquet'
        if os.path.exists(rel_parquet_path):
            relaciones = pd.read_parquet(rel_parquet_path)
            attributes = pd.read_parquet(att_parquet_path)
        else:
            relaciones, attributes = self.procesar_skipgrams(n=n, k=k, cruce=cruce, incluir=incluir, cambios=cambios, **kargs)
            print('Escribiendo Parquets')
            relaciones.to_parquet(rel_parquet_path)
            attributes.to_parquet(att_parquet_path)

        return relaciones, attributes

    def procesar_skipgrams(self,n=3, k=1, cruce='index', incluir=['PROPN', 'NOUN', 'VERB', 'ADJ'], cambios=None, **kargs):
        if cambios == None:
            cambios = {**deepcopy(self.terminos), **
                       deepcopy(self.cambios_config)}

        columnas = [f't{n}' for n in range(n)]

        print('Procesando skipgrams')
        schema = ' string, '.join(columnas)+' string, cruce string, freq int'

        res = (self.sdf.mapInPandas(lambda d: spark_skipgrams(d, n=n, k=k, cruce=cruce, incluir=incluir, cambios=cambios,**kargs), schema=schema)
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

        