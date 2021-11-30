import re
import json
import os
from copy import deepcopy

import shutil

from pyspark.sql.functions import desc

from .spark_udfs import *


class Seleccion:
    sdf = None

    def __init__(self, terminos, spark, parquet_path='../../datasets/complete',
                 cambios_path='./cambios.json', datasets_path='./datasets', filtro=None):
        """ Retorna un objeto Seleccion

        Crea un nuevo objeto que contiene un diccioario de terminos para ser aplicados a un conjunto de datos.

        Parametros:

        terminos: Dictionary, e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
            Es un diccionario de terminos a ser buscados en el conjunto de sentencias. 
            Cada llave será agregada como una columna del conjunto de datos, mientras que cada valor
            para el cojunto de valores representa una expresión regular.

        spark:  SparkContext
            Es el contexto de una sesión de Spark. Se encargará de leer los datos desde un parquet.

        parquet_path: String
            Directorio donde se encuentran ubicados los archivos .parquet.

        cambios_path: String
            Directorio donde se almacenan los cambios*

        datasets_path: String
            Directiorio donde se encuentran el conjunto de datos filtrado.
        """
        self.terminos = terminos
        self.spark = spark
        self.parquet_path = parquet_path
        self.datasets_path = datasets_path
        self.filtro_carga = filtro

        if os.path.exists(cambios_path):
            with open(cambios_path) as f:
                self.cambios_config = json.load(f)
        else:
            self.cambios_config = {}

        if not os.path.exists(datasets_path):
            os.makedirs(datasets_path)

        if self.filtro_carga is None:
            self.sdf = self.spark.read.parquet(self.parquet_path)
        else:
            self.sdf = self.spark.read.parquet(
                self.parquet_path).where(self.filtro_carga)

    def cargar_datos(self):
        if os.path.exists(self.parquet_path):
            if self.filtro_carga is None:
                self.sdf = self.spark.read.parquet(self.parquet_path)
            else:
                self.sdf = self.spark.read.parquet(
                    self.parquet_path).where(self.filtro_carga)

    def guardarDatos(self, parquet_file='terminos.parquet', partition_cols=None, borrar=True):
        if self.sdf is not None:
            parquet_path = self.datasets_path + '/' + parquet_file
            if borrar and os.path.exists(parquet_path):
                shutil.rmtree(parquet_path)

            if partition_cols is None:
                self.sdf.write.parquet(parquet_path)
            else:
                (self.sdf.repartition(*partition_cols)
                .write.mode('append').partitionBy(*partition_cols).parquet(parquet_path))

    def cargarPreprocesados(self, parquet_file='terminos.parquet', filtro=None):
        parquet_path = self.datasets_path + '/' + parquet_file
        if os.path.exists(parquet_path):
            if filtro is None:
                self.sdf = self.spark.read.parquet(parquet_path)
            else:
                self.sdf = self.spark.read.parquet(parquet_path).where(filtro)

    def filtrar_sentencias(self, parquet_file='terminos.parquet', precargar=True, preprocess=None, save=True, keepRowEmpty=False, partition_cols=None, overwriteColumns=False):
        """ Sobreescribe el Dataframe aplicando el filtro de las sentencias de acuerdo a los terminos de la clase.

        Parametros:

        parquet_file: String
            Nombre del directorio donde se almacenaras los archivos parquet correspondientes a las sentencias filtradas, si ya existe los lee.

        preprocess : function
            Una funcion para aplicar como preprocesamiento del dataset, antes de aplicar los terminos (e.g solo_portanto())
        """
        parquet_path = self.datasets_path + '/' + parquet_file
        if precargar and os.path.exists(parquet_path):
            self.sdf = self.spark.read.parquet(parquet_path)
        else:
            self.__busqueda_terminos(preprocess, keepRowEmpty, overwriteColumns)
            if save:
                self.guardarDatos(parquet_file, partition_cols=partition_cols)

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

    def tabla_terminos_anno(self):
        terminos_cols = [col.replace(' ', '_') for col in self.terminos.keys()]
        ldf = (self.sdf
               .select('anno', *terminos_cols)
               .groupby('anno').sum()
               .sort('anno')
               .toPandas()
               .set_index('anno')
               .drop(columns=['sum(anno)'])
               )
        ldf.columns = self.terminos.keys()
        return ldf

    def __busqueda_terminos(self, preprocess=None, keepRowEmpty=False, overwriteColumns = False):
        """
        Aplica la busqueda de terminos distribuido con Spark, y sobreescribe el dataframe.

        Retorna:
             Un DataFrame de Spark con la busqueda de terminos aplicada.

        Parametros:

            preprocess: Function
                Funcion para aplicar un proprocesamiento al texto de la sentencia.

        Nota:
            1) Crea un nuevo diccionario de los terminos, compilando la lista de expresiones regulares de cada llave del diccionario de terminos.
            2) Agregar las nuevas columnas al schema de acuerdo a las llaves del diccionario.
            3) Ejecuta la busqueda distribuida con spark.
            4) Guarda el nuevo esquema. 
        """
        schema = deepcopy(self.sdf.schema)

        # Nuevo diccionario para contener los terminos y sus expresiones regulares compiladas.
        term_regex = {}
        for cat, lst in self.terminos.items():
            col_name = cat.replace(' ', '_')
            term_regex[col_name] = [re.compile(r'\s+' + t.replace(' ', r'[\s\.\,\-\)\;\:\]]+'),  re.X | re.M | re.I)
                                    for t in lst]  # Compila las experesiones regulares
            # Agregar las nuevas columnas
            if overwriteColumns == False:
                schema.add(col_name, 'integer', True)

        self.sdf = (self.sdf.rdd
                    # Ejecuta la busqueda de terminos con spark
                    .map(lambda row: spark_buscar_terminos_doc(row, term_regex, preprocess=preprocess, keepRowEmpty=keepRowEmpty))
                    .filter(lambda d: d is not None)
                    .toDF(schema=schema)
                    .persist()
                    )  # Guarda el nuevo esquema
        return self.sdf

    def sub_busqueda(self, terminos_sub, actualizar_sdf=False, preprocess=None, keepRowEmpty=False):
        """
            Aplica búsqueda de los terminos sobre el dataset, a diferencia de filtrar sentencias, este metodo
            recibe por parametro si se desea actualzar el sdf.


            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                terminos_sub: Dictionary, e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de terminos a ser buscados en el conjunto de sentencias. 
                    Cada llave será agregada como una columna del conjunto de datos, mientras que cada valor
                    para el cojunto de valores representa una expresión regular.

                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.

        """
        schema = deepcopy(self.sdf.schema)
        term_regex = {}

        # Crear las nuevas columnas
        for cat, lst in terminos_sub.items():
            col_name = cat.replace(' ', '_')
            term_regex[col_name] = [re.compile(r'\s+' + t.replace(' ', r'[\s\.\,\-\)\;\:\]]+'))
                                    for t in lst]
            schema.add(col_name, 'integer', True)

        # Aplicar la busqueda de terminos
        self.subbusqueda = (self.sdf.rdd
                            .map(lambda row: spark_buscar_terminos_doc(row, term_regex, preprocess=preprocess, keepRowEmpty=keepRowEmpty))
                            .filter(lambda d: d is not None)
                            .toDF(schema=schema)
                            .persist()
                            )
        # Verificar para actualizar el sdf
        if actualizar_sdf:
            self.terminos = {**self.terminos, **terminos_sub}
            self.sdf = self.subbusqueda

        return self.subbusqueda
