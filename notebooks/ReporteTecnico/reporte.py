
import sys
sys.path.insert(0, "/home/jovyan/Work/ej/paquetes/nlppen/")
from datetime import date

from nlppen.extraccion.utils.Txt2Numbers import Txt2Numbers
from nlppen.analisis import Analisis
from nlppen.seleccion import Seleccion
from nlppen.spark_udfs import solo_portanto, spark_get_spacy
from nlppen.sentencias_estructurales import SentenciasEstructurales

from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.sql import functions as F

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
pd.options.display.latex.repr=True

# Configuración permanente de los gráficos


def init_spark():
    spark = (SparkSession
         .builder
         .appName("Transforming sentences")
         .config("spark.num.executors", "2")
         .config("spark.executor.memory", "10g")
         .config("spark.executor.cores", "4")
         .config("spark.driver.memory", "10g")
         .config("spark.memory.offHeap.enabled", True)
         .config("spark.memory.offHeap.size", "64g")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .getOrCreate())

    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    return spark, sc


def cargar_datos(spark):
    terminos = {
        'seguimiento': [r'\bseguimiento\b'],
        'se ordena': [r'\bse ordena\b'],
        'plan': [r'\bplan\b'],
        'plazo': [r'\bplazo\b']
    }
    seleccion = Seleccion(
        terminos, spark, parquet_path='../../../../src/datasets/complete/', 
        datasets_path='../../datasets/estructurales/')
    seleccion.cargarPreprocesados()

    estructurales = SentenciasEstructurales(seleccion)
    
    
    return estructurales


def preprocesar_plazos(estructurales):
    estructurales.seleccion.sdf = (estructurales.seleccion.sdf
        .where('anno < 2019')
        .withColumn('dias_tramite', F.datediff('fechahora_ext', 'FechaSolicitud'))
        .withColumn('plazosDefinidos', F.explode_outer('plazosDefinidos'))
        .withColumn('plazosDefinidos', F.datediff('plazosDefinidos', F.lit(date(1970, 1, 1))))
        .withColumn('num_resolucion', F.regexp_replace('num_resolucion', "[^\d]", ""))
        ).persist()



def grafico_se_ordena_anno(estructurales):
    s = estructurales.seleccion.sdf
    df1 = (s
           .where('termino_ext == "Con lugar" OR  termino_ext == "Con lugar parcial" ')
       .groupby('anno')
       .count()
       .sort('anno')
       .withColumnRenamed('count', 'Total Con lugar')
      ).toPandas().set_index('anno')

    df2 = (s
           .where('se_ordena != 0')
           .where('termino_ext == "Con lugar" OR  termino_ext == "Con lugar parcial" ')
         .groupby('anno')
         .count()
         .sort('anno')
         .withColumnRenamed('count', 'Se Ordena')
        ).toPandas().set_index('anno')


    title = "Sentencias 'Con lugar' con los términos 'Se ordena'"

    ax = df1.join(df2).plot.bar(figsize=(16,8), title=title, xlabel='Año', ylabel='Cantidad')
    sns.despine(ax=ax)
    
    return ax