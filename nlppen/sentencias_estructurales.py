from .spark_udfs import *

from copy import deepcopy

class SentenciasEstructurales():
    def __init__(self, seleccion):
        self.seleccion = seleccion
    
    def __agregarColumnasSchema(self, columnas):
        """
        Agrega nuevas columnas al schema del sdf.

        Retorna:
             Un nuevo esquema con las nuevas columnas, y las columnas agregadas

        Parametros:

        columnas: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
            Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
            a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.
        """

        schema = deepcopy(self.seleccion.sdf.schema)
        newColumns = []
        for colum, type in columnas.items():
            #Convertir y agregar todas las columnas
            col_name = colum.replace(' ', '_')
            newColumns.append(col_name)
            schema.add(col_name, type, True)
        return (schema,newColumns)

    def extraerExtension(self, addColumns, actualizar_sdf = False):
        """
            Extrae las extesiones de las sentencias de acuerdo a la sentencia total y la parte del por lo tanto

            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.

        """
        
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_extension(row, newColumns, solo_portanto))
                    .toDF(schema=schema)
                    .persist()
                    )

        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado

    def separarSeOrdena(self, addColumns, spacy=False, actualizar_sdf=False):
        """
            Extrae todos los patrones del por tanto, que inician con la palabra se ordena hasta el signo
            de puntuación punto(.).
            Luego de extraer estos patrones obtiene las entidades asociadas y las agrega a la columna.

            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.

                spacy: Booleano.
                    True para usar spacy para obtener las entidades.
                    False para usar stanza para obtener las entidades.
                
                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.

        """
        se_ordena_pattern = [{"LOWER": "se"}, {'LEMMA': 'ordenar'},
                      {"TEXT": {"REGEX": "^(?!\.)"}, "OP": "+"},
                      {"TEXT": '.'}]
        patterns  = [se_ordena_pattern]
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_entidades_se_ordena(row, newColumns , patterns, solo_portanto , useSpacy=spacy))
                    .toDF(schema=schema)
                    .persist()
                    )

        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado

    def plazosDefinidos(self, addColumns, actualizar_sdf=False):
        plazos_pattern =  [{"LOWER": "plazo"},
          {"TEXT": {"REGEX": "^(?!\.|[Hh][OoÓó][Rr][Aa]([Ss])?|[dD][iíIÍ][ÁAaa]([Ss])?|[Mm][EeÉé][Ss]([EeÉé][Ss])?|[ÁáAa][Ññ][OoÓó]([Ss])?)"}, "OP": "+"},
          {"TEXT": {"REGEX": "\.|[Hh][OoÓó][Rr][Aa]([Ss])?|[dD][iíIÍ][ÁAaa]([Ss])?|[Mm][EeÉé][Ss]([EeÉé][Ss])?|[ÁáAa][Ññ][OoÓó]([Ss])?"}},
          ]
        patterns  = [plazos_pattern]
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_plazos(row, newColumns , patterns, solo_portanto))
                    .toDF(schema=schema)
                    .persist()
                    )

        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado