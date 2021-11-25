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
            Extrae la extension de la sentencia.

            Retorna:
                Un nuevo SDF. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.
                
                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.
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
    
    def agregarIDSentencia(self, addColumns, datasetSentencias, actualizar_sdf = False):
        """
            Extrae el numero de sentencia desde el encabezado.

            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.

                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.
        """
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_agregar_ID(row, newColumns, datasetSentencias))
                    .toDF(schema=schema) 
                    .persist()
                    )
        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado
    
    def extraerNumeroSentencia(self, addColumns, actualizar_sdf = False):
        """
            Extrae el numero de sentencia desde el encabezado.

            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.

                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.
        """
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_numero_sentencia(row, newColumns, solo_encabezado))
                    .toDF(schema=schema) 
                    .persist()
                    )
        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado

    def extrarFechaRecibido(self, addColumns, actualizar_sdf = False):
        """
            Extrae la fecha en la que fue recibida la sentencia.

            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.

                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.
        """
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_fecha_recibido(row, newColumns, solo_resultando))
                    .toDF(schema=schema)
                    .persist()
                    )

        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado
    
    def extrarCitaSentenciasFecha(self, addColumns, datasetSentencias, actualizar_sdf = False):
        """
            Extrae la fecha en la que fue recibida la sentencia.

            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.

                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.
        """
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_fecha_cita_sentencia(row, newColumns, datasetSentencias, None))
                    .toDF(schema=schema)
                    .persist()
                    )

        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado
    
    def extraerInstrumentosInternacionales(self, addColumns, actualizar_sdf=False):
        """
            Extrae todos los instrumentos internacionales del considerando y las agrega a la columna.

            Retorna:
                Un nuevo SDF. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.
                
                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.

        """
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_instrumentos_internacionales(row, newColumns , solo_considerando))
                    .toDF(schema=schema)
                    .persist()
                    )

        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado

    def extraerDerechos(self, addColumns, actualizar_sdf=False):
        """
            Extrae una lista de derechos humanos basados en patrones de spacy del considerando de la sentencia.

            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.

                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.
        """
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_derechos(row, newColumns, solo_considerando))
                    .toDF(schema=schema)
                    .persist()
                    )

        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado
    
    def extraerDerechosSinNormalizar(self, addColumns, actualizar_sdf=False):
        """
            Extrae una lista de derechos humanos basados en un patron general de spacy del considerando de la sentencia. 
            Puede incluir matches no normalizados o que no correspondan directamente a un derecho humano.

            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.

                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.
        """
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_derechos_sin_normalizar(row, newColumns, solo_considerando))
                    .toDF(schema=schema)
                    .persist()
                    )

        if actualizar_sdf:
            self.seleccion.sdf = resultado

        return resultado

    def separarSeCondena(self, addColumns, spacy=False, actualizar_sdf=False):
        """
            Extrae todos los patrones del por tanto, que inician con la palabra se condena hasta el signo
            de puntuación punto(.).
            Luego de extraer estos patrones obtiene las entidades asociadas y las agrega a la columna.

            Retorna:
                Un nuevo SDF. 

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
        se_condena_pattern = [{"LOWER": "se"}, {"IS_STOP": True, "OP": "*"}, 
                      {'LEMMA': {"IN":['condenar']} },
                      {"TEXT": {"REGEX": "^(?!\.)"}, "OP": "+"},
                      {"TEXT": '.'}]
        patterns  = [se_condena_pattern]
        (schema, newColumns) = self.__agregarColumnasSchema(addColumns)
        resultado = (self.seleccion.sdf.rdd
                    .map( lambda row : spark_extraer_entidades_se_ordena(row, newColumns , patterns, solo_portanto , useSpacy=spacy))
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
                Un nuevo SDF. 

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
        se_ordena_pattern = [{"LOWER": "se"}, {"IS_STOP": True, "OP": "*"},  {'LEMMA': {"IN":['ordenar',]} },
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
        """
            Extrae todos lo patrones que inician con la palabra plazo y terminan con la palabra horas, días, meses o año y sus variantes. 
            Si no encuentra ninguna se detiene en el primer punto (.).
            
            Retorna:
                Un nuevo SDF, no reemplaza el anterior. 

            Parametros:

                addColumns: Dictionary e.g {llave_1, [valor_1_1, valor_1_n], ... , llave_n, [valor_n_1, valor_n_n]}
                    Es un diccionario de columnas a agregar al schema del sdf. Las llaves corresponde
                    a los nombres de las columnas, el valor corresponde al tipo de dato DataType object de Spark.
                
                actualizar_sdf: Booleano.
                    True para reescribir el sdf.
                    False para no sobreescribir el sdf.

        """

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