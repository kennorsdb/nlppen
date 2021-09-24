
import re
import pandas as pd

class Txt2Date:
    def __init__(self):
        self.horaRegularExpression = re.compile(r"[Hh][OoÓó][Rr][Aa]([Ss])?")
        self.diasRegularExpression = re.compile(r"[dD][iíIÍ][ÁAaa]([Ss])?")
        self.mesesRegularExpression = re.compile(r"[Mm][EeÉé][Ss]([EeÉé][Ss])?")
        self.annosRegularExpression = re.compile(r"[ÁáAa][Ññ][OoÓó]([Ss])?")
    
    def txt2Date(self, text, number):
        """
        Convierte un texto y un número a un TimeDelta a partir del uso de expresiones regulares para
        identficar si es hora, día, mes, año. Para los meses y años se hace la conversión a días.
        1 mes = 30 dias
        1 años = 365 días. 
        

        Retorna:
             TimeDelta de pandas.

        Parametros:
            text: String
                Texto de entrada que indica la unidad de tiempo.

            number: Float
                Número que indica la magnitud de la unidad de tiempo.
        """

        if self.horaRegularExpression.search(text) != None:
            return pd.Timedelta(hours=number)
        if self.diasRegularExpression.search(text) != None:
            return pd.Timedelta(days=number)
        if self.mesesRegularExpression.search(text) != None:
            days = number*30
            return pd.Timedelta(days=days)
        if self.annosRegularExpression.search(text) != None:
            days = number*365
            return pd.Timedelta(days=days)
        return None