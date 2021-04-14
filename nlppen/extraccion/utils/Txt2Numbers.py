import re
from datetime import date, time

class Txt2Numbers:

    def __init__(self):
        self.limpiar = re.compile (r'[ \s]+', re.M)
        self.numberList = {
            r"cero":    0,
            r"una":     1,
            r"un":      1,
            r"uno":     1,
            r"primero":     1,
            r"dos":     2,
            r"tres":    3,
            r"cuatro":  4,
            r"cinco":   5,
            r"seis":    6,
            r"siete":   7,
            r"ocho":    8,
            r"nueve":   9,
            r"diez":    10,
            r"once":    11,
            r"doce":    12,
            r"trece":   13,
            r"catorce": 14,
            r"quince":  15,
            r"dieciseis":   16,
            r"dieciséis":   16,
            r"diecisiete":  17,
            r"dieciocho":   18,
            r"diecinueve":  19,
            r"veinte":      20,
            r"veintiuna":   21,
            r"veintiuno":   21,
            r"veintiún":   21,
            r"veintiun":   21,
            r"veintidos":   22,
            r"veintidós":   22,
            r"veintitres":  23,
            r"veintitrés":  23,
            r"veinticuatro":24,
            r"veinticinco": 25,
            r"veintiseis":  26,
            r"veintiséis":  26,
            r"veintisiete": 27,
            r"veintiocho":  28,
            r"veintinueve": 29,
            r"treinta":     30,
            r"treinta y una":   31,
            r"treinta un":   31,
            r"treinta y un":   31,
            r"treinta y uno":   31,
            r"treinta y dos":   32,
            r"treinta y tres":  33,
            r"treinta y cuatro":34,
            r"treinta y cinco": 35,
            r"treinta y seis":  36,
            r"treinta y siete": 37,
            r"treinta y ocho":  38,
            r"treinta y nueve": 39,
            r"cuarenta":        40,
            r"cuarenta y una":  41,
            r"cuarenta y un":  41,
            r"cuarenta un":  41,
            r"cuarenta y uno":  41,
            r"cuarenta y dos":  42,
            r"cuarenta y tres": 43,
            r"cuarenta y cuatro":44,
            r"cuarenta y cinco": 45,
            r"cuarenta y seis":  46,
            r"cuarenta y siete": 47,
            r"cuarenta y ocho":  48,
            r"cuarenta y nueve": 49,
            r"cincuenta":        50,
            r"cincuenta y una":  51,
            r"cincuenta un":  51,
            r"cincuenta y un":  51,
            r"cincuenta y uno":  51,
            r"cincuenta y dos":  52,
            r"cincuenta y tres": 53,
            r"cincuenta y cuatro":54,
            r"cincuenta y cinco":55,
            r"cincuenta y seis":56,
            r"cincuenta y siete":57,
            r"cincuenta y ocho":58,
            r"cincuenta y nueve":59,
            r"ochenta y siete":87,
            r"ochenta y ocho":88,
            r"ochenta y nueve":89,
            r"noventa":90,
            r"noventa y uno":91,
            r"noventa y un":91,
            r"noventa y dos":92,
            r"noventa y tres":93,
            r"noventa y cuatro": 94,
            r"noventa y cinco":95,
            r"noventa y seis":96,
            r"noventa y siete":97,
            r"noventa y ocho":98,
            r"noventa y nueve":99,
            r"mil novecientos":19,
            r"dos mil":20,
            r"enero":1,
            r"febrero":2,
            r"marzo":3,
            r"abril":4,
            r"mayo":5,
            r"junio":6,
            r"julio":7,
            r"agosto":8,
            r"setiembre":9,
            r"octubre":10,
            r"noviembre":11,
            r"diciembre":12
        }
        

    def number(self, text):
        return self.numberList[self.limpiar.sub(" ", text).lower()]