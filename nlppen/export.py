import os
import pandas as pd
from bokeh.io import show, export_png, export_svgs
import string
from openpyxl.utils import get_column_letter

class Export:

    def __init__(self, directory='exports'):
        self.directory = directory
        if not os.path.exists('./' + directory):
            os.makedirs(directory)
        self.exportar_excel = {}

    def __call__(self, nombre, df=None, plot=None):
        self.agregar(nombre, df, plot)

    def agregar(self, nombre, df=None, plot=None):
        if df is not None:
            self.exportar_excel[nombre] = df.reset_index()

        if plot is not None:
            plot.output_backend = "svg"
            export_svgs(plot, filename=self.directory + '/' +
                        nombre.replace(" ", "_") + ".svg")
            export_png(plot, filename=self.directory +
                       nombre.replace(" ", "_") + ".png")

    def guardar(self, nombreArchivo):
        letras = []
        with pd.ExcelWriter(nombreArchivo + '.xlsx') as writer:
            for key in self.exportar_excel:
                self.exportar_excel[key].to_excel(writer, sheet_name=key, index=False)
                worksheet = writer.sheets[key]
                
                for idx, col in enumerate(self.exportar_excel[key]):  # loop through all columns
                    series = self.exportar_excel[key][col]
                    max_len = max((
                        series.astype(str).map(len).max(),  # len of largest item
                        len(str(col))  # len of column name/header
                        )) + 1  # adding a little extra space
                    worksheet.column_dimensions[get_column_letter(idx+1)].width = min((max_len, 50))  # set column width

                

    
