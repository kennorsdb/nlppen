import os
import re

from .export import Export

class Descripcion():
  def __init__(self, df, extraccion_path='./extraccion/'):
    self.df = df
    self.extraccion_path = extraccion_path
    self.export = Export(extraccion_path)


  def guardar_lista(self, columnas, sheet='Lista'):
    self.export(sheet, self.df[columnas])

  def guardar_df(self, df, sheet='Lista'):
    self.export(sheet, df)

  def cruce_variables(self, var1, var2, sheet=None):
    ldf = (self.df[['archivo', var1, var2]]
              .groupby([var1,var2]).count()
              .sort_values([var1], ascending=True)
              .fillna(0)
              .unstack(fill_value=0)
              .T.fillna(0))

    if sheet is not None:
      self.export(sheet, ldf.copy())

    return ldf

  def resumen_constitucion(self, articulos_col='Constitución_ents', sheet=None):
    ldf = (self.df[articulos_col]
      .dropna()
      .apply(lambda ls: set([int(s) for art in ls 
                              for s in re.findall(r"\d+", art) if art is not None ]))
      .explode()
      .value_counts()
      .to_frame()
      .sort_index()
      .reset_index())

    ldf.columns = ['Art.', 'Sentencias que citan el artículo']

    if sheet is not None:
      self.export(sheet, ldf.copy())

    return ldf

  def cruce_constitucion(self, cruce, articulos_col='Constitución_ents', sheet=None):
    ldf = (self.df[self.df[articulos_col].notna()][[articulos_col, cruce]]
            .groupby(cruce)
            .apply(lambda df: set([st for ls in df[articulos_col].dropna() 
                                      for art in ls 
                                      for st in re.findall(r"\d+", art)]))
            .explode()
            .rename('Artículos')
            .reset_index()
            .reset_index()
            .groupby([cruce, 'Artículos'])
            .count()
            .unstack()
            .fillna(0)
          )

    ldf.columns = ldf.columns.get_level_values(1)
    
    if sheet is not None:
      self.export(sheet, ldf.copy())

    return ldf
      

  def cruce_entidades(self, cruce, entidades_col, ent_name='Entidad', sheet=None):
    ldf = (self.df[self.df[entidades_col].notna()][[entidades_col, cruce]]
            .groupby(cruce)
            .apply(lambda df: set([ent for ls in df[entidades_col].dropna() 
                                      for ent in ls]))
            .explode()
            .rename(ent_name)
            .reset_index()
            .reset_index()
            .groupby([cruce, ent_name])
            .count()
            .unstack()
            .fillna(0).T
          )
    
    if sheet is not None:
      self.export(sheet, ldf.copy())

    return ldf

