import matplotlib.pyplot as plt

from wordcloud import WordCloud
import networkx as nx
import pandas as pd
import numpy as np
import re 

# Visualización

from bokeh.palettes import Colorblind
from bokeh.plotting import figure, from_networkx, show
from bokeh.themes import Theme
from bokeh.io import output_notebook, show, curdoc
from bokeh.models import (Text, GraphRenderer, MultiLine, BoxZoomTool, HoverTool, ResetTool,
                          BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter,
                          ColumnDataSource)

from pyspark.sql.functions import desc

curdoc().theme = Theme('/home/jovyan/Work/erca/bokeh_pen.json')
output_notebook()
pd.set_option('plotting.backend', 'pandas_bokeh')

class Visualizacion:

    @staticmethod
    def wordcloud(wdf, palabra_col='palabra', freq_col='freq', width=1600, height=800, max_words=4000):
      text_wc = wdf.select(palabra_col, freq_col).toPandas().set_index('palabra').to_dict()[freq_col]

      wc = WordCloud(background_color="white", width=width, height=height, max_words=max_words)
      # generate word cloud
      wc.generate_from_frequencies(text_wc)

      # show
      fig = plt.figure(figsize=(20,20))
      plt.imshow(wc, interpolation="bilinear")
      plt.axis("off")
      plt.show()

    @staticmethod
    def crear_grafo_correlaciones(relaciones, features):
        colores = {
            'NOUN': Colorblind[5][0], 
            'VERB': Colorblind[5][1], 
            'ADJ': Colorblind[5][3],
            'PROPN' : Colorblind[5][4],
            'AUX' : Colorblind[5][4],
            'ADV' : Colorblind[5][4],
            'ADP' : Colorblind[5][4],
            'SCONJ' : Colorblind[5][4],
            'DET' : Colorblind[5][4],
            'NUM' : Colorblind[5][4],
            'PRON' : Colorblind[5][4],
            'CCONJ' : Colorblind[5][4],
            }
        token_color = features.apply(lambda row: colores[row['POS']] if row['POS'] in colores else Colorblind[5][4], axis=1)

        font_sizes = (features.index.value_counts() / features.index.value_counts().max() * 15 + 12).astype(str) + 'px'

        G = nx.from_pandas_edgelist(relaciones, 't0', 't1', 'weight')

        # Atributos de los nodos
        features['token'] = features.index.values

        nx.set_node_attributes(G, token_color.to_dict(), 'color')
        nx.set_node_attributes(G, features.apply(lambda row: row['POS'], axis=1).to_dict(), 'POS')
        nx.set_node_attributes(G, font_sizes.to_dict(), 'font_size')
        nx.set_node_attributes(G, features[['token', 'token']].to_dict()['token'], 'token')
        nx.set_node_attributes(G, features.index.value_counts().to_dict(), 'cantidad')

        return G


    @staticmethod
    def mostrar_correlaciones(G, weight_col='weight',scala=10, plot_width=1400, plot_height=700, title=None):

        plot = figure(plot_width=plot_width,
                      plot_height=plot_height,
                      title=title,
                      x_range=(-2, 2), y_range=(-2, 2))

        graph = GraphRenderer()

        node_hover_tool = HoverTool(tooltips=[("index", "@index"), ('POS', '@POS')])
        plot.add_tools(node_hover_tool, BoxZoomTool(), ResetTool())

        graph = from_networkx(G, nx.spring_layout
                              , scale=scala, center=(0,0))
        graph.node_renderer.glyph = Text(text='index', text_color='color', text_font_size='font_size')
        graph.edge_renderer.glyph = MultiLine(line_color="lightblue", line_alpha=1, line_width=weight_col)
        plot.axis.visible = False
        plot.background_fill_alpha = 0
        plot.xgrid.grid_line_color = None
        plot.ygrid.grid_line_color = None
        plot.renderers.append(graph)

        show(plot)

    @staticmethod
    def grafico_freq_palabras(wdf, skip=0, limit=30, **kwargs):
        df = (wdf
            .groupby('palabra').sum()
            .sort(desc('sum(count)'))
            .limit(skip+limit)
            .toPandas()[skip:]
            .set_index('palabra'))
        
        args = {
            'stacked': True,
            'figsize': (800,700),
            'title': 'Frecuencia de palabras para las sentencias relacionadas', 
            'xlabel': 'Frecuencia',
            'ylabel': 'Palabra',
            'legend': False,
            #   colormap=erca_colores,
            # 'show_figure': False
        }
        args.update(kwargs)
        
        plot = df.plot.barh(**args)
        return df, plot
        
    @staticmethod
    def grafico_freq_palabras_cruce(wdf, cruce, index_col='index', skip=0, limit=30, **kwargs):
        df_cruce = (wdf
                    .filter(f'{index_col} == "{cruce}"')
                    .groupby('palabra').sum()
                    .sort(desc('sum(count)'))
                    .limit(skip+limit)
                    .withColumnRenamed("sum(count)", cruce)
                    .toPandas()[skip:]
                    .set_index('palabra')
                )

        filter_string = '"' + '", "'.join(df_cruce.index) + '"'
        df_total = (wdf
                    .filter(f'palabra IN ({filter_string})')
                    .groupby('palabra')
                    .sum()
                    .withColumnRenamed("sum(count)", 'Total(Dif.)')
                    .toPandas()
                    .set_index('palabra')
                )
        df = df_cruce.join(df_total)
        df['Total(Dif.)'] = df['Total(Dif.)'] - df[cruce]
        
        args = {
            'stacked': True,
            'figsize': (800,700),
            'title': 'Frecuencia de palabras para las sentencias relacionadas', 
            'xlabel': 'Frecuencia',
            'ylabel': 'Palabra',
            'legend': True,
            'colormap': ['#2E86C1', '#D6EAF8'],
            'show_figure': False
        }
        args.update(kwargs)
        
        plot = df.plot.barh(**args)
        return df, plot

    @staticmethod
    def grafico_skipgrams(rel, skip=0, limit=30, **kwargs):
        col_names = [ col for col in rel.columns if re.match('t\d+',col) ]
        df = (rel[col_names]
            .apply(lambda row: ', '.join(row), axis=1)
            .to_frame()
            .join(rel[['freq']])
            .groupby(0).sum()
            .sort_values('freq', ascending=False)[skip:limit]
            )

        args = {
            'stacked': True,
            'figsize': (800,700),
            'title': 'Frecuencia de ngramas para las sentencias relacionadas', 
            'xlabel': 'Frecuencia',
            'ylabel': 'Palabra',
            'legend': True,
            'colormap': ['#2E86C1', '#D6EAF8'],
            'show_figure': False
        }
        args.update(kwargs)

        plot = df.plot.barh(**args)
        return df, plot

    @staticmethod
    def grafico_skipgrams_cruce(rel, cruce, skip=0, limit=30, **kwargs):
        col_names = [ col for col in rel.columns if re.match('t\d+',col) ]

        df = (rel[col_names]
                    .apply(lambda row: ', '.join(row), axis=1)
                    .to_frame()
                    .join(rel[['freq', 'cruce']])
            )


        df_cruce = (df
                    .query(f'cruce == "{cruce}"')
                    .groupby(0).sum()
                    .sort_values('freq', ascending=False)[skip:limit]
                    .join(df.groupby(0).sum()['freq'], rsuffix='_t')
                    .rename(columns={'freq':cruce,'freq_t':'Total'})
                )
        df_cruce['Total(Dif)'] = df_cruce['Total'] - df_cruce[cruce]

        args = {
            'stacked': True,
            'figsize': (800,700),
            'title': 'Frecuencia de ngramas para las sentencias relacionadas', 
            'xlabel': 'Frecuencia',
            'ylabel': 'Palabra',
            'legend': True,
            'colormap': ['#2E86C1', '#D6EAF8'],
            'show_figure': False
        }
        args.update(kwargs)

        plot = df_cruce[[cruce, 'Total(Dif)']] .plot.barh(**args)
        return df_cruce, plot

    @staticmethod
    def terminos_cruce_circulos(df, term, cruce, bins=[0,1,2,3,4,5,10,20,50, 100,100000], index_col='index', title=None, circle_factor=0.7, circle_min=3):
        ldf = df
        ldf['d_bins'] = pd.cut(df[term], bins)
        ldf = ldf[['d_bins',cruce, index_col]].groupby(['d_bins',cruce]).count().reset_index()
        ldf.loc[ldf[index_col] == 0,index_col] = pd.NA
        ldf.index = ldf.index.astype(str)
        ldf['d_bins'] = ldf.astype(str)
        
        ldf['size'] = ldf[index_col] * circle_factor + circle_min
        
        source = ColumnDataSource(ldf)
        mapper = LinearColorMapper('Spectral11', low=0, high=ldf[index_col].max())
        
        p = figure(title=title, x_range=ldf.d_bins.unique(), y_range=ldf[cruce].unique(), width=800, height=800,)
        p.circle(x='d_bins', y=cruce, size='size', source=source, alpha=0.7, line_color=None, 
                fill_color={'field': 'size', 'transform': mapper},)
        
        p.xaxis.axis_label = 'Frecuencia de términos'
        p.yaxis.axis_label = cruce.title()
        p.xaxis.major_label_orientation = np.pi/2
        
        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="9px",
                        ticker=BasicTicker(desired_num_ticks=5),
                        formatter=PrintfTickFormatter(format="%d docs"),
                        label_standoff=10, border_line_color=None, height=150, margin=0 , width=16, padding=30)
        p.add_layout(color_bar, 'right')
        
        return p

    @staticmethod
    def terminos_cruce_circulos_v2(df, term, cruce, bins=[0,1,2,3,4,5,10,20,50, 100,100000], index_col='index', title=None, circle_factor=0.7, circle_min=3):
        ldf = df
        ldf['d_bins'] = pd.cut(df[term], bins)
        ldf = ldf[['d_bins',cruce, index_col]].groupby(['d_bins',cruce]).count().reset_index()
        ldf.loc[ldf[index_col] == 0,index_col] = pd.NA
        ldf.index = ldf.index.astype(str)
        ldf['d_bins'] = ldf.astype(str)
        
        ldf['size'] = ldf[index_col] * circle_factor + circle_min
        
        source = ColumnDataSource(ldf)
        mapper = LinearColorMapper('Spectral11', low=0, high=ldf[index_col].max())
        
        p = figure(title=title, x_range=ldf.d_bins.unique(), y_range=ldf[cruce].unique(), width=800, height=800,)
        p.circle(x='d_bins', y=cruce, size='size', source=source, alpha=0.7, line_color=None, 
                fill_color={'field': 'size', 'transform': mapper},)
        
        p.xaxis.axis_label = 'Frecuencia de términos'
        p.yaxis.axis_label = cruce.title()
        p.xaxis.major_label_orientation = np.pi/2
        
        color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="9px",
                        ticker=BasicTicker(desired_num_ticks=5),
                        formatter=PrintfTickFormatter(format="%d docs"),
                        label_standoff=10, border_line_color=None, height=150, margin=0 , width=16, padding=30)
        p.add_layout(color_bar, 'right')
        
        return p