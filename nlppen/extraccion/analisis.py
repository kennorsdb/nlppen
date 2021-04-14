import sys
sys.path.append('../')

# Presentación de información en las celdas
from IPython.core.display import HTML

# Almacenamiento en Base de Datos
from .mongodb import MongoTxA, AllDataQuery, GetYearsQuery, GetExtractedCount, CustomFindQuery, CustomAggregationQuery
from bson.objectid import ObjectId

# Herramientas de estructuras de datos y procesamiento
import numpy as np
import pickle
import pandas as pd
from joblib import Memory
import matplotlib as plt

# Herramientas de cálculo
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Redes Neuronales
import keras as keras
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import load_model
import tensorflow as tf


# Visualización Bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import HoverTool, ColumnDataSource, ColorBar, LinearColorMapper, BasicTicker, PrintfTickFormatter, NumeralTickFormatter, Slider, CustomJS
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, GraphRenderer, StaticLayoutProvider, Oval
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.io import output_notebook
from bokeh.models.markers import Circle
from bokeh.palettes import RdYlGn, Category20c, viridis, Spectral4
from bokeh.models.widgets import Dropdown, DataTable, DateFormatter, TableColumn
from bokeh.layouts import column, row
from bokeh.transform import cumsum, transform
output_notebook()

# Visualización Plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Manipulación de Grafos
import networkx as nx  

# WordClouds
from wordcloud import WordCloud

#Visualización Plotly
import matplotlib.pyplot as plt

# Clusters
from MulticoreTSNE import MulticoreTSNE as TSNE

# Rutinas de Data Scaling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

# Herramientas de visualización
from utils import plot_confusion_matrix, JupLogging
from tqdm import tqdm_notebook as tqdm

# Inicialización de algunos módulos
cachedir = '../.pycache'
memory = Memory(cachedir, verbose=0)

log, _ = JupLogging.getLogger() 

coloresPen = np.array([ '#0075A7', '#E7723A', '#757E84', '#BDDD06A', '#BA9679', '#EFB740',
    '#325935', '#1E4592', '#F3953F', '#95C13D', '#1EABE2', '#EFB55C', '#DD4847', '#F9BABA', '#F59191', '#0D8C44', '#734792', 
    '#B27BB3', '#0B789D', '#C4DB9C', '#657B85', '#88A5B2', '#DBF1FD', '#B0D6E8', '#D0B3D6', '#9469AD', '#AF3931', '#0B5167', 
    '#66C1BF', '#86CAE4', '#FDD800', '#FFE374', '#C59133', '#FFF277', '#5F9E8E', '#CD7629', '#A3A06F', '#F5E28A'
    ])


@memory.cache
def cargar_annos():
    mongo = MongoTxA(GetYearsQuery(), db="SalaC2", collection="All"  )
    annos = pd.DataFrame([ doc for doc in tqdm(mongo.getCursor(0), total=mongo.getTotalDocs())])
    annos = annos.set_index('anno').sort_index()
    return annos

def plot_annos(annos):
    source = ColumnDataSource(annos)
    tooltips = [  ("Año", "@anno"), ("Cantidad", "@cantidad") ]
    p = figure(x_range=annos.index.values, plot_height=350, title="Frecuencia de documentos por año", toolbar_location=None, tools="hover", tooltips=tooltips,)
    
    p.vbar(x='anno', top='cantidad', width=0.9, source=source)
    p.xaxis.major_label_orientation = 3.1416/2
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = 'Años'
    p.yaxis.axis_label = 'Cantidad de Documentos'
    show(p)

@memory.cache
def cargar_frecuenciaExtraccion():
    mongo = MongoTxA(GetExtractedCount(), db="SalaC2", collection="All"  )
    freq = pd.DataFrame([ doc for doc in tqdm(mongo.getCursor(0), total=mongo.getTotalDocs())])
    return freq.T.sort_values(by=[0], ascending=True)

def plot_frecuenciaExtraccion(freq_ext):
    source = ColumnDataSource(data={'index': freq_ext.index.values, 'Cantidad': freq_ext[0].values, 'Porcentaje': freq_ext[0].values/freq_ext[0].loc['Total']*100  ,'color':coloresPen[0:len(freq_ext.index.values)]  })

    tooltips = [  ("Variable", "@index"), ("Cantidad", "@Cantidad"), ("Porcentaje", "@Porcentaje%") ]
    p = figure( y_range=freq_ext.index.values, plot_height=400, title="Cantidad de documentos de cada variable extraída", toolbar_location='above', tools="hover", tooltips=tooltips)

    p.hbar( y='index', height=0.9, right='Cantidad', source=source , color='color')

    p.y_range.range_padding = 0.1
    p.ygrid.grid_line_color = None
    p.axis.minor_tick_line_color = None
    p.outline_line_color = None
    p.xaxis.axis_label = 'Cantidad'
    p.xaxis.major_label_orientation = 3.1416/2
    p.xaxis.formatter=NumeralTickFormatter(format="00")

    show(p)
    
@memory.cache
def load_classif_Sala():
    with open('../datasets/Clasificacion-SegundoTexto.pickle', 'rb') as handle:
        return pickle.load(handle)

@memory.cache
def vectorizarCorpus(corpus, max_df=0.95, min_df=2, max_features=4000):
    log.info("Creando vector del corpus")
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df,max_features=max_features)
    corpus_vec = vectorizer.fit_transform(corpus)
    
    log.info("Normalizando vectores usando Tf-Idf")
    idf = TfidfTransformer()
    return idf.fit_transform(corpus_vec), vectorizer, idf

# @memory.cache
def tsne(data):
    data = data.toarray()
    tsne_model = TSNE(n_jobs=-1, n_components=3, n_iter=300, perplexity=5 ,verbose=1, init='random')
    tsne =  tsne_model.fit_transform(data)
    return tsne

@memory.cache
def load_classif_Sala():
    with open('../datasets/Clasificacion-SegundoTexto.pickle', 'rb') as handle:
        return pickle.load(handle)
    
# @memory.cache
def lda(corpus, n_components=10):
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5, learning_method='online', learning_offset=50., random_state=0, n_jobs=-1)
    lda.fit(corpus)
    return lda
    
def plot3D(data, colors="#0B5167", hoverInfo=None):
    trace = go.Scatter3d( x= data[:, 0], y= data[:, 1], z= data[:, 2], 
        text = hoverInfo,
        hoverinfo = 'text',                 
        mode='markers', marker=dict( size=1, color=colors, opacity=0.8 ) )
    data = [trace]

    layout = go.Layout( title=go.layout.Title( text='Plot Title', xref='paper', x=0 ), 
                        xaxis=go.layout.XAxis( title=go.layout.xaxis.Title( text='x Axis', font=dict( family='Courier New, monospace', size=18, color='#7f7f7f' ) ) ),
                        yaxis=go.layout.YAxis( title=go.layout.yaxis.Title( text='y Axis', font=dict( family='Courier New, monospace', size=18, color='#7f7f7f' ) ) ) )
    layout = go.Layout( margin=dict( l=0, r=0, b=0, t=0 ) )
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename='simple-3d-scatter.html')
    
def wordClouds(model, feature_names, n_top_words):
    res = []
    fig, ax = plt.subplots(figsize=(30, 20))
    for topic_idx, topic in enumerate(model.components_):
        plt.subplot(4, 3, topic_idx+1)
        d = {}
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            d[feature_names[i]] = topic[i];
        
        res.append(d)
        wc = WordCloud(background_color="white", max_words=1000)
        wc.generate_from_frequencies(d)
        
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
    plt.show()
        
    return res


def plot_temas(data):
    df1 = data[['tema','subtema','_id']].groupby(['tema','subtema']).count()

    original = ColumnDataSource(data=df1)
    source = ColumnDataSource(dict(subtemas=[], cantidad=[], color=[], angle=[]))
    callback = CustomJS(args=dict(source=source, original=original, colors=Category20c[20]+Category20c[20]+Category20c[20]), code="""
        subtemas = [];
        cantidad = [];
        col = [];
        angle = [];
        total = 0;
        console.log(source);
        index = original.data.tema_subtema.reduce(function(a, e, i) {
                    if (e[0] === cb_obj.value) {
                        a.push(i);
                        subtemas.push(e[1]);
                        valor = original.data._id[i]
                        cantidad.push(valor);
                        total += valor
                    }
                    return a;
                }, []);

        for (let i in cantidad) {
            angle.push(cantidad[i]/total * 2 * 3.1416);
            col.push(colors[i]);
        }

        source.data.subtemas = subtemas;
        source.data.cantidad = cantidad;
        source.data.color = col;
        source.data.angle = angle;

        source.change.emit();
    """)

    menu = list(zip(list(df1.index.levels[0]),list(df1.index.levels[0])))
    dropdown = Dropdown(label="Selecciones un Tema", button_type="success", menu=menu)
    dropdown.js_on_change('value', callback)

    p = figure(plot_height=800, plot_width=800, title="Frecuencia de las resoluciones en los subtemas", toolbar_location=None,
               tools="hover", tooltips="@subtemas: @cantidad", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend='subtemas', source=source)

    p.axis.axis_label=None
    p.axis.visible=False
    p.grid.grid_line_color = None
    p.legend.label_text_font_size = '6pt'

    columns = [
            TableColumn(field="subtemas", title="Subtema"),
            TableColumn(field="cantidad", title="Cantidad"),
        ]
    data_table = DataTable(source=source, columns=columns, width=800, height=280)


    show(column(row(dropdown,p), data_table))
    
    
@memory.cache
def dataScaling(corpus, tags):
    log.info("OverSampling usando SMOTE ")
    smote =  SMOTE('minority', n_jobs=-1)
    corpus, y_binary = smote.fit_sample(corpus, tags)
    log.info("---> OverSampling final en " + str(corpus.shape[1]) + " muestras")
    enn = EditedNearestNeighbours('not minority', n_jobs=-1)
    log.info("UnderSampling usando ENN")
    corpus, y_binary = enn.fit_sample(corpus, y_binary)
    log.info("---> UnderSampling final en " + str(corpus.shape[1]) + " muestras")
    return corpus, y_binary

def encodeLabels(labels, encoder = None):
    if encoder is None:
        encoder = LabelEncoder()
    y_labels = encoder.fit_transform(list(labels))
    return encoder, y_labels


class NNModel:
    def __init__(self, corpus, tags):
        print("Separando los datos de entrenamiento y de prueba ")
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split( corpus, tags, test_size=0.20)
        print("Normalizando los datos")
        self.x_train = self.x_train.astype('float16')
        self.x_test = self.x_test.astype('float16')
        self.inputLayerDim = self.x_train.shape[1]
        self.ouputLayerDim = tags.shape[1]
        print("---> InputLayer: " + str(self.inputLayerDim) + "  --  OutputLayer: " + str(self.inputLayerDim))
        
    def buildModel(self):
        log.info("Construyecndo el modelo")
        self.model = Sequential()
        self.model.add(Dense(units=800, activation='relu', input_dim=self.inputLayerDim))
        self.model.add(Dense(units=200, activation='relu' ))
        self.model.add(Dense(units=300, activation='relu'))
        self.model.add(Dense(units=400, activation='relu'))
        self.model.add(Dense(units=100, activation='relu'))
        self.model.add(Dense(units=self.ouputLayerDim, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy',keras.metrics.categorical_accuracy, keras.metrics.top_k_categorical_accuracy, tf.keras.metrics.Recall(), keras.metrics.AUC(name='auc')])
        
    def fit(self, verbose=0):
        log.info("Iniciando entrenamiento")
        self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=verbose)
        log.info("Enrenamiento finalizado")
        
    def plot_roc(self):
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(self.y_test, self.y_pred_model.ravel())
        auc_keras = auc(fpr_keras, tpr_keras)
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        
    def test(self, plotConfusionMatrix=False, encoder=None):
        log.info("Pruebas de rendimiento: ")
        loss_and_metrics = self.model.evaluate(self.x_test, self.y_test, batch_size=32, verbose=0)
        log.info("Pruebas de rendimiento: Loss: " + str(loss_and_metrics[0]) + " Categorical Accuracy:" + str(loss_and_metrics[1]))
        self.y_pred_model = self.model.predict(self.x_test, batch_size=64, verbose=0)
        self.y_pred_bool = np.argmax(self.y_pred_model, axis=1)
        report = classification_report( np.argmax(self.y_test, axis=1), self.y_pred_bool)
        # Se incorporan las muestras del test al modelo
        self.model.fit(self.x_test, self.y_test, epochs=5, batch_size=32, verbose=0)
        plt.legend(loc='lower right')
#         self.plot_roc()
        if plotConfusionMatrix and encoder is not None:
            plot_confusion_matrix.plot_confusion_matrix( np.argmax(self.y_test, axis=1), self.y_pred_bool, classes=encoder.classes_ , normalize=True  )
        
        return loss_and_metrics, report, self.y_pred_model
        
    def load_model(self, file):
        self.model = load_model(file)
    
    def get_model(self):
        return self.model
    
    
    
    
    
    
class PredictCorpus:
    def __init__(self, model, encoder, vectorizer, tfidf):
        with open('../datasets/corpus_noTAGS_ALL.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
            
        self.corpus = tfidf.transform(vectorizer.transform(self.data.cleanText))
        self.model = model
        self.encoder = encoder
        
    def predict(self):
        self.predicted = self.model.predict(self.corpus, batch_size=128, verbose=0)
        self.estimated = self.encoder.inverse_transform(self.predicted.argmax(axis=1))
        
    def resultById(self, id):
        cat = self.predicted[id].argsort()[-4:]
        return pd.DataFrame( {'categorías': self.encoder.inverse_transform(cat), 'Porcentaje %' : self.predicted[id][cat]*100 })
        
    def getData(self):
        return self.data
        
    def save2Db(self, fieldName="estimated", tagName = 'category'  ):
        m = MongoTxA(AllDataQuery(), db="SalaC2", collection="All"  )

        for i, r in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            cat = self.predicted[i].argsort()[-4:]
            est = [ {tagName: self.encoder.inverse_transform([j])[0]  , '%': self.predicted[i][j].astype(float)   }  for j in cat ]
            m.bufferUpdate(ObjectId(r['_id']), {fieldName: est})
        
        m.updateBuffertoServer()
        print('--> Listo')
        
    def lime(self, idx, vectorizer, encoder):
        from lime import lime_text
        from sklearn.pipeline import make_pipeline
        c = make_pipeline(vectorizer, self.model)


        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=encoder.classes_)
        exp = explainer.explain_instance(self.data.cleanText[idx], c.predict_proba, num_features=20, labels=range(38))
        print('Documento id: %d' % idx)
        print('Clase Predicha =', noTags_data.estimated[idx])
        # print(noTags_data.data['rawTxt'][idx])
        cls = 25
        print(noTags_data.data['archivo'][idx])
        print ('Explanation for class %s' % encoder.classes_[cls])
        print ('\n'.join(map(str, exp.as_list(label=cls))))
        exp.show_in_notebook(text=data.cleanText[idx], labels=(cls,))
        
    def saveAllPredicted2Db(self, fieldName="estimated", tagName = 'category'  ):
        m = MongoTxA(AllDataQuery(), db="SalaC2", collection="All"  )

        for i, r in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            cat = self.predicted[i].argsort()
            est = [ {tagName: self.encoder.inverse_transform([j])[0]  , '%': self.predicted[i][j].astype(float)   }  for j in cat ]
            m.bufferUpdate(ObjectId(r['_id']), {fieldName: est})
        
        m.updateBuffertoServer()
        print('--> Listo')  
        
        
        
def load_estimated():
    agg = [ {'$match': {'error':False, 'estimated': {'$exists': 1}}},
            {'$project': {'tema': '$estimated.tema', 'termino':'$estimated.termino'} } ]
    mongo = MongoTxA(CustomAggregationQuery(agg), db="SalaC2", collection="All"  )
        # Se obtiene los temas principales y secundarios
    dataT = pd.DataFrame([ doc for doc in tqdm(mongo.getCursor(0), total=mongo.getTotalDocs())])
    dataT['TemaPrincipal'] = dataT['tema'].apply(lambda doc: doc[3]['tema'])
    dataT['Porcentaje'] = dataT['tema'].apply(lambda doc: doc[3]['%'])
    dataT['TemaSecundario'] = dataT['tema'].apply(lambda doc: doc[2]['tema'])
    dataT['PorcentajeSecundario'] = dataT['tema'].apply(lambda doc: doc[2]['%'])
    return dataT



def intervalos_temas(dataT):
    # Se agrupan los porcentajes por tema e intervalos 
    d1 = dataT.groupby(['TemaPrincipal']).apply(lambda grp : grp['Porcentaje'].groupby(pd.cut( grp['Porcentaje'], np.arange(0, 1.001, 0.1))))
    # Se obtiene el valor relativo
    distAbs = d1.apply(lambda t: t.count())
    distRel = distAbs.apply( lambda v: v.apply(lambda d: d/sum(v)), axis=1 )

    # Se buscan las relaciones para el grafo
    dfMelt = pd.DataFrame(distRel.values, columns=list(distRel.columns.values.astype(str)))
    dfMelt['Tema'] = distRel.index.values
    dfMelt = dfMelt.melt(id_vars=['Tema'], value_name='Valor', var_name='Intervalo')
    # dfEdges = dfEdges[dfEdges['Probabilidad'] < 0.1]
    temas = distRel.index.values.astype(str)
    intervalos = list(distRel.columns.values.astype(str))
    
    colors =["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=distRel.values.min(), high=distRel.values.max())

    # Se intancia la figura
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"
    TITLE = "Distribución de Probabilidades para los diferentes temas estimados por el modelo"

    p = figure(title=TITLE, tools=TOOLS, toolbar_location='below',
                x_axis_location="above", plot_width=900, plot_height=900,
                x_range=temas, y_range=intervalos)


    # Opciones visuales varias 
    p.xaxis.major_label_orientation = 3.1416/3
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0

    # Los cuadros de color
    p.rect(x='Tema', y='Intervalo',  width=1, height=1, source=dfMelt, fill_color= {'field': 'Valor', 'transform': mapper})

    # La barra de color
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d docs"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    show(p)
    
    
    
    
def grafo_temas(data):
    # Se agrupan los porcentajes por tema e intervalos 
    d1 = data.groupby(['TemaPrincipal']).apply(lambda grp : grp['Porcentaje'].groupby(pd.cut( grp['Porcentaje'], np.arange(0, 1.001, 0.1))))
    # Se obtiene el valor relativo
    distAbs = d1.apply(lambda t: t.count())
    distRel = distAbs.apply( lambda v: v.apply(lambda d: d/sum(v)), axis=1 )
    temas = distRel.index.values.astype(str)
    # Buscamos las relaciones de probabilidad cercana en los documentos
    data['Diferencia'] = (data['Porcentaje']-data['PorcentajeSecundario']).abs()
    df1 = data.loc[data['Diferencia'] < 0.15]
    grafo = df1[['TemaPrincipal','TemaSecundario','Diferencia']]
    dg_group = grafo.groupby(['TemaPrincipal', 'TemaSecundario']).count().stack()
    # Se inicializa el grafo
    DG = nx.DiGraph()
    DG.add_nodes_from(temas)
    DG.add_weighted_edges_from(list(zip(dg_group.index.get_level_values(0), dg_group.index.get_level_values(1), 12* dg_group.values/ dg_group.values.max())))
    graph = GraphRenderer()

    plot = figure(title="Networkx Integration Demonstration", x_range=(-2,2), y_range=(-2,2), plot_width=1000, plot_height=1000,
                  tools="zoom_in, zoom_out", toolbar_location='below')

    colors = viridis(len(DG.nodes))

    graph_renderer  = from_networkx(DG, nx.spring_layout, scale=5, center=(0,0))

    hover = HoverTool(tooltips=[("Prob: ", "@index")])
    plot.add_tools(hover, TapTool(), BoxSelectTool())

    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color="colors")
    graph_renderer.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
    graph_renderer.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])
    graph_renderer.node_renderer.data_source.data['colors'] = colors
    graph_renderer.node_renderer.glyph.update(size=20, fill_color="colors")


    graph_renderer.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=1)
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width="weight", line_cap='round')
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width="weight")

    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()

    plot.renderers.append(graph_renderer)


    show(plot)