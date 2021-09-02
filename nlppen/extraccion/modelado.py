from joblib import Memory
import numpy as np
import pickle

# Herramientas de cálculo
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Redes Neuronales
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import tensorflow as tf

# Rutinas de Data Scaling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours


# Inicialización de algunos módulos
cachedir = './.pycache'
memory = Memory(cachedir, verbose=0)


class NNModel:
    def __init__(self, corpus, tags):
        self.corpus = corpus
        self.tags = tags


    def vectorizarCorpus(self, max_df=0.95, min_df=2, max_features=4000):
        print('Creando vector del corpus')
        self.vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features)
        corpus = self.vectorizer.fit_transform(self.corpus)
        print('Normalizando vectores usando Tf-Idf')
        self.tfidf_transformer = TfidfTransformer()
        self.corpus = self.tfidf_transformer.fit_transform(corpus)


    def label_encoder(self):
        print('Label Encoder')
        self.encoder = LabelEncoder()
        y_int = self.encoder.fit_transform(self.tags)
        self.y_binary = to_categorical(y_int)

    @staticmethod
    @memory.cache
    def dataScaling(corpus, tags):
        print('OverSampling usando SMOTE ')
        smote =  SMOTE('minority', n_jobs=-1)
        corpus, y_binary = smote.fit_resample(corpus, tags)
        print('---> OverSampling final en ' + str(corpus.shape[0]) + ' muestras')

        print('UnderSampling usando ENN')
        enn = EditedNearestNeighbours('not minority', n_jobs=-1)
        corpus, y_binary = enn.fit_resample(corpus, y_binary)
        print('---> UnderSampling final en ' + str(corpus.shape[0]) + ' muestras')
        return corpus, y_binary

    def data_split(self):
        print('Separando los datos de entrenamiento y de prueba ')
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split( self.corpus, self.y_binary, test_size=0.20)
        print('Normalizando los datos')
        self.x_train = self.x_train.astype('float16')
        self.x_test = self.x_test.astype('float16')
        self.inputLayerDim = self.x_train.shape[1]
        self.ouputLayerDim = self.y_binary.shape[1]
        print('---> InputLayer: ' + str(self.inputLayerDim) + '  --  OutputLayer: ' + str(self.ouputLayerDim))
        

    def data_preparation(self):
        self.vectorizarCorpus()
        self.label_encoder()
        self.corpus, self.y_binary = self.dataScaling(self.corpus.toarray(), self.y_binary)
        self.data_split()
        self.buildModel()


    def buildModel(self):

        print('Construyendo el modelo')
        self.model = Sequential()
        self.model.add(Dense(units=800, activation='relu', input_dim=self.inputLayerDim))
        self.model.add(Dense(units=200, activation='relu' ))
        self.model.add(Dense(units=300, activation='relu'))
        self.model.add(Dense(units=400, activation='relu'))
        self.model.add(Dense(units=100, activation='relu'))
        self.model.add(Dense(units=self.ouputLayerDim, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', 
            metrics=[ 
                'accuracy',
                keras.metrics.categorical_accuracy, 
                keras.metrics.top_k_categorical_accuracy, 
                keras.metrics.Recall(), 
                keras.metrics.AUC(name='auc')
                ])
        
    def fit(self, verbose=0):
        print('Iniciando entrenamiento')
        self.train_history = self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=verbose, validation_split=0.2)
        print('Entrenamiento finalizado')

    def test(self, plotConfusionMatrix=False, encoder=None):
        print('Pruebas de rendimiento: ')
        loss_and_metrics = self.model.evaluate(self.x_test, self.y_test, batch_size=32, verbose=0)
        print('Pruebas de rendimiento: Loss: ' + str(loss_and_metrics[0]) + ' Categorical Accuracy:' + str(loss_and_metrics[1]))
        self.y_pred_model = self.model.predict(self.x_test, batch_size=64, verbose=0)
        self.y_pred_bool = np.argmax(self.y_pred_model, axis=1)
        report = classification_report( np.argmax(self.y_test, axis=1), self.y_pred_bool)
        print(report)
        # Se incorporan las muestras del test al modelo
        self.model.fit(self.x_test, self.y_test, epochs=5, batch_size=32, verbose=0)

        
    def load_model(self, model_file, encoder_file, vectorizer_file, tfidf_file):
        self.model = keras.models.load_model(model_file)
        with open(encoder_file,'rb') as infile:
            self.encoder = pickle.load(infile)
        
        with open(vectorizer_file,'rb') as infile:
            self.vectorizer = pickle.load(infile)
        
        with open(tfidf_file,'rb') as infile:
            self.tfidf_transformer = pickle.load(infile)
        
         
    
    def save_model(self, model_file, encoder_file, vectorizer_file, tfidf_file):
        self.model.save(model_file)
        
        with open(encoder_file,'wb') as output:
            pickle.dump(self.encoder, output)
        
        with open(vectorizer_file,'wb') as output:
            pickle.dump(self.vectorizer, output)
        
        with open(tfidf_file,'wb') as output:
            pickle.dump(self.tfidf_transformer, output)


    def get_model(self):
        return self.model


    def predict(self, corpus_pred, verbose=0):
        self.corpus_pred = self.vectorizer.transform(corpus_pred)
        self.corpus_pred = self.tfidf_transformer.transform(self.corpus_pred)

        self.predicted = self.model.predict(self.corpus_pred.toarray(), batch_size=32, verbose=verbose)
        self.estimated = self.encoder.inverse_transform(self.predicted.argmax(axis=1))
        
