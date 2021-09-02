# -*- coding: utf-8 -*-
# Librería de mongoDB donde se va a guardar los logs y los encabezados.
from pymongo import MongoClient, CursorType, UpdateOne
import abc  # Python's built-in abstract class library
import math

class QueryStrategy:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def query(self, collection, skip):
        pass


class AllDataQuery(QueryStrategy):
    def query(self, collection, skip=0, limit=0):
        return collection.find({'num': {'$gte': skip}},{'archivo':1, 'txt':1, 'anno':1, 'num':1, 'error':1, 'secciones':1, 'texts.html':1}, no_cursor_timeout=True).limit(limit)

class PreExpedientesQuery(QueryStrategy):
    def query(self, collection, skip=0, limit=0):
        return collection.find({'num': {'$gte': skip}},{'archivo':1, 'txt':1, 'anno':1, 'num':1, 'texts.tika':1, 'error':1}).limit(limit)


class PreEntitiesQuery(QueryStrategy):
    def query(self, collection, skip=0, limit=0):
        return collection.find({'entidades': {'$exists': 0}},{'txt':1, 'archivo':1, 'error':1, 'anno':1}, no_cursor_timeout=True).sort("_id", -1).skip(skip).limit(limit)
#         return collection.find({'entidades': {'$exists': 0}},{'txt':1, 'archivo':1, 'error':1, 'anno':1}).sort("_id", -1).skip(skip).limit(limit)
#         return collection.find({'entidades': {'$exists': 0}},{'textos':1, 'archivo':1, 'error':1, 'anno':1}).sort("_id", -1).skip(skip).limit(limit)

class YearQuery(QueryStrategy):
    def __init__(self, year):
        self.year = year

    def query(self, collection, skip=0, limit=0):
        return collection.find({'anno': str(self.year)}).skip(skip).limit(limit)


class DocQuery(QueryStrategy):
    def __init__(self, docName):
        self.docName = docName

    def query(self, collection, skip=0, limit=0):
        return collection.find({'archivo': self.docName})

    
class CustomFindQuery(QueryStrategy):
    def __init__(self, queryCustom={}, project=None):
        self.queryCustom = queryCustom
        self.projectCustom = project

    def query(self, collection, skip=0, limit=0):
        return collection.find(self.queryCustom, self.projectCustom)

class IdQuery(QueryStrategy):
    def __init__(self, id):
        self.id = id

    def changeId(self, id):
        self.id = id

    def query(self, collection, skip=0, limit=0):
        return collection.find_one({'_id': self.id})

class PathQueryById(QueryStrategy):
    def __init__(self, id):
        self.id = id

    def changeId(self, id):
        self.id = id

    def query(self, collection, skip=0, limit=0):
        return collection.find_one({'_id': self.id}, {'_id': True, 'path': True})
    
class ExpedienteQuery(QueryStrategy):
    def __init__(self, expediente):
        self.expediente = expediente

    def changeExpediente(self, expediente):
        self.expediente = expediente

    def query(self, collection, skip=0, limit=0):
        return collection.find_one({'expediente': self.expediente})
    
    
class CustomAggregationQuery(QueryStrategy):
    def __init__(self, query ):
        self.querydoc = query

    def changeQuery(self, query):
        self.querydoc = query

    def query(self, collection, skip=0, limit=0):
        return collection.aggregate(self.querydoc)


class GetYearsQuery(QueryStrategy):
    def query(self, collection, skip=0, limit=0):
        return collection.aggregate(
            [
                {"$match": {"error" : False}},
                {"$match": {"anno": {"$ne": None}}},
                {"$sort": { "anno" : 1}},
                {"$group": {"_id": {"anno": "$anno"}, "cantidad": {"$sum": 1.0}}},
                {"$addFields": {"anno": "$_id.anno"}},
                {"$project": {"_id": 0.0,  "anno": 1.0, "cantidad": 1.0}}
            ]
        )

class GetBadReadQuery(QueryStrategy):
    def query(self, collection, skip=0, limit=0):
        return collection.aggregate(
            [
                {"$sort": {"anno": 1}},
                {"$group": {"_id": "$anno", "count": {"$sum": 1}, "docs": {"$push": "$procesar"}}},
                {"$project": {"_id": 0, "anno" : "$_id", "count": 1, "errors": {"$size": {"$filter": {"input": "$docs", "cond": {"$eq": ["$$this", False]}}}}}},
                {"$sort": {"anno" : 1}}
            ]
        )

class GetExtractedCount(QueryStrategy):
    def query(self, collection, skip=0, limit=0):
        return collection.aggregate(
                    [{"$project": {"expediente": {"$cond": [ "$expediente", 1.0, 0.0 ] },
                                "fechahora": {"$cond": [ "$extraccion.fechahora", 1.0, 0.0 ] },
                                "redactor": {"$cond": [ "$extraccion.redactor", 1.0, 0.0 ] },
                                "conVotoSalvado": {"$cond": [ "$extraccion.conVotoSalvado", 1.0, 0.0 ] },
                                "recurrente": {"$cond": [ "$extraccion.recurrente", 1.0, 0.0 ] },
                                "recurrido": {"$cond": [ "$extraccion.recurrido", 1.0, 0.0 ] },
                                "sentencia": {"$cond": [ "$extraccion.sentencia", 1.0, 0.0 ] },
                                "tipoResolucion": {"$cond": [ "$extraccion.tipoResolucion", 1.0, 0.0 ] }
                    }},
                    {"$group": {"_id": 0.0, "Total": {"$sum": 1.0 },
                                "Expedientes": {"$sum": "$expediente" },
                                "Fecha y Hora": {"$sum": "$fechahora" },
                                "Redactor": {"$sum": "$redactor" },
                                "Con Voto Salvado": {"$sum": "$conVotoSalvado" },
                                "Recurrente": {"$sum": "$recurrente" },
                                "Recurrido": {"$sum": "$recurrido" },
                                "Sentencia": {"$sum": "$sentencia" },
                                "Tipo de Resolución": {"$sum": "$tipoResolucion" }
                    }}, {'$project': {'_id': 0}}])

    
class MuestraQuery(QueryStrategy):
    def __init__(self, cantidad):
        self.isFind = True
        self.cantidad = cantidad

    def query(self, collection, skip=0, limit=0):
        # return collection.aggregate([{'$match': {'procesar': True}},{ '$sample': { 'size': self.cantidad } }, {'$project': {'rawText': 1, 'archivo': 1}}])
        return collection.aggregate([{ '$sample': { 'size': self.cantidad } }, {'$project': {'rawText': 1, 'cleanText':1, 'archivo': 1, 'encabezado':1}}])
                        


class GetCategoriesCount(QueryStrategy):
    def query(self, collection, skip=0, limit=0):
        return collection.find({'_id': {'$ne': None}})

    

class MongoTxA:
    def __init__(self, strgy: QueryStrategy, MONGOURI='mongodb://172.17.0.1:27017/', db='SalaC', collection='All', preload=True):
        self.MONGOURI = MONGOURI
        self.dbName = db
        self.collectionName = collection
        self.mongo = MongoClient(self.MONGOURI)
        self.db = self.mongo[self.dbName]
        self.docsDB = self.mongo[self.dbName][self.collectionName]
        self.activeCursor = None
        self.queryStrategy = strgy
        self.parallel = 1
        if preload and hasattr(self.queryStrategy.query(self.docsDB), 'count'):
            self.totalDocs = self.queryStrategy.query(self.docsDB).count()
        else:
            self.totalDocs = 0
        self.batchSize = self.totalDocs // self.parallel
        self.closeConnection()
        self.bulkBuffer = []

    def setParallel(self, parallel):
        self.parallel = parallel
        self.batchSize = math.ceil(self.totalDocs // parallel) + 1

    def getBatchSize(self):
        return self.batchSize

    def getAllDocs(self):
        if self.mongo is None:
            self.mongo = MongoClient(self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
        return self.queryStrategy.query(self.docsDB, 0)

    def getTotalDocs(self):
        return self.totalDocs

    def closeConnection(self):
        if self.activeCursor is not None:
            self.activeCursor.close()
            self.activeCursor = None
        if self.mongo is not None:
            self.mongo.close()
            self.db = None
            self.docsDB = None
            self.mongo = None

    def getCursor(self, id=0):
        if self.mongo is None:
            self.mongo = MongoClient(self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
        return self.queryStrategy.query(self.docsDB, id * self.batchSize, self.batchSize)

    def bufferUpdate(self, _id, doc):
        self.bulkBuffer.append(UpdateOne({'_id': _id}, {'$set': doc}))

    def updateBuffertoServer(self):
        if self.mongo is None:
            self.mongo = MongoClient(self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
            
        if len(self.bulkBuffer) != 0:
            self.docsDB.bulk_write(self.bulkBuffer)
            self.bulkBuffer = []
            
    def saveDoc(self, _id, doc ):
        if self.mongo is None:
            self.mongo = MongoClient(self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
            
        self.docsDB.update_one( {'_id': _id}, {'$set': doc})

    def deleteFields(self, _id, doc ):
        if self.mongo is None:
            self.mongo = MongoClient(self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
            
        self.docsDB.update_one( {'_id': _id}, {'$unset': doc})

    def insert_many(self, archivos):
        if self.mongo is None:
            self.mongo = MongoClient (self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
        self.docsDB.insert_many(archivos)

    def insertDoc(self, doc):
        if self.mongo is None:
            self.mongo = MongoClient(self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
        self.docsDB.insert_one(doc)

    def deleteDoc(self, _id):
        if self.mongo is None:
            self.mongo = MongoClient(self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
        
        self.docsDB.delete_one( {'_id': _id})

    def getAll(self):
        return [doc for doc in self.getCursor()]

    def create_index(self, field):
        if self.mongo is None:
            self.mongo = MongoClient(self.MONGOURI)
            self.db = self.mongo[self.dbName]
            self.docsDB = self.mongo[self.dbName][self.collectionName]
        self.docsDB.create_index(field)