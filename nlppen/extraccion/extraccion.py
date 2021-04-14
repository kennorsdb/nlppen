import os
from multiprocessing import (Manager, Pool, freeze_support)
from time import sleep
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np

from Db import AllDataQuery, GetYearsQuery, MongoTxA
from ProcessException import ProcessException


class Process:
    def __init__(self):
        self.excludeExt = ['.tmp', '.db', '.lnk', '.thmx', '.bk',
                           '.xps', '.rf', '.xml', '.rcv', '.gif', '.jpg', '.dpj', '.rt_']
        self.FILES_PATH = "./data/raw/"

    def setNumbersOnCollection(self, collection):
        strgy = AllDataQuery(collection=collection)
        mongo = MongoTxA(strgy)
        mongo.setParallel(1)
        print('Total Documents: ' + str(mongo.getTotalDocs()))

        cursor = mongo.getCursor(0)
        i = 0
        for doc in cursor:
            mongo.bufferUpdate( doc['_id'], {'num' : i})
            i += 1

            if (i%10000) == 0:
                print(i)
        
        mongo.updateBuffertoServer()

    def cursorProcess(self, id, func, db, errors, stats):
        """
        Procesa un único cursor.
        """
        processed = 0
        startTimer = timer()
        bdTimer = timer()
        cursor = db.getCursor(id)
        cursor.batch_size(20)
        errorsCount = 0
        lastProcessed = 0
        lastTime = 0
        for doc in cursor:
            bdlookup = timer() - bdTimer
            try:
                processed += 1
                # Se procesa el documento
                result = func(doc)
                # print(processed)
                # Se registran los cambios
                if result is not None: 
                    db.bufferUpdate( doc['_id'], result)
                    db.updateBuffertoServer()
            # Se registran los errores
            except ProcessException as nE:
                errorsCount += 1
                errors[doc['anno']].append({
                        '_id': doc['_id'],
                        'archivo': doc['archivo'],
                        'errores': nE.error,
                        'textdoc': doc['rawTxt']
                    })
            # Se actualizan las estadísticas
            if processed % (10*id + 5) == 0:
                stats[id] = {
                    'processed': processed,
                    'totalTime': timer() - startTimer,
                    'batchVelocity': (processed-lastProcessed)/(timer()-lastTime),
                    'errors': errorsCount,
                    'running': True,
                    'bdlookup': bdlookup,
                    'batch': db.getBatchSize()
                }
                lastProcessed = processed
                lastTime = timer()
            bdTimer = timer()
        
        # Se guardan los documentos actualizados
        db.updateBuffertoServer()

        # Se retornan los datos
        stats[id] = {
            'processed': processed,
            'totalTime': timer() - startTimer,
            'errors': errorsCount,
            'running': False,
            'bdlookup': 0,
            'batch': db.getBatchSize(),
            'batchVelocity': 0
        }
        print(str(id) + "--: Stop")
        return stats[id]


    def startProcess(self, func, queryStrgy, parallel=10, wrtFile=False, debug=True):
        # Inicialización del manager
        manager = Manager()
        # Timer principal
        startTimer = timer()
        # Se inicializan las variables del proceso
        stats = manager.list()  # lista sincronizada
        for _ in range(parallel):
            stats.append({})
        print('Anos')
        errors = manager.dict()  # Errores del proceso por año
        yearsCursor = MongoTxA(GetYearsQuery()).getCursor(0) 
        years = {}
        for d in yearsCursor:
            errors[d['anno']] = manager.list()


        print('Consulta')
        # Se consulta la cantidad de documentos del query
        mongo = MongoTxA(queryStrgy, db='SalaC2')
        mongo.setParallel(parallel)
        
        if debug:
            print('Total Documents: ' + str(mongo.getTotalDocs()))

        print('Total')
        # Se crean los procesos
        if parallel == 1:
            self.cursorProcess(0,func, mongo, errors,stats)
        else:    
            pool = Pool(processes=parallel+1)
            pool.apply_async(self.stadistics, (stats, ))
            processList = [pool.apply_async(self.cursorProcess, (id, func, mongo , errors, stats, )) for id in range(parallel)]
            # Se espera a que todos los procesos terminen
            print(processList)
            output = [p.get() for p in processList]
        print('fin')

        ret = None

        if debug:
            print('Saving errors...')
        
        ret = self.saveErrors(errors,wrtFile)
        
        if debug:
            print('Time -> ' + str(startTimer - timer()))
            print(str(parallel) + '--> Total time: ' + str(timer()-startTimer))

        return ret 

    def stadistics(self, stats):
        processed = [0 for x in stats]
        velocity = [0 for x in stats] 
        allVelocity = []
        errors = [0 for x in stats]
        batch = [0 for x in stats]
        running = [True for x in stats]
        bdlookup = [0 for x in stats]
        ids = range(len(stats))
        plt.show()
        
        while(any(running)):
            for i in ids:
                if  stats[i] != {}:
                    processed[i] = stats[i]['processed']
                    batch[i] = stats[i]['batch']-processed[i]
                    totalTime = stats[i]['totalTime']
                    velocity[i] = stats[i]['batchVelocity']
                    errors[i] = stats[i]['errors']
                    running[i] = stats[i]['running']
                    bdlookup[i] = stats[i]['bdlookup']
        
            
            totalDocs = np.sum(processed) 
            allVelocity.append(velocity.copy())    

            plt.subplot(2, 1, 1)  
            plt.bar(ids, processed)        
            plt.bar(ids, batch, bottom=processed)        
            plt.title('Documentos Procesados')
            plt.ylabel('cantidad') 

            plt.subplot(2, 1, 2)  
            temp = np.array(allVelocity).T  
            for i in temp:
                plt.plot(np.arange(i.size), i)
            
            plt.title('Documentos Procesados')
            plt.ylabel('cantidad')       

            plt.pause(1)           
            plt.gcf().clear()  

    def saveErrors(self, errors, wrtFile=False):
        ret = []
        errorStats = '|  Año  | Total Documentos | Cantidad Errores | Porcentaje de Error | \n' \
                     '| ----- | ---------------- | ---------------- | ------------------- | \n'
        yearsCursor = MongoTxA(GetYearsQuery()).getCursor(0) 
        years = {}
        for d in yearsCursor:
            years[d['anno']] = d['cantidad']

        print(years)

        for anno in errors._getvalue():
            errorsTxt = ''
            # if(len(errors[anno]) != 0):
            #     print(anno + " -> " + str(len(errors[anno])) + " errores")
            if wrtFile:
                for er in errors[anno]:
                    errorsTxt = errorsTxt + "\n\n===============\n"
                    errorsTxt = errorsTxt + er['archivo']
                    errorsTxt = errorsTxt + "\n===============\n"
                    errorsTxt = errorsTxt + er['textdoc']
            
                file = open("errores\\" +  anno + "-errores.txt", "wb")
                file.write(errorsTxt.encode("utf-8"))
                file.close()

            errorStats = errorStats + '| {0:5} | {1:16.0f} | {2:16} | {3:18.1f}% |\n' \
                             .format(anno, years[anno], len(errors[anno]), 100*(len(errors[anno])/years[anno]))
            ret.append({'anno': anno, 'TotalDocs': years[anno], 'errores': len(errors[anno]), 'tasaError': 100*(len(errors[anno])/years[anno]) })

        #if wrtFile:
        file = open("errores\\00-stats.md", "wb")
        file.write(str(errorStats).encode("utf-8"))
        file.close()

        return ret
