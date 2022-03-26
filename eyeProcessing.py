"""
Il seguente programma si occupa di mostrare un'animazione dell'andamento nel tempo degli eventi
contenuti nel file dvSaveEyesEvents.json, mentre successivamente viene mostrato l'andamento complessivo
delle polarità relative agli stessi eventi.
"""

import utilities.functions as ft
import numpy as np
import matplotlib.pyplot as plt

bBoxes = []
with open(ft.jsonPath+"eyesBoundingBoxes.json", 'r') as file:
    bBoxes = ft.load(file)
maxLE,maxRE = ft.getMaxMinBoundingBoxes(bBoxes)

stLE,timestampLE,stRE,timestampRE = [],[],[],[]
xDim,yDim = 346, 260
mat = np.zeros((yDim,xDim))
index, acc, sumLE, sumRE = 0, 0, 0, 0
step, initStep, valueStep = 0, 1, 0.05
with open(ft.jsonPath+"dvSaveEyesEvents.json", 'r') as file:
    data = ft.load(file)
    while index < len(data):
        acc,idx = 0, 0
        step = valueStep
        action = ""
        if index == 0:
            step = initStep
        while (index+acc < len(data)) and (data[index+acc]["timestamp"]<(data[index]["timestamp"]+step)):
            event = data[index+acc]
            if event["polarity"] == 1:
                mat[event["y"],event["x"]] += 1
            else:
                mat[event["y"],event["x"]] -= 1
            
            string = ft.recognizeEyesBoxes(event["x"],event["y"],maxLE,maxRE)
            if string == "leftEye":
                sumLE = int(np.sum(mat[maxLE[0][1]:maxLE[1][1],maxLE[0][0]:maxLE[1][0]]))
                stLE.append(sumLE)
                timestampLE.append(event["timestamp"])
            elif string == "rightEye":
                sumRE = int(np.sum(mat[maxRE[0][1]:maxRE[1][1],maxRE[0][0]:maxRE[1][0]]))
                stRE.append(sumRE)
                timestampRE.append(event["timestamp"])
            acc += 1
        plt.xlim(125,200)
        plt.ylim(130,100)
        plt.title("Eyes Events")
        plt.xlabel('\nrightEyeSum: '+str(sumRE)+', leftEyeSum: '+str(sumLE)+'\nt: '+str(event["timestamp"]))
        plt.matshow(mat,fignum=False,interpolation='nearest',cmap='plasma')
        plt.pause(1)
        index = index+acc+1
plt.close()

#il primo intervallo è un campione random
ft.printSumEyes(stLE,stRE,timestampLE,timestampRE,initStep, valueStep)