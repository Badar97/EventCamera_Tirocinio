import utilities.functions as ft
import numpy as np
import matplotlib.pyplot as plt

bBoxes = []
with open(ft.jsonPath+"lipsBoundingBoxes.json", 'r') as file:
    bBoxes = ft.load(file)
maxLI = ft.getLipsMaxMinBoundingBoxes(bBoxes)
stLI,timestampLI = [],[]
xDim,yDim = 346, 260
mat = np.zeros((yDim,xDim))
index, acc, sumLI = 0, 0, 0
step, initStep, valueStep = 0, 1, 0.5
with open(ft.jsonPath+"dvSaveLipsEvents.json", 'r') as file:
    data = ft.load(file)
    while index < len(data):
        acc,idx = 0, 0
        step = valueStep
        mat = np.zeros((yDim,xDim))
        action = ""
        if index == 0:
            step = initStep
        while (index+acc < len(data)) and (data[index+acc]["timestamp"]<(data[index]["timestamp"]+step)):
            event = data[index+acc]
            if event["polarity"] == 1:
                mat[event["y"],event["x"]] += 1
            else:
                mat[event["y"],event["x"]] -= 1

            string = ft.recognizeLipsBoxes(event["x"],event["y"],maxLI)
            if string == "lips":
                sumLI = int(np.sum(mat[maxLI[0][1]:maxLI[1][1],maxLI[0][0]:maxLI[1][0]]))
                stLI.append(sumLI)
                timestampLI.append(event["timestamp"])
            acc += 1

        plt.xlim(125,200)
        plt.ylim(180,155)
        plt.title("Events")
        plt.xlabel('\nlipsSum: '+str(sumLI)+'\nt: '+str(event["timestamp"]))
        plt.matshow(mat,fignum=False,interpolation='nearest',cmap='plasma')
        plt.pause(1)

        index = index+acc+1
plt.close()

#il primo intervallo e un campione random
ft.printLipsTot(stLI,timestampLI,initStep,valueStep)
