import utilities.functions as ft
import numpy as np
import matplotlib.pyplot as plt

bBoxes = []
with open(ft.jsonPath+"totBoundingBoxes.json", 'r') as file:
    bBoxes = ft.load(file)
maxLE,maxRE,maxLI = ft.getTotMaxMinBoundingBoxes(bBoxes)
stLE,timestampLE,stRE,timestampRE,stLI,timestampLI = [],[],[],[],[],[]
xDim,yDim = 346, 260
mat = np.zeros((yDim,xDim))
index, acc, sumLE, sumRE, sumLI = 0, 0, 0, 0, 0
step, initStep, valueStep = 0, 1, 1
with open(ft.jsonPath+"dvSaveTotEvents.json", 'r') as file:
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

            string0 = ft.recognizeLEBoxes(event["x"],event["y"],maxLE)
            string1 = ft.recognizeREBoxes(event["x"],event["y"],maxRE)
            string2 = ft.recognizeLipsBoxes(event["x"],event["y"],maxLI)
            if string0 == "leftEye":
                sumLE = int(np.sum(mat[maxLE[0][1]:maxLE[1][1],maxLE[0][0]:maxLE[1][0]]))
                stLE.append(sumLE)
                timestampLE.append(event["timestamp"])
            if string1 == "rightEye":
                sumRE = int(np.sum(mat[maxRE[0][1]:maxRE[1][1],maxRE[0][0]:maxRE[1][0]]))
                stRE.append(sumRE)
                timestampRE.append(event["timestamp"])
            if string2 == "lips":
                sumLI = int(np.sum(mat[maxLI[0][1]:maxLI[1][1],maxLI[0][0]:maxLI[1][0]]))
                stLI.append(sumLI)
                timestampLI.append(event["timestamp"])
            acc += 1

        plt.xlim(125,200)
        plt.ylim(180,100)
        plt.title("Events")
        plt.xlabel('\nrightEyeSum: '+str(sumRE)+', leftEyeSum: '+str(sumLE)+', lipsSum: '+str(sumLI)+'\nt: '+str(event["timestamp"]))
        plt.matshow(mat,fignum=False,interpolation='nearest',cmap='plasma')
        plt.pause(1)
        index = index+acc+1
plt.close()
ft.printSumTot(stLE,stRE,stLI,timestampLE,timestampRE,timestampLI,initStep,valueStep)
