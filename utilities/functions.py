"""
Il seguente file contiene tutte le funzioni utilizzate per la realizzazione del programma complessivo
"""

import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import utilities.landmarks as ld


'''general'''

#path della directory di lavoro
genPath = os.getcwd()+"/"
jsonPath = os.getcwd()+"/code/json/"

#elems: 0 = RightEye, 1 = Lips, 2 = LeftEye
numElems = 3
numEye,numLips = ld.getCoupleEyes(),ld.getCoupleLips()

#in primo luogo: x = 0 e y = 1, poi x= "modSpeed" e y = "direction"
numCoords = 2

#range nell'analisi dei timestamp di interesse
epsilon = 0.001

#Formattazione del json
def toJson(struct, cls = None):
        return json.dumps(struct,indent=2, sort_keys=True,cls = cls)

#converte oggetto json in oggetto python
def load(file):
    return json.load(file)

def writeJson(elems,fileName,cls = None):
    with open(jsonPath+fileName+".json","w") as file:
        file.write(toJson(elems,cls))

'''for faceMeshing.py'''

def calculateKeypoints(face_landmarks):
    dict,app,landmark_list = {},{},[]
    rightEye, leftEye, lips, silhouette = [], [], [], []
    re, le, li = [], [], []
    for l in ld.LANDMARKS:
        landmark = ld.recognizeLandmark(l)
        if(landmark == "leftEye"):
            leftEye.append(face_landmarks.landmark[l])
            app["x"], app["y"] = face_landmarks.landmark[l].x, face_landmarks.landmark[l].y
            le.append(app)
        if(landmark == "lips"):
            lips.append(face_landmarks.landmark[l])
            app["x"], app["y"] = face_landmarks.landmark[l].x, face_landmarks.landmark[l].y
            li.append(app)
        if(landmark == "rightEye"):
            rightEye.append(face_landmarks.landmark[l])
            app["x"], app["y"] = face_landmarks.landmark[l].x, face_landmarks.landmark[l].y
            re.append(app)
        if(landmark == "silhouette"):
            silhouette.append(face_landmarks.landmark[l])
        app = {}
        dict["rightEye"], dict["leftEye"], dict["lips"] = re, le, li
        landmark_list.append(leftEye)
        landmark_list.append(lips)
        landmark_list.append(rightEye)
        landmark_list.append(silhouette)

    return dict, landmark_list

'''for faceProcessing.py'''

#ritorna una lista contenente tutti i valori dei timestamps dell'oggetto python
def getTimestamps(data):
    timestamps = []
    for i in range(0,len(data)):
        timestamps.append(data[i]["timestamp"])
    return timestamps

#ritorna il numero di coppie di keypoints per ogni landmark dell'oggetto passato come parametro
#ordine della lista: 0=LeftEye, 1=Lips, 2=RightEye
def getCoupleCoords():
    return [numEye, numLips, numEye]

#matrice contenente x,y per ogni timestamp per ogni coppia del landmark
def getXYValues(data,numCoupleCoords,landmark):
    mat = np.zeros((len(data),numCoupleCoords,numCoords))
    for i in range(0,len(data)):
        for j in range(0,numCoupleCoords):
            mat[i,j,0], mat[i,j,1] = data[i][landmark][j]["x"], data[i][landmark][j]["y"]
    return mat

#calcolo della velocità mediante il metodo di Eulero con differenze centrali
def vel(mat,i,k,x,timestamps):
    #necessario per l'analisi ridotta, se t+1 == 0 e t-1!=0 e viceversa, ottengo valori delle derivate
    #sballati
    if (mat[i+1,k,x]== 0 and mat[i-1,k,x]!=0) or (mat[i+1,k,x]!=0 and mat[i-1,k,x] == 0):
        return 0
    else:
        return ((mat[i+1,k,x] - mat[i-1,k,x])/(2*(timestamps[i+1]-timestamps[i-1])))

#calcolo della direzione 
def dir(x,y):
    return math.atan2(y,x)

#restituisce una lista contentente i valori delle derivate e delle direzioni a partire dalle matrici
#per ogni timestamp per ogni coppia di landmarks
def calculateDerivatives(mat1,mat2,mat3,data,timestamps,coupleCoords):
    der = []
    dict = {}
    for i in range(1,len(data)-1):
        dict["timestamp"] = timestamps[i]
        le, li, re = [], [], []
        #per ogni landmarks (LE,LI,RE)
        for j in range(0,numElems):
            #per ogni coppia di keypoints k presenti nel landmark j
            for k in range(0,coupleCoords[j]):              
                if j == 0:
                    le.append(stats(mat1,i,k,timestamps))
                if j == 1:
                    li.append(stats(mat2,i,k,timestamps))
                if j == 2:
                    re.append(stats(mat3,i,k,timestamps))
        #aggiungo le informazioni relative all'elemento i nel dizionario
        dict["leftEye"], dict["lips"], dict["rightEye"] = le, li, re
        #aggiungo il dizionario alla lista contenente tutte le informazioni di tutti gli elementi
        der.append(dict)
        dict = {}
    return der

#restituisce un dizionario contenente modulo/direzione della velocità
def stats(mat,i,k,timestamps):
    app = {}
    vX = vel(mat,i,k,0,timestamps)
    vY = vel(mat,i,k,1,timestamps)
    app["modSpeed"] =  math.sqrt(math.pow(vX,2)+math.pow(vY,2))
    app["direction"] = dir(vX,vY)
    return app

#matrice contenente i valori delle velocità e della direzione
def getDerValues(data,numCoupleCoords,landmark):
    mat = np.zeros((len(data),numCoupleCoords,numCoords))
    for i in range(0,len(data)):
        for j in range(0,numCoupleCoords):
            mat[i,j,0], mat[i,j,1] = data[i][landmark][j]["modSpeed"], data[i][landmark][j]["direction"]
    return mat

#Grafici relativi al modulo e la direzione della velocità per ogni timestamps per ogni keypoints del landmark
def printGraph(mat,n,timestamps,name):
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.suptitle(name)
    ax1.set_ylabel('Y - Modulo Velocità')
    for i in range(0,n):
        ax1.plot(np.array(timestamps),np.array(mat[:,i,0]))
    ax2.set_xlabel('X - Time')
    ax2.set_ylabel('Y - Direzione')
    for i in range(0,n):
        ax2.plot(np.array(timestamps),np.array(mat[:,i,1]))
    plt.show()

'''for filteredProcessing.py'''

#filtraggio del contenuto in base al json analyze.json
def filterValues(value,redTimestamps):
    for i in range(0,len(redTimestamps)):
        if ( (value >= (redTimestamps[i,0] - epsilon)) and (value <= (redTimestamps[i,1] + epsilon)) ):
            return True
    return False

#matrice con n righe quante sono gli elementi del file, due colonne: 0 = inizio intervallo, 3 = fine intervallo di interesse
def getReducedTimestamps(string):
    with open(jsonPath+string+".json", 'r') as file:
        events = 3
        data = json.load(file)
        mat = np.zeros((len(data),events))
        for i in range(0,len(data)):
            for actions in data[i]["events"]:
                if(actions["event"] == "open") or (actions["event"] == "random0"):
                    mat[i,0] = actions["timestamp"]
                if(actions["event"] == "close") or (actions["event"] == "random1"):
                    mat[i,1] = actions["timestamp"]
                if(actions["event"] == "reOpen") or (actions["event"] == "random2"):
                    mat[i,2] = actions["timestamp"]
    return mat

#filtro le coppie di coordinate x,y della matrice in base al timestamp se presente o meno nella lista dei frame da analizzare
def getXYFilteredValues(data,numCoupleCoords,redTimestamps,landmark):
    mat = np.zeros((len(data),numCoupleCoords,numCoords))
    for i in range(0,len(data)):
        boolean = filterValues(data[i]["timestamp"],redTimestamps)
        if boolean:
            for j in range(0,numCoupleCoords):
                mat[i,j,0], mat[i,j,1] = data[i][landmark][j]["x"], data[i][landmark][j]["y"]
    return mat

#filtro le coppie di coordinate x,y della matrice in base al loro valore, se = 0 li oscuro per il print
def getNanValues(data,numCoupleCoords,landmark):
    mat = np.zeros((len(data),numCoupleCoords,numCoords))
    for i in range(0,len(data)):
        for j in range(0,numCoupleCoords):
            mat[i,j,0], mat[i,j,1] = data[i][landmark][j]["modSpeed"], data[i][landmark][j]["direction"]
            #se i valori sono 0, li oscuro per il print
            if data[i][landmark][j]["modSpeed"] == 0:
                mat[i,j,0] = np.nan
            if data[i][landmark][j]["direction"] == 0:
                mat[i,j,1] = np.nan
    return mat

#restituisce una lista contentente i valori delle derivate e delle direzioni a partire dalle matrici, per ogni timestamp, per ogni coppia di landmarks
def calculateRedDerivatives(mat1,mat2,mat3,data,timestamps,coupleCoords):
    der = []
    dict = {}
    for i in range(1,len(data)-1):
        dict["timestamp"] = timestamps[i]
        le, li, re = [], [], []
        #per ogni landmarks (LE,LI,RE)
        for j in range(0,numElems):
            #per ogni coppia di keypoints k presenti nel landmark j
            for k in range(0,coupleCoords[j]):              
                if j == 0:
                    le.append(stats(mat1,i,k,timestamps))
                if j == 1:
                    li.append(stats(mat2,i,k,timestamps))
                if j == 2:
                    re.append(stats(mat3,i,k,timestamps))
        #aggiungo le informazioni relative all'elemento i nel dizionario
        dict["leftEye"], dict["lips"], dict["rightEye"] = le, li, re
        #aggiungo il dizionario alla lista contenente tutte le informazioni di tutti gli elementi
        der.append(dict)
        dict = {}
    return der

'''for eyeMeshing.py'''

#necessario per l'encoding
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#valori delle coordinate (x,y) oer ogni coppia di landmark, relativo solo agli occhi
def getEyesValues(face_landmarks,redTimestamps,timestamp):
    dict,app = {},{}
    re, le = [], []
    boolean = filterValues(timestamp,redTimestamps)
    if boolean:
        for l in ld.LANDMARKS:
            if(ld.recognizeLandmark(l) == "leftEye"):
                app["x"], app["y"] = face_landmarks.landmark[l].x, face_landmarks.landmark[l].y
                le.append(app)
            if(ld.recognizeLandmark(l) == "rightEye"):
                app["x"], app["y"] = face_landmarks.landmark[l].x, face_landmarks.landmark[l].y
                re.append(app)
            app = {}
            dict["rightEye"], dict["leftEye"] = re, le
    return dict

#bounding boxes relativi agli occhi
def getMaxMinXYValues(dict,type,image):
    delta = 0
    mat = np.zeros((1,numEye,numCoords))
    for k in range(0,len(dict[type])):
        elem = dict[type][k]
        mat[0,k,0],mat[0,k,1] = elem["x"],elem["y"]
    XMin, XMax, YMin, YMax = mat[:,:,0].min(), mat[:,:,0].max(), mat[:,:,1].min(), mat[:,:,1].max()  
    start,stop = (int(XMin*image.shape[1]-delta),int(YMin*image.shape[0]+delta)), (int(XMax*image.shape[1]+delta),int(YMax*image.shape[0]-delta))
    return start,stop

#filtraggio del contenuto in base ai timestamp presenti in analyze.json, file contenente gli intervalli da analizzare
def filterEventTimestamps(value,redTimestamps,c):
    for i in range(c,len(redTimestamps)):
        #se il valore è compreso nell'intervallo, ho finito la ricerca
        if (value >= (redTimestamps[i,0] - epsilon) and value <= (redTimestamps[i,2] + epsilon)):
            return c,True
        #altrimenti se il valore è maggiore dell'estremo superiore, potrebbe essere nell'intervallo successivo
        elif value > (redTimestamps[i,2] + epsilon):
            c += 1
        #altrimenti se il valore è più piccolo dell'estremo inferiore, non serve andare avanti, non c'è sicuramente
        elif value < (redTimestamps[i,0] - epsilon):
            return c,False
    #il valore non c'è
    return c,False

#filtraggio degli eventi, in base alla loro appartenenza o meno ai bounding boxes
def filterEventCoords(x,y,startLE,stopLE,startRE,stopRE):
    if x>=startLE[0] and x<=stopLE[0]:
        if y>=startLE[1] and y<=stopLE[1]:
            return "leftEye",True
    elif x>=startRE[0] and x<=stopRE[0]:
        if y>=startRE[1] and y<=stopRE[1]:
            return "rightEye",True
    return "none",False

'''eyeProcessing.py'''

#calcolo i massimi bounding box relativi agli occhi sx e dx
def getMaxMinBoundingBoxes(list):
    xMinLE, xMaxLE, yMinLE, yMaxLE = [],[],[],[]
    xMinRE, xMaxRE, yMinRE, yMaxRE = [],[],[],[]
    for elems in list:
        bbLE = elems["leftEye"]
        xMinLE.append(bbLE["xMin"])
        xMaxLE.append(bbLE["xMax"])
        yMinLE.append(bbLE["yMin"])
        yMaxLE.append(bbLE["yMax"])
        #check max left

        bbRE = elems["rightEye"]
        xMinRE.append(bbRE["xMin"])
        xMaxRE.append(bbRE["xMax"])
        yMinRE.append(bbRE["yMin"])
        yMaxRE.append(bbRE["yMax"])
        #check max right

    startLE,stopLE = (min(xMinLE),min(yMinLE)),(max(xMaxLE),max(yMaxLE))
    startRE,stopRE = (min(xMinRE),min(yMinRE)),(max(xMaxRE),max(yMaxRE))
    maxLE,maxRE = [startLE,stopLE],[startRE,stopRE]
    return maxLE,maxRE     
        
#controllo se l'evento appartiene al bounding box dell'occhio sx, dx o nessuno dei due
def recognizeEyesBoxes(x,y,maxLE,maxRE):
    startLE,stopLE = maxLE[0],maxLE[1]
    startRE,stopRE = maxRE[0],maxRE[1]
    if x>=startLE[0] and x<=stopLE[0]:
        if y>=startLE[1] and y<=stopLE[1]:
            return "leftEye"
    elif x>=startRE[0] and x<=stopRE[0]:
        if y>=startRE[1] and y<=stopRE[1]:
            return "rightEye"
    return "none"

#stampa delle polarità degli eventi 
def printSumEyes(mat1,mat2,time1,time2,init,delta):
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.suptitle('Accumulazione iniziale: '+str(init)+'s, finestra di accumulazione: '+str(delta)+'s')
    ax1.set_ylabel('Y - sumLeftEye')
    ax1.plot(np.array(time1),np.array(mat1),color = 'r')
    ax2.set_ylabel('Y - sumRightEye')
    ax2.set_xlabel('X - Time')
    ax2.plot(np.array(time2),np.array(mat2),color = 'r')
    plt.show()
