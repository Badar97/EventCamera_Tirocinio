"""
Il seguente programma si occupa del processamento delle informazioni contenute nel file dvSaveFrames.json,
limitando l'analisi agl intervalli contenuti nei file analyzeEye.json e analyzeLips.json
Le informazioni vengono salvate nel file dvSaveReducedDerivatives.json
"""

import utilities.functions as ft

redTimestampsEye = ft.getReducedTimestamps('analyzeEyes')
redTimestampsLips = ft.getReducedTimestamps('analyzeLips')
                
with open(ft.jsonPath+"dvSaveFrames.json", 'r') as file:
    data = ft.load(file)
    timestamps = ft.getTimestamps(data)
    coupleCoords = ft.getCoupleCoords()

    matLE = ft.getXYFilteredValues(data,coupleCoords[0],redTimestampsEye,'leftEye')
    matLI = ft.getXYFilteredValues(data,coupleCoords[1],redTimestampsLips,'lips')
    matRE = ft.getXYFilteredValues(data,coupleCoords[2],redTimestampsEye,'rightEye')

    #calcolo delle velocità e direzioni
    der = ft.calculateRedDerivatives(matLE,matLI,matRE,data,timestamps,coupleCoords)

    #scrivo le informazioni nel file
    ft.writeJson(der,"dvSaveReducedDerivatives")


#grafico dei valori delle velocità e della direzione per ogni coppia per ogni landmark
with open(ft.jsonPath+"dvSaveReducedDerivatives.json", 'r') as file:
    data = ft.load(file)
    matLE = ft.getNanValues(data,coupleCoords[0],'leftEye')
    matLI = ft.getNanValues(data,coupleCoords[1],'lips')
    matRE = ft.getNanValues(data,coupleCoords[2],'rightEye')
    
    ft.printGraph(matLE,coupleCoords[0],timestamps[1:-1],'Left Eye')
    ft.printGraph(matRE,coupleCoords[2],timestamps[1:-1],'Right Eye')
    ft.printGraph(matLI,coupleCoords[1],timestamps[1:-1],'Lips')
    