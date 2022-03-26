"""
Il seguente programma si occupa del processamento delle informazioni contenute nel file 'dvSaveFrames.json',
calcolando direzione e velocità dei keypoints relativi ai diversi landmarks,
mostrando i loro andamenti mediante dei grafici.
Successivamente queste informazioni vengono salvate in un altro file 'dvSaveDerivatives.json'.
"""
import utilities.functions as ft

with open(ft.jsonPath+"dvSaveFrames.json", 'r') as file:
    data = ft.load(file)
    timestamps = ft.getTimestamps(data)
    coupleCoords = ft.getCoupleCoords()

    matLE = ft.getXYValues(data,coupleCoords[0],'leftEye')
    matLI = ft.getXYValues(data,coupleCoords[1],'lips')
    matRE = ft.getXYValues(data,coupleCoords[2],'rightEye')

    #calcolo delle velocità e direzioni
    der = ft.calculateDerivatives(matLE,matLI,matRE,data,timestamps,coupleCoords)

    #scrivo le informazioni nel file
    ft.writeJson(der,"dvSaveDerivatives")

#grafico dei valori delle velocità e della direzione per ogni coppia per ogni landmark
with open(ft.jsonPath+"dvSaveDerivatives.json", 'r') as file:
    data = ft.load(file)
    matLE = ft.getDerValues(data,coupleCoords[0],'leftEye')
    matLI = ft.getDerValues(data,coupleCoords[1],'lips')
    matRE = ft.getDerValues(data,coupleCoords[2],'rightEye')
    ft.printGraph(matLE,coupleCoords[0],timestamps[1:-1],'Left Eye')
    ft.printGraph(matLI,coupleCoords[1],timestamps[1:-1],'Lips')
    ft.printGraph(matRE,coupleCoords[2],timestamps[1:-1],'Right Eye')