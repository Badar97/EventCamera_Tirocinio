"""
Il seguente programma è diviso in due parti:
la prima è una versione ridotta di faceMeshing.py, che mostra a video il volto circondato da due bounding
boxes tempo varianti, che delimiteranno l'area di appartenenza degli eventi successivamente catturati.
Le informazioni relative agli occhi vengono salvate nel file dvSaveEyesFrames, mentre quelle relative ai 
bounding boxes vengono salvate nel file eyesBoundingBoxes.json

La seconda parte analizza tutti gli eventi appartenenti agli intervalli delineati nel file analyzeEyes.json,
se tali eventi appartengono ai bounding boxes dei due occhi, allora vengono salvati nel file dvSaveEyesEvents.json
"""

import cv2
import utilities.landmarks as ld
import utilities.functions as ft
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from dv import AedatFile

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

image = 0
list,redDdict,count,baseDate = [], {}, 0, 0
reducedTimestamps = ft.getReducedTimestamps("analyzeEyes")

bboxes,eyesDict = [],{}

with AedatFile(ft.genPath+"dvSave.aedat4") as recording:
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        for frame in recording["frames"]:
            if count == 0:
                baseDate = frame.timestamp/1e6
                count = 1
            timestamp = (frame.timestamp/1e6)-baseDate
            image = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2RGB)
            annotated_image = image.copy()
            results = face_mesh.process(image)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    dict,landmark_list = ft.calculateKeypoints(face_landmarks)
                    redDict = ft.getEyesValues(face_landmarks,reducedTimestamps,timestamp)
                    
                    mp_drawing.draw_landmarks(image=annotated_image,landmark_list=landmark_pb2.NormalizedLandmarkList(landmark = landmark_list[0]),landmark_drawing_spec=mp_drawing.DrawingSpec(color=ld.RED_COLOR, thickness=1, circle_radius=1))
                    mp_drawing.draw_landmarks(image=annotated_image,landmark_list=landmark_pb2.NormalizedLandmarkList(landmark = landmark_list[2]),landmark_drawing_spec=mp_drawing.DrawingSpec(color=ld.YELLOW_COLOR, thickness=1, circle_radius=1))
                    
                    startLE,stopLE = ft.getMaxMinXYValues(dict,"leftEye",image)
                    startRE,stopRE = ft.getMaxMinXYValues(dict,"rightEye",image)

                    eyesDict = {}
                    
                    cv2.putText(annotated_image, 't: '+str(timestamp), (0,20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                    cv2.rectangle(annotated_image,startLE,stopLE,ld.RED_COLOR,1)
                    cv2.rectangle(annotated_image,startRE,stopRE,ld.YELLOW_COLOR,1)
                    cv2.imshow("eyeMeshing", annotated_image)
                    
                    if(bool(redDict)):
                        eyesDict["timestamp"] = timestamp
                        eyesDict["leftEye"] = {'xMin':startLE[0],'yMin':startLE[1],'xMax':stopLE[0],'yMax':stopLE[1]}
                        eyesDict["rightEye"] = {'xMin':startRE[0],'yMin':startRE[1],'xMax':stopRE[0],'yMax':stopRE[1]}
                        bboxes.append(eyesDict)
                        redDict["timestamp"] = timestamp
                        list.append(redDict)
                cv2.waitKey(1)

ft.writeJson(list,"dvSaveEyesFrames")
ft.writeJson(bboxes,"eyesBoundingBoxes")

list, dict, count, idx = [], {}, 0, 0
bool2 = False

with AedatFile(ft.genPath+"dvSave.aedat4") as f:
    events = np.hstack([packet for packet in f['events'].numpy()])
    for event in events:
        timestamp = (event["timestamp"]/1e6)-baseDate
        idx, bool1 = ft.filterEventTimestamps(timestamp,reducedTimestamps,idx)
        if bool1:
            string,bool2 = ft.filterEventCoords(event["x"],event["y"],startLE,stopLE,startRE,stopRE)
        if bool1 and bool2:
            dict["timestamp"], dict["x"], dict["y"], dict["polarity"],dict["type"]  = timestamp,event["x"],event["y"],event["polarity"],string
            list.append(dict)
            dict = {}

ft.writeJson(list,"dvSaveEyesEvents",cls=ft.NumpyEncoder)