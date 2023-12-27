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

bboxes,totDict = [],{}

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
                    redDict = ft.getLipsValues(face_landmarks)
                    
                    mp_drawing.draw_landmarks(image=annotated_image,landmark_list=landmark_pb2.NormalizedLandmarkList(landmark = landmark_list[1]),landmark_drawing_spec=mp_drawing.DrawingSpec(color=ld.RED_COLOR, thickness=1, circle_radius=1))
                    
                    startLI,stopLI = ft.getLipsMaxMinXYValues(dict,"lips",image)

                    totDict = {}
                    
                    cv2.putText(annotated_image, 't: '+str(timestamp), (0,20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                    cv2.rectangle(annotated_image,startLI,stopLI,ld.RED_COLOR,1)
                    cv2.imshow("totMeshing", annotated_image)
                    
                    if(bool(redDict)):
                        totDict["timestamp"] = timestamp
                        totDict["lips"] = {'xMin':startLI[0],'yMin':startLI[1],'xMax':stopLI[0],'yMax':stopLI[1]}
                        bboxes.append(totDict)
                        redDict["timestamp"] = timestamp
                        list.append(redDict)
                cv2.waitKey(1)

ft.writeJson(list,"dvSaveLipsFrames")
ft.writeJson(bboxes,"lipsBoundingBoxes")

list, dict, count, idx = [], {}, 0, 0
bool2 = False

with AedatFile(ft.genPath+"dvSave.aedat4") as f:
    events = np.hstack([packet for packet in f['events'].numpy()])
    for event in events:
        timestamp = (event["timestamp"]/1e6)-baseDate
        string,bool2 = ft.filterLipsEventCoords(event["x"],event["y"],startLI,stopLI)
        if bool2:
            dict["timestamp"], dict["x"], dict["y"], dict["polarity"],dict["type"]  = timestamp,event["x"],event["y"],event["polarity"],string
            list.append(dict)
            dict = {}

ft.writeJson(list,"dvSaveLipsEvents",cls=ft.NumpyEncoder)
