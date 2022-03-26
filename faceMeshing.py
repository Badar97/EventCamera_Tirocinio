"""
Il seguente programma si occupa dell'apertura del file con estensione '.aedat4',
vengono iterati i frame e contemporaneamente viene applicato l'algoritmo faceMesh per la rilevazione
dei keypoints appartenenti al volto.
I dati relativi ai keypoints identificati dall'algoritmo vengono salvati in un file 'dvSaveFrames.json'
"""

import cv2
import utilities.landmarks as ld
import utilities.functions as ft
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from dv import AedatFile

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

list, dict, count = [], {}, 0
with AedatFile(ft.genPath+"dvSave.aedat4") as f:
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        for frame in f["frames"]:
            if count == 0:
                baseDate = frame.timestamp/1e6
                count = 1
            timestamp = (frame.timestamp/1e6)-baseDate

            image = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2RGB)
            annotated_image = image.copy()
            results = face_mesh.process(image)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    dict, landmark_list = ft.calculateKeypoints(face_landmarks)

                    mp_drawing.draw_landmarks(image=annotated_image,landmark_list=landmark_pb2.NormalizedLandmarkList(landmark = landmark_list[0]),landmark_drawing_spec=mp_drawing.DrawingSpec(color=ld.RED_COLOR, thickness=1, circle_radius=1))
                    mp_drawing.draw_landmarks(image=annotated_image,landmark_list=landmark_pb2.NormalizedLandmarkList(landmark = landmark_list[1]),landmark_drawing_spec=mp_drawing.DrawingSpec(color=ld.YELLOW_COLOR, thickness=1, circle_radius=1))
                    mp_drawing.draw_landmarks(image=annotated_image,landmark_list=landmark_pb2.NormalizedLandmarkList(landmark = landmark_list[2]),landmark_drawing_spec=mp_drawing.DrawingSpec(color=ld.BLUE_COLOR, thickness=1, circle_radius=1))
                    mp_drawing.draw_landmarks(image=annotated_image,landmark_list=landmark_pb2.NormalizedLandmarkList(landmark = landmark_list[3]), landmark_drawing_spec = mp_drawing.DrawingSpec(color=ld.GREEN_COLOR, thickness=1, circle_radius=1))
                    
                    cv2.putText(annotated_image, 't: '+str(timestamp), (0,20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                    cv2.imshow("faceMeshing", annotated_image)
                
                dict["timestamp"] = timestamp
                list.append(dict)
                
                cv2.waitKey(1)
                """
                #scansione manuale dei timestamp
                if cv2.waitKey() == ord('q'):
                    break
                """
                
ft.writeJson(list,"dvSaveFrames")