from cv2 import cv2
import mediapipe as mp
import numpy as np
import time
import pandas
from training.train import Net

layers = [[42], [1200, "relu"], [600, "relu"], [5, "softmax"]] # relu relu softmax
net = Net(layers,load_mode=True)
data =pandas.read_csv("C:\\Users\\ASUS\\PycharmProjects\\tensorflowEnvironment\\Deneme1\\CollectingPose\\poz_train.csv")
input_data = data.iloc[:,3:45].to_numpy()
output_data = data.iloc[:,45:50].to_numpy()

data = pandas.read_csv("C:\\Users\\ASUS\\PycharmProjects\\tensorflowEnvironment\\Deneme1\\CollectingPose\\poz_val.csv")
validation_data_input = data.iloc[:,3:45].to_numpy()
validation_data_output = data.iloc[:,45:50].to_numpy()



cap= cv2.VideoCapture(0)

cap.set(3,224)
cap.set(4,224)

mpHand = mp.solutions.hands
hands=mpHand.Hands()

mpDraw = mp.solutions.drawing_utils

pTime = 0
fileId = 0
handPoseList = []


while cap.isOpened():
    tahminEdilenPose = "None"
    success, img = cap.read()

    img=cv2.flip(img,1)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    data = pandas.DataFrame(columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
                                     "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38", "39", "40", "41"])

    resultsHand = hands.process(imgRGB)
    handPoseList = []

    if resultsHand.multi_hand_landmarks:
        lms = resultsHand.multi_hand_landmarks
        listLandmark = []
        cx= []
        cy = []
        lastId = 0

        for handLms in resultsHand.multi_hand_landmarks:

            mpDraw.draw_landmarks(img,handLms,mpHand.HAND_CONNECTIONS)




            print(fileId,handLms.landmark[0].x,handLms.landmark[0].y,handLms.landmark[1].x,handLms.landmark[1].y)


        handPoseList.append([fileId, listLandmark])


        data = data.append({"0": handLms.landmark[0].x, "1": handLms.landmark[0].y, "2": handLms.landmark[1].x,
                            "3": handLms.landmark[1].y,
                            "4": handLms.landmark[2].x, "5": handLms.landmark[2].y, "6": handLms.landmark[3].x,
                            "7": handLms.landmark[3].y,
                            "8": handLms.landmark[4].x, "9": handLms.landmark[4].y, "10": handLms.landmark[5].x,
                            "11": handLms.landmark[5].y,
                            "12": handLms.landmark[6].x, "13": handLms.landmark[6].y, "14": handLms.landmark[7].x,
                            "15": handLms.landmark[7].y,
                            "16": handLms.landmark[8].x, "17": handLms.landmark[8].y, "18": handLms.landmark[9].x,
                            "19": handLms.landmark[9].y,
                            "20": handLms.landmark[10].x, "21": handLms.landmark[10].y, "22": handLms.landmark[11].x,
                            "23": handLms.landmark[11].y,
                            "24": handLms.landmark[12].x, "25": handLms.landmark[12].y, "26": handLms.landmark[13].x,
                            "27": handLms.landmark[13].y,
                            "28": handLms.landmark[14].x, "29": handLms.landmark[14].y, "30": handLms.landmark[15].x,
                            "31": handLms.landmark[15].y,
                            "32": handLms.landmark[16].x, "33": handLms.landmark[16].y, "34": handLms.landmark[17].x,
                            "35": handLms.landmark[17].y,
                            "36": handLms.landmark[18].x, "37": handLms.landmark[18].y, "38": handLms.landmark[19].x,
                            "39": handLms.landmark[19].y,
                            "40": handLms.landmark[20].x, "41": handLms.landmark[20].y
                            }, ignore_index=True)


        veri = np.array(data.to_numpy())
        tahmin = net.predict(veri)
        p = 0
        if(tahmin.max() == tahmin[0]):
            tahminEdilenPose = "1"
        elif(tahmin.max() == tahmin[1]):
            tahminEdilenPose = "2"
        elif (tahmin.max() == tahmin[2]):
            tahminEdilenPose = "3"
        elif (tahmin.max() == tahmin[3]):
            tahminEdilenPose = "4"
        elif (tahmin.max() == tahmin[4]):
            tahminEdilenPose = "5"


    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f"FPS: {int(fps)}",(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    cv2.putText(img, f"TAHMIN: {tahminEdilenPose}", (50, 100), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("IMG",img)
    if cv2.waitKey(1) & 0XFF == ord("q"):
        break



print(data.columns)
