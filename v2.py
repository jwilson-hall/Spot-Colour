import cv2
import os
import numpy as np
# from gevent import config
thres = 0.45 # Threshold to detect object
image_path = os.path.dirname

classNames= []

classFile ='coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# configPath = 'frozen_inference_graph.pb'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)



def run_algorithm(img, numb):
    classIds, confs, bbox = net.detect(img,confThreshold = 0.5)
    # print("BOX: ",bbox[0])
    # x1 = bbox[0][0]
    # y1 = bbox[0][1]
    # x2 = bbox[0][2]
    # y2 = bbox[0][3]
    boxes = []
    confidenceList = []
    for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        boxes.append([x1,y1,x2,y2])
        confidenceList.append(confs)
        # print(confs)
        # print(classNames[classId])
        # cv2.rectangle(img,[x1,y1,x2,y2],color=(0,0,255),thickness=3)
        # print(box)

    # saliency = cv2.saliency.Objectness_create()
    # (success, saliencyMap) = saliency.computeSaliency(img)
    # saliencyMap = (saliencyMap * 255).astype("uint8")

    # Initialise the more fine-grained saliency detector and compute the saliencyMap
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    newSaliencyMap=saliencyMap*255

    threshMap = cv2.threshold(newSaliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # threshMap =  cv2.threshold(newSaliencyMap.astype("uint8"),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # blur = cv2.GaussianBlur(newSaliencyMap,(5,5),0)
    # threshMap = cv2.threshold(blur.astype("uint8"),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    output_img = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    testGrey = img.copy()
    for i in range(len(img)):
        for j in range(len(img[0])):
            testGrey[i][j] = np.array([output_img[i][j],output_img[i][j],output_img[i][j]])
    # print(type(output_img))
    # print(type(img))
    # print(type(output_img[50][50]))
    # print(type(img[50][50]))
    # print(type(np.array([output_img[0][0],output_img[0][0],output_img[0][0]])))
    # for i in range(len(threshMap)):
    #     for j in range(len(threshMap[0])):
    #         if (threshMap[i][j]==255 and (i > y1 and j > x1) and (i < y2 and j < x2)):
    #             img[i][j] = np.array([output_img[i][j],output_img[i][j],output_img[i][j]])
    # print(bbox)
    maxX = 0
    maxY = 0
    minX = len(img)*len(img[0])
    minY = len(img)*len(img[0])
    newThrMap = threshMap.copy()
    newThrMap.fill(0)
    # boxes.remove(boxes[0])
    for b in boxes:
        maxX = max(maxX,b[0],b[2])
        maxY = max(maxY,b[1],b[3])
        minX = min(minX,b[2],b[0])
        minY = min(minY,b[3],b[1])
    print(minX,":",minY,"  ",maxX,":",maxY)
    for b in boxes:
        x1 = b[0]
        y1 = b[1]
        x2 = b[2]
        y2 = b[3]
        # start_point = (x1, y1)
        # end_point = (x2+x1, y2+y1)
        # cv2.rectangle(testGrey,[x1,y1,x2,y2],color=(150,150,0),thickness=2)
        # cv2.rectangle(testGrey,start_point,end_point,color=(0,0,255),thickness=2)
        # print(x1,y1,x2,y2)
        # maxJ = 0
        # maxI = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (threshMap[i][j] == 255 and (j >= x1 and i >= y1) and (j <= x2+x1 and i <= y2+y1)):
                    testGrey[i][j] = img[i][j]
                    newThrMap[i][j] = 255                

    # cv2.resize(testGrey, (810, 1080))  
    # cv2.imshow("Output",testGrey)
    # cv2.imshow("Threshold map",newThrMap)
    # data_path = os.path.dirname(os.getcwd())+"\\data\\"
    # Writing Images
    cv2.imwrite(data_path+"train_output\\v2\\"+str(numb)+"_p.jpg",testGrey)
    cv2.imwrite(data_path+"train_output\\v2\\"+str(numb)+"_thr.jpg",newThrMap)
    # cv2.imshow("Output image",output_img)
    # cv2.imshow("Output", saliencyMap)
    # cv2.waitKey(0)
data_path = os.getcwd()+"\\"
# img = cv2.imread(data_path+"test_data\\"+str(4)+'.jpg')
# run_algorithm(img, 1)
for i in range(1, 11):
    img = cv2.imread(data_path+"train_data\\"+str(i)+'.jpg')
    run_algorithm(img, i)