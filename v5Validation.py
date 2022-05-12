#C1949699
#
#
import math
import random
import re
import time
from turtle import numinput
from cv2 import INTER_AREA, INTER_BITS, INTER_CUBIC, INTER_LANCZOS4, INTER_LINEAR, INTER_LINEAR_EXACT, INTER_MAX, imread, imshow, waitKey
import numpy as np
import cv2
import os
from multiprocessing import Process, Manager

data_path = os.getcwd()
dataset_path = data_path+"\\cross_validation\\"
imageDataLocation = dataset_path+"\\images\\"
truthDataLocation = dataset_path+"\\truth\\"
LOG = []
datasetTrain = []
datasetVal = []
runTimeNumber = int(1)

LongRangeWorker = np.arange(3,31,6)
confidenceList = np.divide(np.subtract(np.arange(30,85,15),5),100)

#REGION functions

# Threshold to detect object
def calculateSTD(evaluationList, evaluationMean):
    n = 1
    if len(evaluationList)!=1:
        n = len(evaluationList)-1
    sumX = 0
    for score in evaluationList:
        sumX+=(score-evaluationMean)**2
    standardDeviation = sumX / n
    standardDeviation = math.sqrt(standardDeviation)
    return standardDeviation
    
def runTimeCount():
    i = 1
    fileList = os.listdir(data_path)
    while "log"+str(i)+".txt" in fileList:
            i+=1    
    return int(i)

def saveLog():    
    i = 1
    fileList = os.listdir(data_path)
    if "log1.txt" not in fileList:
        with open("log"+str(i)+".txt","a") as f:
            for line in LOG:
                f.write(line+"\n")
    else:
        while "log"+str(i)+".txt" in fileList:
            i+=1
        with open("log"+str(i)+".txt","a") as f:
            for line in LOG:
                f.write(line+"\n")
    LOG.clear()

def addToLog(line,varname):
    # print("Line",line)
    if varname == "BinaryMasks":
        LOG.append(varname,line)
    elif isinstance(line, list):
        # log.append(varname)
        LOG.append(str(varname+" "+f'{line}'.split('=')[0]))
    elif isinstance(line, str or int):
        # log.append(varname)
        LOG.append(str(varname)+" "+str(line))
    elif isinstance(line, float):
        LOG.append(str(varname)+" "+str(line))

def calc_IoU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    # print(mask1_area, " : ", mask2_area)
    intersection = np.count_nonzero(np.logical_and( mask1,  mask2))
    # print("intersection",intersection)
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou

# def runTest():
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):   
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def select_new_dataset():
    datasetTrain.clear()
    datasetVal.clear()
    files = os.listdir(imageDataLocation)
    for i in range(14):
        file = random.choice(files) 
        while file in datasetTrain:
            file = random.choice(files)
        datasetTrain.append(file)
    for file in files:
        if file not in datasetTrain:
            datasetVal.append(file)
#ENDREGION
# configPath = 'frozen_inference_graph.pb'
net5 = cv2.dnn.readNetFromTensorflow("dnn\\frozen_inference_graph_coco.pb","dnn\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net6 = cv2.dnn_DetectionModel(weightsPath,configPath)
net6.setInputSize(320,320)
net6.setInputScale(1.0/127.5)
net6.setInputMean((127.5,127.5,127.5))
net6.setInputSwapRB(True)


# test_data_path = os.path.dirname(os.getcwd())+"\\data\\new_test_data\\"
def selectParameters5():#selecting the parameters that will be swapped for each iteration
    PARAMETERS = list()
    PARAMETERS.append(random.choice(confidenceList))
    PARAMETERS.append(random.choice(LongRangeWorker))
    return PARAMETERS

def workFunc(img, workerRange,x,y):
    count = 0
    down = 0
    up = 0
    left = 0
    right = 0
    tempFalse = False
    for i in range(x+1,min(x+workerRange,len(img))):
        if img[i][y] != 0:
            if tempFalse!= True:
                count+=1 
                down+=1
                tempFalse=True
                break    
    tempFalse = False
    for i in range(x-1,max(x-workerRange,-1),-1):
        if img[i][y] != 0:
            if tempFalse != True:
                count+=1
                up+=1
                tempFalse=True
                break
    tempFalse = False
    for i in range(y+1,min(y+workerRange,len(img[0])-1)):
        if img[x][i] != 0:
            if tempFalse!= True:
                count+=1
                right+=1
                tempFalse=True
                break
    tempFalse = False
    for i in range(y-1,max(y-workerRange,-1),-1):
        if img[x][i] != 0:
            if tempFalse != True:
                count+=1
                left+=1
                tempFalse=True
                break
    return count
    # return up,down,left,right
def fillSmallSpaceV5(img, workerRange,x,y):
    count = 0
    down = 0
    up = 0
    left = 0
    right = 0
    tempFalse = False
    for i in range(x+1,min(x+workerRange,len(img))):
        if img[i][y] != 0:
            if tempFalse!= True:
                count+=1 
                down+=1
                tempFalse=True
                break    
    tempFalse = False
    for i in range(x-1,max(x-workerRange,-1),-1):
        if img[i][y] != 0:
            if tempFalse != True:
                count+=1
                up+=1
                tempFalse=True
                break
    if count == 2:
        return count
    else:
        count = 0
    tempFalse = False
    for i in range(y+1,min(y+workerRange,len(img[0])-1)):
        if img[x][i] != 0:
            if tempFalse!= True:
                count+=1
                right+=1
                tempFalse=True
                break
    tempFalse = False
    for i in range(y-1,max(y-workerRange,-1),-1):
        if img[x][i] != 0:
            if tempFalse != True:
                count+=1
                left+=1
                tempFalse=True
                break
    return count

def smoothing(mask,lineImage, workerRange,x,y):
    maskCount = 0
    lineCount = 0
    tempFalse = False
    for i in range(x+1,min(x+workerRange,len(mask))):
        if mask[i][y] != 0:
            if tempFalse!= True:
                maskCount+=1
                tempFalse=True
                break
        elif lineImage[i][y] != 0:
            if tempFalse != True:
                lineCount+=1
                tempFalse=True
                break
    tempFalse = False
    for i in range(x-1,max(x-workerRange,-1),-1):
        if mask[i][y] != 0:
            if tempFalse!= True:
                maskCount+=1
                tempFalse=True
                break
        elif lineImage[i][y] != 0:
            if tempFalse != True:
                lineCount+=1
                tempFalse=True
                break    
    
    tempFalse = False
    for i in range(y+1,min(y+workerRange,len(mask[0])-1)):
        if mask[x][i] != 0:
            if tempFalse!= True:
                maskCount+=1
                tempFalse=True  
                break
        elif lineImage[x][i] != 0:
            if tempFalse != True:
                lineCount+=1
                tempFalse=True
                break

    tempFalse = False
    for i in range(y-1,max(y-workerRange,-1),-1):
        if mask[x][i] != 0:
            if tempFalse!= True:
                maskCount+=1
                tempFalse=True
                break      
        elif lineImage[x][i] != 0:
            if tempFalse != True:
                lineCount+=1
                tempFalse=True
                break

    return lineCount,maskCount

    # return up,down,left,right
def findOptimalLinesV5(ogSlice,numb):
    temp = ogSlice.copy()
    temp = cv2.GaussianBlur(ogSlice, (13,13), cv2.BORDER_CONSTANT)
    # ogSlice = cv2.Canny(ogSlice,125,150)
    temp = cv2.Canny(temp,100,175)
    size = np.size(temp)
    whiteCount = np.count_nonzero(temp)
    compactness = (whiteCount/size)*100
    print("Compactness ",compactness,"%1",numb)
    # while (compactness < 6 or compactness > 9):
    if compactness < 3:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (5,5), cv2.BORDER_CONSTANT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        print("Compactness ",compactness,"%2",numb)
    if compactness < 3.5:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (5,5), cv2.BORDER_CONSTANT)
        # threshold        
        temp = cv2.Canny(temp,100,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        print("Compactness ",compactness,"%5",numb)
    if compactness > 8:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (7,7), cv2.BORDER_REFLECT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        print("Compactness ",compactness,"%3",numb)
    if compactness > 9:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (9,9), cv2.BORDER_CONSTANT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        print("Compactness ",compactness,"%4",numb)
    if compactness < 6:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (5,5), cv2.BORDER_CONSTANT)
        # threshold        
        temp = cv2.Canny(temp,150,175)
        size = np.size(temp)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        print("Compactness ",compactness,"%5",numb)
    print("Compactness ",compactness,"%",numb)
    return temp

def v5(img,numb,PARAMETERS,dictOfBinaryMask): 
    H, W, _ = img.shape
    # Create black image with same dimensions as input image
    black_image = np.zeros((H, W), np.uint8)
    # Detect objects inside input image
    blob = cv2.dnn.blobFromImage(img, swapRB=True)
    net5.setInput(blob)
    boxes, masks = net5.forward(["detection_out_final", "detection_masks"])
    classIds, _, boxes2 = net6.detect(img,confThreshold = PARAMETERS[0])

    if not isinstance(classIds,tuple):
        detection_count = boxes.shape[2]
        boundingBox = []
        for i in range(detection_count):
            box = boxes[0, 0, i]    
            classID = int(box[1])
            confidence = box[2]
            if confidence < PARAMETERS[0]:
                continue
            
            #Get box coordinates
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            (x1, y1, x2, y2) = box.astype("int")
            # boundingBox.append([x1,y1,x2,y2])
            roi_Width = x2 - x1
            roi_Height = y2 - y1
            
            mask = masks[i, classID]
            mask = cv2.resize(mask, (roi_Width, roi_Height),interpolation=cv2.INTER_CUBIC)
            _, mask = cv2.threshold(mask, 0.6, 255, cv2.THRESH_BINARY)
            
            for i in range(len(mask)):
                for j in range(len(mask[0])):
                    if mask[i][j] != 0:
                        black_image[i+y1][j+x1] = mask[i][j]

        for box in boxes2:
            boundingBox.append([box[0],box[1],box[2]+box[0],box[3]+box[1]])
        maxX = 0
        maxY = 0
        minX = len(img)*len(img[0])
        minY = len(img)*len(img[0])
        for box in boundingBox:
            maxX = max(maxX,box[2])
            maxY = max(maxY,box[3])
            minX = min(minX,box[0])
            minY = min(minY,box[1])

        ogSlice = img[minY:maxY, minX:maxX]
        maskSlice = black_image[minY:maxY, minX:maxX]
        ogSlice = findOptimalLinesV5(ogSlice,numb)
        ogSlice = cv2.dilate(ogSlice,(5,5),iterations=3)
        # cv2.imshow("OG Slice ", ogSlice)
        newSlice = ogSlice.copy()
        newSlice.fill(0)

        for b in boundingBox:
            for i in range(b[0]-minX,(b[2])-minX):
                for j in range(b[1]-minY,(b[3])-minY):
                    newSlice[j][i] = ogSlice[j][i]
        
        ogSlice = newSlice
        # cv2.imshow("OG Slice ", ogSlice)
        cpSlice = maskSlice.copy()#
        
        workerRange = int(max(len(ogSlice)/PARAMETERS[1],len(ogSlice[0])/PARAMETERS[1]))
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                if maskSlice[i][j] != 255:
                    workVal = smoothing(maskSlice,ogSlice,workerRange,i,j)
                    if (workVal[0] >= 2 and workVal[1] >= 2) or (workVal[0] > 3) :
                        cpSlice[i][j] = 255
                    # elif workVal[0] > 3:
                    #     cpSlice[i][j] = 255
        # ogSlice = cpSlice.copy()
        cpSlice2 = cpSlice.copy()
        cpSlice = cv2.bitwise_or(maskSlice,ogSlice)
        cpSlice = cv2.bitwise_or(cpSlice2,cpSlice)
        # cpSlice = cpSlice2
        for i in range(len(cpSlice)):
            for j in range(len(cpSlice[0])):
                if maskSlice[i][j] != 255:
                    workVal = fillSmallSpaceV5(ogSlice,2,i,j)
                    if workVal == 2:
                        cpSlice[i][j] = 255

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cpSlice)
        # imshow_components(output)

        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        biggestObject = max(sizes)/2
        
        img2 = output.copy()
        img2.fill(0)
        for i in range(0, nb_components):
            if sizes[i] >= biggestObject:
                img2[output == i + 1] = 255
        
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                cpSlice[i][j] = img2[i][j]
        maskSlice = cpSlice.copy()
        # imshow("Cp Slice",maskSlice)
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                if maskSlice[i][j] != 255:
                    workVal = smoothing(maskSlice,cpSlice,workerRange,i,j)
                    if workVal[0] > 3 or workVal[1] > 3 :
                        maskSlice[i][j] = 255
        cpSlice = maskSlice.copy()

        rmSlice = cpSlice.copy()
        drcontours = rmSlice.copy()
        drcontours = cv2.cvtColor(drcontours, cv2.COLOR_GRAY2RGB)
        removeIslands = cv2.pyrDown(rmSlice)
        _, threshed = cv2.threshold(rmSlice, 0, 255, cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
        #find maximum contour and draw   
            cmax = max(contours, key = cv2.contourArea) 
            epsilon = 0.002 * cv2.arcLength(cmax, True)
            approx = cv2.approxPolyDP(cmax, epsilon, True)
            cv2.drawContours(drcontours, [approx], -1, (0, 255, 0), 2)
            width, height = rmSlice.shape
            # imshow("Contour", drcontours)
            # waitKey(0)
            #fill maximum contour and draw   
            removeIslands = np.zeros( [width, height, 3],dtype=np.uint8 )
            cv2.fillPoly(removeIslands, pts =[cmax], color=(255,255,255))
            cpSlice = cv2.cvtColor(removeIslands, cv2.COLOR_BGR2GRAY)

        # waitKey(0)
        black_image_lines = black_image.copy()
        # black_image_lines.fill(0)
        for i in range(len(cpSlice)):
            for j in range(len(cpSlice[0])):
                if cpSlice[i][j] != 0:
                    black_image_lines[i+minY][j+minX] = cpSlice[i][j]
        dictOfBinaryMask[numb] = black_image_lines
    else:
        fMask = img.copy()
        fMask = cv2.cvtColor(fMask, cv2.COLOR_BGR2GRAY)
        fMask.fill(0)
        dictOfBinaryMask[numb] = fMask

def findOptimalParams(listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain):
    for i in range(1):
        PARAMETERS = selectParameters5()
        # print("PARAMS ",PARAMETERS)
        if PARAMETERS in evaluationParameters:
            break
        evaluationParameters.append(PARAMETERS)#[PARAMETERS[0],PARAMETERS[1],PARAMETERS[1]]
        # print(evaluationParameters)
        # addToLog(PARAMETERS.copy(),"Parameters")
        jobs = []
        for image in datasetTrain:
            current = imread(imageDataLocation+image)
            p1 = Process(target=v5,args=[current,image,PARAMETERS,dictOfBinaryMask])
            p1.start()
            jobs.append(p1)                    
        for job in jobs:
            job.join()
        averageIOU=0
        # print("Running Test on evalutaions")
        for image in datasetTrain:
            imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
            max_IoU = 0
            for i in range (1,6):
                mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                mask2 = imread(truthDataLocation+mask2)
                mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
            averageIOU+=max_IoU                    
        averageIOU /= len(datasetTrain)
        # print(averageIOU)
        listOfEvaluations.append(averageIOU)

def runTestV5():
    listTestEvaluations = []
    numRange = 20
    startTime = time.time()
    for i in range(numRange):
        print("Current run time: ",time.time()-startTime)
        print(i,"iteration")
        addToLog(i,"Loop: ")
        select_new_dataset()
        addToLog(datasetTrain,f'{datasetTrain=}'.split('=')[0])
        addToLog(datasetVal,f'{datasetVal=}'.split('=')[0])
        # listOfBinaryMask = []
        # for i in range(10):
        listOfEvaluations = []
        evaluationParameters = []
        averageIOU = 0
        optimalParams = []
        with Manager() as manager:
            dictOfBinaryMask = manager.dict()
            evaluationParameters = manager.list()
            listOfEvaluations = manager.list()
            PARAMETERS = manager.list()
            jobs = []
            for _ in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            print("Job Completed 1: ",i)
            for _ in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            print("Job Completed 2: ",i)
            for _ in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            print("Job Completed 3: ",i)
            print("Length of listOfEvaluations",len(listOfEvaluations))
            print("Length of evaluationParameters",len(evaluationParameters))
            print("Length of listTestEvaluations",len(listTestEvaluations))
            optimalParamsResults = listOfEvaluations[listOfEvaluations.index(max(listOfEvaluations))]
            optimalParams = evaluationParameters[listOfEvaluations.index(max(listOfEvaluations))]
            addToLog(optimalParamsResults,"Evaluation Score Train Set")
            print("Second part")
            
            for image in datasetVal:
                current = imread(imageDataLocation+image)
                p1 = Process(target=v5,args=[current,image,optimalParams,dictOfBinaryMask])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            averageIOU=0
            for image in datasetVal:
                imageNumber = int(re.compile(r'\d+(?:\.\d+)?').findall(image)[0])
                max_IoU = 0
                for i in range (1,6):
                    mask2 = str(imageNumber)+"_gt"+str(i)+".jpg"
                    mask2 = imread(truthDataLocation+mask2)
                    mask2 = cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY)                
                    max_IoU = max(max_IoU,calc_IoU(dictOfBinaryMask[image],mask2))
                averageIOU+=max_IoU
            averageIOU /= len(datasetVal)
            addToLog(optimalParams,"Optimal Parameters")
            addToLog(averageIOU,"Evaluation Score Validation Set")
            listTestEvaluations.append(averageIOU)
            print("Second part Done")
    averageTestScore = sum(listTestEvaluations)/len(listTestEvaluations)
    addToLog(averageTestScore,"Average Validation Score")
    addToLog(calculateSTD(listTestEvaluations,averageTestScore), "Standard Deviation")
    addToLog("v5","Pipeline: ")

if __name__ == "__main__":
    start_time = time.time()
    runTestV5()
    print("--- %s seconds ---" % (time.time() - start_time))
    runTimeNumber = str(runTimeCount()) 
    saveLog()
    print("Run time number ",runTimeNumber)