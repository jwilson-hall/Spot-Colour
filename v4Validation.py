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

smallRangeWorker = np.arange(55,101,5)
LongRangeWorker = np.arange(3,31,3)
confidenceList = np.divide(np.subtract(np.arange(15,85,15),5),100)

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
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net4 = cv2.dnn_DetectionModel(weightsPath,configPath)
net4.setInputSize(320,320)
net4.setInputScale(1.0/127.5)
net4.setInputMean((127.5,127.5,127.5))
net4.setInputSwapRB(True)


# test_data_path = os.path.dirname(os.getcwd())+"\\data\\new_test_data\\"
def selectParameters4():#selecting the parameters that will be swapped for each iteration
    PARAMETERS = list()
    PARAMETERS.append(random.choice(confidenceList))
    PARAMETERS.append(random.choice(smallRangeWorker))
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
def fillSmallSpace(img, workerRange,x,y):
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
    # return up,down,left,right

def find_optimal_lines(ogSlice,numb):
    temp = ogSlice.copy()
    temp = cv2.GaussianBlur(ogSlice, (13,13), cv2.BORDER_CONSTANT)
    # ogSlice = cv2.Canny(ogSlice,125,150)
    temp = cv2.Canny(temp,100,175)
    size = np.size(ogSlice)
    whiteCount = np.count_nonzero(temp)
    compactness = (whiteCount/size)*100
    # print("Compactness ",compactness,"%",numb)
    if compactness < 2:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (5,5), cv2.BORDER_CONSTANT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(ogSlice)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        # print("Compactness ",compactness,"%",numb)
    if compactness > 4:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (9,9), cv2.BORDER_REFLECT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(ogSlice)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        # print("Compactness ",compactness,"%",numb)
    if compactness > 5:
        temp = ogSlice.copy()
        temp = cv2.GaussianBlur(ogSlice, (11,11), cv2.BORDER_CONSTANT)
        temp = cv2.Canny(temp,100,175)
        size = np.size(ogSlice)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        # print("Compactness ",compactness,"%",numb)
    if compactness < 1.15:
        temp = ogSlice.copy()
        # temp = cv2.GaussianBlur(ogSlice, (3,3), cv2.BORDER_CONSTANT)
        # threshold
        
        temp = cv2.Canny(temp,100,175)
        size = np.size(ogSlice)
        whiteCount = np.count_nonzero(temp)
        compactness = (whiteCount/size)*100
        # print("Compactness ",compactness,"%",numb)
    return temp

def v4(img,numb,PARAMETERS,dictOfBinaryMask):
    thres = PARAMETERS[0]
    classIds, confs, bbox = net4.detect(img,confThreshold = thres)
    # print("BOX: ",bbox[0])
    # x1 = bbox[0][0]
    # y1 = bbox[0][1]
    # x2 = bbox[0][2]
    # y2 = bbox[0][3]
    boxes = []
    confidenceList = []
    imgNew = img.copy()
    imgNew.fill(0)
    if not isinstance(classIds,tuple):
        for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]
            boxes.append([x1,y1,x2,y2])
            confidenceList.append(confs)
            # cv2.rectangle(img,[x1,y1,x2,y2],color=(0,0,255),thickness=2)

        maxX = 0
        maxY = 0
        minX = len(img)*len(img[0])
        minY = len(img)*len(img[0])
        for b in boxes:
            for i in range(b[0],b[2]+b[0]):
                for j in range(b[1],b[3]+b[1]):
                    imgNew[j][i] = img[j][i]
        
        for b in boxes:
            maxX = max(maxX,b[0]+b[2])
            maxY = max(maxY,b[1]+b[3])
            minX = min(minX,b[0])
            minY = min(minY,b[1])
            
        
        fMask = np.copy(img)
        fMask.fill(0)
        ogSlice = img[minY:maxY,minX:maxX]
        ogSlice = find_optimal_lines(ogSlice,numb)
        ogSlice = cv2.dilate(ogSlice,(7,7))
        
        newSlice = ogSlice.copy()
        newSlice.fill(0)

        for b in boxes:
            for i in range(b[0]-minX,(b[2]+b[0])-minX):
                for j in range(b[1]-minY,(b[3]+b[1])-minY):
                    newSlice[j][i] = ogSlice[j][i]

        ogSlice = newSlice
        cpSlice = ogSlice.copy()
        
        smallRange = int(max(len(ogSlice)/PARAMETERS[1],len(ogSlice[0])/PARAMETERS[1]))
        longRange = int(max(len(ogSlice)/PARAMETERS[2],len(ogSlice[0])/PARAMETERS[2]))
        workVal = 0
        
        
        
        ogSlice = cpSlice.copy()
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = fillSmallSpace(ogSlice,smallRange,i,j)
                if(ogSlice[i][j] != 255):
                    if workVal == 2:
                        cpSlice[i][j] = 255            
        
        ogSlice = cpSlice.copy()
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = workFunc(ogSlice,longRange,i,j)
                if(ogSlice[i][j] != 255):
                    if workVal > 3:
                        cpSlice[i][j] = 255
        ogSlice = cpSlice.copy()
        # print(tempRange)
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = fillSmallSpace(ogSlice,smallRange,i,j)
                if(ogSlice[i][j] != 255):
                    if workVal == 2:
                        cpSlice[i][j] = 255
                        ogSlice[i][j] = 255
        # imshow("ogSlice",ogSlice)
        # imshow("cpSlice",cpSlice)
        # waitKey(0)
        ogSlice = cpSlice.copy()
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = workFunc(ogSlice,smallRange,i,j)
                if(ogSlice[i][j] != 255):
                    if workVal == 4:
                        cpSlice[i][j] = 255
                elif(ogSlice[i][j] != 0):
                    if workVal < 2:
                        # ogSlice[i][j] = 0
                        cpSlice[i][j] = 0
        
        # imshow("ogSlice",ogSlice)
        # imshow("cpSlice",cpSlice)
        # waitKey(0)    
        rmSlice = cpSlice.copy()
        drcontours = rmSlice.copy()
        drcontours = cv2.cvtColor(drcontours, cv2.COLOR_GRAY2RGB)
        removeIslands = cv2.pyrDown(rmSlice)
        _, threshed = cv2.threshold(rmSlice, 0, 255, cv2.THRESH_BINARY)
        contours,_ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print("Cont",contours)
        #find maximum contour and draw
        
        # print("NUMB",numb)
        if len(contours) > 0:
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
            
        # print("Cont",cmax)
        

        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                fMask[minY+i][minX+j] = cpSlice[i][j]
        fMask = cv2.cvtColor(fMask, cv2.COLOR_BGR2GRAY)
        dictOfBinaryMask[numb] = fMask
    else:
        fMask = img.copy()
        fMask = cv2.cvtColor(fMask, cv2.COLOR_BGR2GRAY)
        fMask.fill(0)
        dictOfBinaryMask[numb] = fMask

def findOptimalParams(listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain):
    for i in range(1):
        PARAMETERS = selectParameters4()
        print("PARAMS ",PARAMETERS)
        if PARAMETERS in evaluationParameters:
            break
        evaluationParameters.append(PARAMETERS)#[PARAMETERS[0],PARAMETERS[1],PARAMETERS[2]]
        # print(evaluationParameters)
        # addToLog(PARAMETERS.copy(),"Parameters")
        jobs = []
        for image in datasetTrain:
            current = imread(imageDataLocation+image)
            p1 = Process(target=v4,args=[current,image,PARAMETERS,dictOfBinaryMask])
            p1.start()
            jobs.append(p1)                    
        for job in jobs:
            job.join()
        averageIOU=0
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

def runTestV4():
    listTestEvaluations = []
    numRange = 50
    for i in range(numRange):
        if i % 10 == 0 and numRange > 9:
            print(i*2,"% Done")
        elif numRange < 10:
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
            for i in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            for i in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            for i in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            for i in range(6):
                p1 = Process(target=findOptimalParams,args=[listOfEvaluations,evaluationParameters,dictOfBinaryMask,PARAMETERS,datasetTrain])
                p1.start()
                jobs.append(p1)
            for job in jobs:
                job.join()
            jobs.clear()
            print("Length of listOfEvaluations",len(listOfEvaluations))
            print("Length of evaluationParameters",len(evaluationParameters))
            print("Length of listTestEvaluations",len(listTestEvaluations))
            optimalParamsResults = listOfEvaluations[listOfEvaluations.index(max(listOfEvaluations))]
            optimalParams = evaluationParameters[listOfEvaluations.index(max(listOfEvaluations))]
            addToLog(optimalParamsResults,"Evaluation Score Train Set")
            print("Second part")
            
            for image in datasetVal:
                current = imread(imageDataLocation+image)
                p1 = Process(target=v4,args=[current,image,optimalParams,dictOfBinaryMask])
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
    averageTestScore = sum(listTestEvaluations)/len(listTestEvaluations)
    addToLog(averageTestScore,"Average Validation Score")
    addToLog(calculateSTD(listTestEvaluations,averageTestScore), "Standard Deviation")
    addToLog("V4","Pipeline: ")

if __name__ == "__main__":
    start_time = time.time()
    runTestV4()
    print("--- %s seconds ---" % (time.time() - start_time))
    runTimeNumber = str(runTimeCount()) 
    saveLog()
    print("Run time number ",runTimeNumber)