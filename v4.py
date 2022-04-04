from concurrent.futures import process
import cv2
import os
import numpy as np
from multiprocessing import Process
# from numba import jit
# from gevent import config
thres = 0.5 # Threshold to detect object
image_path = os.path.dirname

# configPath = 'frozen_inference_graph.pb'
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)


# test_data_path = os.path.dirname(os.getcwd())+"\\data\\new_test_data\\"

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



def smoothing(ogSlice, cpSlice,imgRange,iterations):
    for i in range(iterations):
        if(imgRange > len(ogSlice)/2):
            imgRange = int(imgRange*0.5) 
        print("pixel range ",imgRange)
        ogSlice = cpSlice.copy()
        print("Iteration: ",i+1)
        # imgRange=int(imgRange/2)
        for i in range(len(ogSlice)):
            for j in range(len(ogSlice[0])):
                workVal = workFunc(ogSlice,imgRange,i,j)
                if(ogSlice[i][j] == 0):
                    if workVal > 3:
                        cpSlice[i][j] = 255
                elif(ogSlice[i][j] == 255):
                    if workVal < 3:
                        cpSlice[i][j] = 0
    return cpSlice

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
    print("Compactness ",compactness,"%",numb)
    return temp


def run_algorithm(img, numb):
    classIds, confs, bbox = net.detect(img,confThreshold = thres)
    # print("BOX: ",bbox[0])
    # x1 = bbox[0][0]
    # y1 = bbox[0][1]
    # x2 = bbox[0][2]
    # y2 = bbox[0][3]
    boxes = []
    confidenceList = []
    imgNew = img.copy()
    imgNew.fill(0)
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
        # imgNew[b[1]:b[0],b[1]+b[3]:b[0]+b[2]] = img[b[1]:b[0],b[1]+b[3]:b[0]+b[2]] 
        # imgNew[y1:x1,y2+y1:x2+x1] = img[y1:x1,y2+y1:x2+x1]
    # print([minX,minY,maxX,maxY])
    # print(boxes)
    
    # for i in range(len(img)):
    #     for j in range(len(img[0])):
    #         if not ((j >= minX and i >= minY) and (j <= maxX and i <= maxY)):
    #             img[i][j] = [0,0,0]
    # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    
    fOutput = np.copy(img)    
    fMask = np.copy(img)
    fMask.fill(0)
    ogSlice = img[minY:maxY,minX:maxX]
    ogSlice = find_optimal_lines(ogSlice,numb)
    ogSlice = cv2.dilate(ogSlice,(7,7))
    
    # imshow("Lines:",ogSlice)
    # waitKey(0)
    newSlice = ogSlice.copy()
    newSlice.fill(0)

    for b in boxes:
        for i in range(b[0]-minX,(b[2]+b[0])-minX):
            for j in range(b[1]-minY,(b[3]+b[1])-minY):
                newSlice[j][i] = ogSlice[j][i]

    ogSlice = newSlice
    # ogSlice = cv2.dilate(ogSlice,(3,3), iterations=2)
    cpSlice = ogSlice.copy()
    imgRange = int(max(len(ogSlice)*0.08,len(ogSlice[0])*0.08))
    # print("Range: ",imgRange)
    # imgRange = int(len(ogSlice)/5)
    # print("Pixel amount ",workFunc(ogSlice,imgRange,99,98))
    # cpSlice[99][98] = 0
    workVal = 0
    # for i in range(len(ogSlice)):
    #     for j in range(len(ogSlice[0])):
    #         workVal = workFunc(ogSlice,imgRange,i,j)            
    #         if(ogSlice[i][j] != 0):
    #             if workVal < 4:
    #                 cpSlice[i][j] = 0
    # ogSlice = cpSlice.copy()
    
    
    ogSlice = cpSlice.copy()
    for i in range(len(ogSlice)):
        for j in range(len(ogSlice[0])):
            workVal = fillSmallSpace(ogSlice,int(max(len(ogSlice)/100,len(ogSlice[0])/100)),i,j)
            if(ogSlice[i][j] != 255):
                if workVal == 2:
                    cpSlice[i][j] = 255
            # elif(ogSlice[i][j] != 0):
            #     if workVal < 2:
            #         # ogSlice[i][j] = 0
            #         cpSlice[i][j] = 0
    # imshow("ogSlice",ogSlice)
    # imshow("cpSlice",cpSlice)
    # waitKey(0)
    
    ogSlice = cpSlice.copy()
    for i in range(len(ogSlice)):
        for j in range(len(ogSlice[0])):
            workVal = workFunc(ogSlice,int(max(len(ogSlice)/3,len(ogSlice[0])/3)),i,j)
            if(ogSlice[i][j] != 255):
                if workVal > 3:
                    cpSlice[i][j] = 255
    ogSlice = cpSlice.copy()
    tempRange = int(max(len(ogSlice)/50,len(ogSlice[0])/50))
    # print(tempRange)
    for i in range(len(ogSlice)):
        for j in range(len(ogSlice[0])):
            workVal = fillSmallSpace(ogSlice,tempRange,i,j)
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
            workVal = workFunc(ogSlice,imgRange,i,j)
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
    # ogSlice = cpSlice.copy()
    # cpSlice = smoothing(ogSlice,cpSlice,imgRange,iterations=4)
    # imgRange=int(2*imgRange)
    # print("Range: ",imgRange)
    # imshow("ogSlice",ogSlice)
    # imshow("cpSlice",cpSlice)
    
    # imgRange=int(imgRange/2)

    # ogSlice = cpSlice.copy()
    # for i in range(len(ogSlice)):
    #     for j in range(len(ogSlice[0])):
    #         workVal = workFunc(ogSlice,imgRange,i,j)
    #         if(ogSlice[i][j] == 0):
    #             if workVal > 2:
    #                 cpSlice[i][j] = 255
    #         elif(ogSlice[i][j] == 255):
    #             if workVal < 3:
    #                 cpSlice[i][j] = 0

    # ogSlice = cpSlice.copy()
    # imgRange=int(imgRange*4)
    # for i in range(len(ogSlice)):
    #     for j in range(len(ogSlice[0])):
    #         workVal = workFunc(ogSlice,imgRange,i,j)
    #         if(ogSlice[i][j] == 0):
    #             if workVal > 3:
    #                 cpSlice[i][j] = 255
    #         elif(ogSlice[i][j] == 255):
    #             if workVal < 2:
    #                 cpSlice[i][j] = 0

    for i in range(len(ogSlice)):
        for j in range(len(ogSlice[0])):
            fMask[minY+i][minX+j] = cpSlice[i][j]
    
    fOutput = cv2.cvtColor(fOutput, cv2.COLOR_BGR2GRAY)     
    fOutput = cv2.cvtColor(fOutput, cv2.COLOR_GRAY2RGB)

    for i in range(len(fOutput)):
        for j in range(len(fOutput[0])):
            if fMask[i][j][0] == 255:
                fOutput[i][j] = img[i][j]
        
    # imshow("fMask",fMask)
    # imshow("fOutput",fOutput)
    # imshow("ogSlice",ogSlice)
    # imshow("cpSlice",cpSlice)
    # cv2.imwrite(data_path+"v4\\"+str(numb)+"_ogSlice.jpg",cpSlice)
    # imshow("threshMap",threshMap)
    print("Done with Image: ",numb)
    # cv2.imwrite(data_path+"train_output\\v4\\"+str(numb)+"_thr.jpg",fMask)
    # cv2.imwrite(data_path+"train_output\\v4\\"+str(numb)+"_p.jpg",fOutput)

    # cv2.imwrite(data_path+"test_output\\v4\\"+str(numb)+"_thr.jpg",fMask)
    # cv2.imwrite(data_path+"test_output\\v4\\"+str(numb)+"_p.jpg",fOutput)
    # cv2.waitKey(0)

# img = cv2.imread(data_path+"test_data\\"+'6.jpg')
# run_algorithm(img, 1)
data_path = os.getcwd()+"\\"

# for i in range(1,11):
#     img = cv2.imread(data_path+"train_data\\"+str(i)+'.jpg')
#     if __name__ == '__main__':
#         p1 = Process(target=run_algorithm,args=[img,i])
#         p1.start()

for i in range(6,7):
    # if i != 3:
    #     if i != 7:
    img = cv2.imread(data_path+"test_data\\"+str(i)+'.jpg')
    if __name__ == '__main__':
        p1 = Process(target=run_algorithm,args=[img,i])
        p1.start()