import cv2
import os
import numpy as np
# from gevent import config
thres = 0.45 # Threshold to detect object
image_path = os.path.dirname

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'#paths for network

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)#defining network configuration



def run_algorithm(img, numb):
    classIds, confs, bbox = net.detect(img,confThreshold = 0.5)    
    boxes = []
    confidenceList = []
    for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):#get bounding boxes from network, parsing information to list
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        boxes.append([x1,y1,x2,y2])
        confidenceList.append(confidence)
        

    

    # Initialise static saliency fine grained detector and compute the saliencyMap
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    newSaliencyMap=saliencyMap*255

    threshMap = cv2.threshold(newSaliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]#globally thresholding the image using otsu's algorithm creating a binary mask

    output_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#creating a greyscale copy of the original image
    testGrey = img.copy()#creates a colour copy of the original image
    for i in range(len(img)):
        for j in range(len(img[0])):
            testGrey[i][j] = np.array([output_img[i][j],output_img[i][j],output_img[i][j]])
    
    maxX = 0
    maxY = 0
    minX = len(img)*len(img[0])
    minY = len(img)*len(img[0])
    newThrMap = threshMap.copy()
    newThrMap.fill(0)
    # boxes.remove(boxes[0])
    for b in boxes:
        maxX = max(maxX,b[0],b[2])#finding the overall bounding area of all the boxes
        maxY = max(maxY,b[1],b[3])
        minX = min(minX,b[2],b[0])
        minY = min(minY,b[3],b[1])
    print(minX,":",minY,"  ",maxX,":",maxY)
    for b in boxes: # finding the bounding boxes area
        x1 = b[0]
        y1 = b[1]
        x2 = b[2]+b[0]#the b[2] and b[3] represents a length not a coordinate so adding b[0] to b[2] finds the coord of the second point
        y2 = b[3]+b[1]
        
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (threshMap[i][j] == 255 and (j >= x1 and i >= y1) and (j <= x2 and i <= y2)):#selecting pixels that are only inside the bounding boxes and where the mask is white, making the ROI in colour
                    testGrey[i][j] = img[i][j]
                    newThrMap[i][j] = 255

    # cv2.imshow("Output",testGrey)
    # cv2.imshow("Threshold map",newThrMap)
    # data_path = os.path.dirname(os.getcwd())+"\\data\\"
    # Writing Images
    cv2.imwrite(data_path+"train_output\\v2\\"+str(numb)+"_p.jpg",testGrey)#writing output mask and final product of the use of that mask
    cv2.imwrite(data_path+"train_output\\v2\\"+str(numb)+"_thr.jpg",newThrMap)
    # cv2.imwrite(data_path+"test_output\\v2\\"+str(numb)+"_p.jpg",testGrey)#writing output mask and final product of the use of that mask
    # cv2.imwrite(data_path+"test_output\\v2\\"+str(numb)+"_thr.jpg",newThrMap)
    # cv2.imshow("Output image",output_img)
    # cv2.imshow("Output", saliencyMap)
    # cv2.waitKey(0)
data_path = os.getcwd()+"\\"
# img = cv2.imread(data_path+"test_data\\"+str(1)+'.jpg')
# run_algorithm(img, 1)
#running the algorithm over the 10 train and test images
for i in range(1, 11):
    img = cv2.imread(data_path+"train_data\\"+str(i)+'.jpg')
    run_algorithm(img, i)
# for i in range(1, 11):
#     img = cv2.imread(data_path+"test_data\\"+str(i)+'.jpg')
#     run_algorithm(img, i)