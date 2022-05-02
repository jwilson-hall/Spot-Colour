#C1949699
#
#
#
#
#
import cv2
import numpy as np
import os

def static_saliency(img, numb):
    
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()# calculating the static saliency fine grained map
    (success, saliencyMap) = saliency.computeSaliency(img)
    newSaliencyMap=saliencyMap*255#opencv returns a floating point number when computing saliency so multiplying by 255 sets it to be within the limits of 0-255
    cv2.imshow("Saliency", saliencyMap)
    threshMap = cv2.threshold(newSaliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]#thresholding saliency map using otsu's algorithm to get a binary mask that is used when choosing areas that should be in colour
    
    output_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #creating a greyscale copy of the original image
    output_img = cv2.cvtColor(output_img,cv2.COLOR_GRAY2RGB)#converting the greyscale image back to rgb values so that we can set the greyscale value to the RGB tuple
    
    
    for i in range(len(threshMap)):#where the image is black, the original image is put into greyscale for those regions
        for j in range(len(threshMap[0])):
            if (threshMap[i][j]==0):
                img[i][j] = output_img[i][j]

    # cv2.imwrite(data_path+"train_output\\v1\\"+str(numb)+"_p.jpg",img)
    # cv2.imwrite(data_path+"train_output\\v1\\"+str(numb)+"_thr.jpg",threshMap)
    # cv2.imwrite(data_path+"test_output\\v1\\"+str(numb)+"_p.jpg",img)
    # cv2.imwrite(data_path+"test_output\\v1\\"+str(numb)+"_thr.jpg",threshMap)
    cv2.imshow("Output image",img)
    cv2.imshow("Threshold", threshMap)
    cv2.waitKey(0)

data_path = os.getcwd()+"\\"
files = os.listdir(data_path)
# img = cv2.imread(data_path+"test_data\\"+str(1)+'.jpg')
# static_saliency(img,1)
#runnning test and train images
for i in range(8,9):
    img = cv2.imread(data_path+"train_data\\"+str(i)+'.jpg')
    static_saliency(img,i)
# for i in range(1,11):
#     img = cv2.imread(data_path+"test_data\\"+str(i)+'.jpg')
#     static_saliency(img,i)
