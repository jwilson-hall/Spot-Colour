#C1949699
#
#
#
#
#
import cv2
from matplotlib.pyplot import imshow
import numpy as np
import os
from sympy import continued_fraction_iterator


def static_saliency(img, numb):
    
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()#
    (success, saliencyMap) = saliency.computeSaliency(img)
    newSaliencyMap=saliencyMap*255

    threshMap = cv2.threshold(newSaliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    output_img = gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    for i in range(len(threshMap)):
        for j in range(len(threshMap[0])):
            if (threshMap[i][j]==0):
                img[i][j] = np.array([output_img[i][j],output_img[i][j],output_img[i][j]])

    # Show Images
    # cv2.imshow("Output Image", img)
    # cv2.imshow("Thresholds", threshMap)
    # cv2.imshow("Output", saliencyMap)
    # cv2.waitKey(0)
    # Writing Images
    cv2.imwrite(data_path+"train_output\\v1\\"+str(numb)+"_p.jpg",img)
    cv2.imwrite(data_path+"train_output\\v1\\"+str(numb)+"_thr.jpg",threshMap)


train_data_set = "train_data\\"
test_data_set = "test_data\\"
data_path = os.getcwd()+"\\"
files = os.listdir(data_path)

# img = cv2.imread(data_path+train_data_set+str(1)+'.jpg')
# static_saliency(img,1)
for i in range(1,11):
    img = cv2.imread(data_path+train_data_set+str(i)+'.jpg')
    static_saliency(img,i)
