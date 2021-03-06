import numpy as np
import cv2
import os

def calc_IoU(mask1, mask2):   # From the question.
    mask1_area = np.count_nonzero(mask1)
    mask2_area = np.count_nonzero(mask2)
    # print(mask1_area, " : ", mask2_area)
    intersection = np.count_nonzero(np.logical_and( mask1,  mask2))
    # print("intersection",intersection)
    iou = intersection/(mask1_area+mask2_area-intersection)
    return iou


data_path = os.getcwd()+"\\"
# ground_truth_path = data_path+"Ground_Truth\\test\\"
# ground_truth_path = data_path+"Ground_Truth\\train\\"
ground_truth_path = data_path+"hypothesis\\"

train_data_path1 = data_path+"train_output\\v1\\"
train_data_path2 = data_path+"train_output\\v2\\"
train_data_path3 = data_path+"train_output\\v3\\"
train_data_path4 = data_path+"train_output\\v4\\"
train_data_path5 = data_path+"train_output\\v5\\"

test_data_path1 = data_path+"test_output\\v1\\"
test_data_path2 = data_path+"test_output\\v2\\"
test_data_path3 = data_path+"test_output\\v3\\"
test_data_path4 = data_path+"test_output\\v4\\"
test_data_path5 = data_path+"test_output\\v5\\"

hypothesis_path3 = data_path+"hypothesis\\v3\\"
hypothesis_path5 = data_path+"hypothesis\\v5\\"
result_img = 1

def findMaxIoUHypothesis():
    Average_IoU = 0
    for j in range(0,7):
        IoU_Max = 0
        # for i in range(1,6):
        img_A = cv2.imread(ground_truth_path+str(result_img+j)+'_gt.png')
        img_B = cv2.imread(hypothesis_path5+str(result_img+j)+'_thr.jpg')
        # img_C = cv2.imread(ground_truth_path+'1_gt1.jpg')
        IoU_Max = max(calc_IoU(img_A,img_B),IoU_Max)
        print(calc_IoU(img_A,img_B))
        # print("Max: ",IoU_Max)
        # print("")
        Average_IoU+=IoU_Max
        # print("Image: ",result_img+j)
    print("Average Max: ",Average_IoU/7)

def findMaxIoU():
    Average_IoU = 0
    for j in range(10):
        IoU_Max = 0
        for i in range(1,6):
            img_A = cv2.imread(ground_truth_path+str(result_img+j)+'_gt'+str(i)+'.jpg')
            img_B = cv2.imread(train_data_path5+str(result_img+j)+'_thr.jpg')
            # img_C = cv2.imread(ground_truth_path+'1_gt1.jpg')
            IoU_Max = max(calc_IoU(img_A,img_B),IoU_Max)
            print(calc_IoU(img_A,img_B))
        # print("Max: ",IoU_Max)
        # print("")
        Average_IoU+=IoU_Max
        # print("Image: ",result_img+j)
    print("Average Max: ",Average_IoU/10)

findMaxIoUHypothesis()