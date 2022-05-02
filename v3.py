import cv2
from cv2 import imshow
from cv2 import waitKey
from cv2 import INTER_LINEAR
from cv2 import INTER_NEAREST
from cv2 import imwrite
from cv2 import INTER_BITS
from cv2 import INTER_LANCZOS4
from cv2 import INTER_LINEAR_EXACT
from cv2 import INTER_AREA
from cv2 import INTER_CUBIC
import numpy as np
import os
from multiprocessing import Process
# Loading the Mask RCNN model
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


def mask_rcnn(img, numb):
		
	H, W, _ = img.shape

	# Create black image with same dimensions as input image
	black_image = np.zeros((H, W, 1), np.uint8)


	# Detect objects inside input image
	blob = cv2.dnn.blobFromImage(img, swapRB=True)
	net.setInput(blob)

	boxes, masks = net.forward(["detection_out_final", "detection_masks"])
	detection_count = boxes.shape[2]

	for i in range(detection_count):
		box = boxes[0, 0, i]
		class_id = box[1]
		# print(class_id)
		confs = box[2]
		if confs < 0.5:
			continue

		# Get box coordinates
		box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
		(x1, y1, x2, y2) = box.astype('int')
		roi_Width = x2 - x1
		roi_Height = y2 - y1
		roi = black_image[y1: y2, x1: x2]
		
		# roi_height, roi_width, _ = roi.shape
		# cv2.imshow("mask 1",roi)
		# Get the mask
		mask = masks[i, int(class_id)]
		# imwrite(data_path+"MaskNew.jpg",mask)		
		mask = cv2.resize(mask, (roi_Width, roi_Height),interpolation=INTER_CUBIC)
		_, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
		# cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
		# Get mask coordinates
		contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for cnt in contours:
			cv2.fillPoly(roi, [cnt], 255)

	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for i in range(len(img)):
		for j in range(len(img[0])):
			if black_image[i][j] != 255:			
				img[i][j] = grey[i][j]


	cv2.imshow("Image", img)
	cv2.imshow("Black image", black_image)
	cv2.waitKey(0)
	print("Done with image ",numb)
	# cv2.imwrite(data_path+"train_output\\v3\\"+str(numb)+"_p.jpg",img)
	# cv2.imwrite(data_path+"train_output\\v3\\"+str(numb)+"_thr.jpg",black_image)
	# cv2.imwrite(data_path+"test_output\\v3\\"+str(numb)+"_p.jpg",img)
	# cv2.imwrite(data_path+"test_output\\v3\\"+str(numb)+"_thr.jpg",black_image)
	

# looks for current working directory
data_path = os.getcwd()+"\\"

#creates a new process per image as processing time has increased as our solution becomes more complicated
# img = cv2.imread(data_path+'2.jpg')
# mask_rcnn(img,2)
# p1 = Process(target=mask_rcnn,args=[img,2])
# p1.start()
for i in range(8,9):
    img = cv2.imread(data_path+"train_data\\"+str(i)+'.jpg')
    if __name__ == '__main__':
        p1 = Process(target=mask_rcnn,args=[img,i])
        p1.start()

# for i in range(1,2):
#     img = cv2.imread(data_path+"test_data\\"+str(i)+'.jpg')
#     if __name__ == '__main__':
#         p1 = Process(target=mask_rcnn,args=[img,i])
#         p1.start()