import cv2
import numpy as np
import os
from multiprocessing import Process
# Loading the Mask RCNN model
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb","dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


def mask_rcnn(img, numb):
		
	height, width, _ = img.shape

	# Create black image with same dimensions as input image
	black_image = np.zeros((height, width, 3), np.uint8)
	black_image[:] = (0, 0, 0)


	# Detect objects inside input image
	blob = cv2.dnn.blobFromImage(img, swapRB=True)
	net.setInput(blob)

	boxes, masks = net.forward(["detection_out_final", "detection_masks"])
	detection_count = boxes.shape[2]

	for i in range(detection_count):
		box = boxes[0, 0, i]
		class_id = box[1]
		# print(box)
		confs = box[2]
		if confs < 0.40:
			continue

		# Get box coordinates
		x = int(box[3] * width)
		y = int(box[4] * height)
		x2 = int(box[5] * width)
		y2 = int(box[6] * height)

		roi = black_image[y: y2, x: x2]
		roi_height, roi_width, _ = roi.shape

		# Get the mask
		mask = masks[i, int(class_id)]
		mask = cv2.resize(mask, (roi_width, roi_height))
		_, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

		# cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)

		# Get mask coordinates
		contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		white = [255,255,255]
		for cnt in contours:
			cv2.fillPoly(roi, [cnt], white)

		# cv2.imshow("roi", roi)
		# cv2.waitKey(0)
	grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for i in range(len(img)):
		for j in range(len(img[0])):
			if black_image[i][j][0] != 255:			
				img[i][j] = grey[i][j]


	# cv2.imshow("Image", img)
	# cv2.imshow("Black image", black_image)
	# cv2.waitKey(0)
	# cv2.imwrite(data_path+"train_output\\v3\\"+str(numb)+"_p.jpg",img)
	# cv2.imwrite(data_path+"train_output\\v3\\"+str(numb)+"_thr.jpg",black_image)
	cv2.imwrite(data_path+"test_output\\v3\\"+str(numb)+"_p.jpg",img)
	cv2.imwrite(data_path+"test_output\\v3\\"+str(numb)+"_thr.jpg",black_image)
	

# looks for current working directory
data_path = os.getcwd()+"\\"

#creates a new process per image as processing time has increased as our solution becomes more complicated

# for i in range(1,11):
#     img = cv2.imread(data_path+"train_data\\"+str(i)+'.jpg')
#     if __name__ == '__main__':
#         p1 = Process(target=mask_rcnn,args=[img,i])
#         p1.start()

for i in range(1,11):
    img = cv2.imread(data_path+"test_data\\"+str(i)+'.jpg')
    if __name__ == '__main__':
        p1 = Process(target=mask_rcnn,args=[img,i])
        p1.start()