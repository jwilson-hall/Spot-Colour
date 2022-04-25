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
dataset_path = data_path+"\\train_data\\"

toShow = cv2.imread(dataset_path+"8.jpg")
cv2.imshow("File",toShow)
cv2.waitKey(0)

