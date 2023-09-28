import os

import cv2
import matplotlib.pyplot as plt
import imutils

from tqdm import tqdm

INPUT_FOLDER_PATH = "PATH/TO/INPUT/FOLDER"
OUTPUT_FOLDER_PATH = "PATH/TO/OUTPUT/FOLDER"

if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

for elm in tqdm(os.listdir(INPUT_FOLDER_PATH)):

    img = cv2.imread(os.path.join(INPUT_FOLDER_PATH, elm))
    img = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    tresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    tresh = cv2.erode(tresh, None, iterations=2)
    tresh = cv2.dilate(tresh, None, iterations=2)

    cnts = cv2.findContours(tresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:,:,0].argmin()][0])
    extRight = tuple(c[c[:,:,0].argmax()][0])
    extTop = tuple(c[c[:,:,1].argmin()][0])
    extBot = tuple(c[c[:,:,1].argmax()][0])

    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                  extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS]
    
    fname = elm.split(".")[0]
    fname = fname + "_cropped.png"

    savepath = os.path.join(OUTPUT_FOLDER_PATH, fname)

    plt.imsave(savepath, new_img)