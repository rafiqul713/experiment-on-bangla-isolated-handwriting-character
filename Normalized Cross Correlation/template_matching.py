import numpy as np
import cv2
import glob
import glob
import os.path
from os.path import basename
from skimage import util
from skimage.filter import threshold_otsu

from skimage import data
from skimage.feature import match_template

#Form Path
PATH=""

formList=glob.glob(PATH+"*.tif")
print("Total Form found in this Folder:  " + str(len(formList)))
for form in formList:
    readedForm = cv2.imread(form)
    formName=os.path.basename(form)
    formName=os.path.splitext(formName)[0]
    print ("Now Extract from FROM NAME: "+formName)

    #convert to Grayscale
    imageGray = cv2.cvtColor(readedForm, cv2.COLOR_BGR2GRAY)
    image = imageGray
    #Template
    templatePath=""
    os.chdir(templatePath)
    templateList=glob.glob("*.png")
    templateList=sorted(templateList)
    for img in templateList:
        imageName = os.path.basename(img)
        templateTitle=os.path.splitext(imageName)[0]
        print(templateTitle)
        cv_img = cv2.imread(img)
        template = cv2.imread(imageName)

        # convert to Grayscale
        templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = templateGray
        # convert to binary
        _, image = cv2.threshold(imageGray, 170, 255, cv2.THRESH_BINARY)
        _, template = cv2.threshold(templateGray, 170, 255, cv2.THRESH_BINARY)
        image = ~image
        template = ~template
        # get the shape of the template
        w, h = template.shape[::-1]

        # find the image by cv2
        result = cv2.matchTemplate(image, template,
                               cv2.TM_CCORR_NORMED) 
        result = result * result
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        x, y = max_loc

        print(np.max(result))

        # extract images from the form
        extractedImage = image[y + 5: y + h, x + w: x + w + w]
        result = result * 255
	savePath=""
        cv2.imwrite(savePath+ "/" + formName + "_" + templateTitle + ".png",
            extractedImage)
