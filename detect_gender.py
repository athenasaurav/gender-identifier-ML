#######################################################################
# Copyright (C)                                                       #
# 2018-2020 Abhinav Sagar(abhinav.sagar2016@vitstudent.ac.in)         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import cvlib as cv


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = ap.parse_args()

# download pre-trained model file (one-time download)
dwnld_link = "https://drive.google.com/open?id=1r1a44yDyuwCXQAClbN9VY-qK0Bjt4U74"
model_path = get_file("weights.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())
                     
# load model
model = load_model(model_path)

# read input image
image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()

# load pre-trained model
model = load_model(model_path)

# detect faces in the image
face, confidence = cv.detect_face(image)

classes = ['man','woman']

# loop through detected faces
for idx, f in enumerate(face):

     # get corner points of face rectangle       
    (startX, startY) = f[0], f[1]
    (endX, endY) = f[2], f[3]

    # draw rectangle over face
    cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

    # crop the detected face region
    face_crop = np.copy(image[startY:endY,startX:endX])

    # preprocessing for gender detection model
    face_crop = cv2.resize(face_crop, (96,96))
    face_crop = face_crop.astype("float") / 255.0
    face_crop = img_to_array(face_crop)
    face_crop = np.expand_dims(face_crop, axis=0)

    # apply gender detection on face
    conf = model.predict(face_crop)[0]
    print(conf)
    print(classes)

    # get label with max accuracy
    idx = np.argmax(conf)
    label = classes[idx]

    label = "{}: {:.2f}%".format(label, conf[idx] * 100)

    Y = startY - 5 if startY - 5 > 5 else startY + 5

    # write label and confidence above face rectangle
    cv2.putText(image, label, (startX, Y),  cv2.FONT_HERSHEY_DUPLEX,
                0.7, (0, 255, 0), 2)

# display output
cv2.imshow("gender detection", image)

# press any key to close window           
cv2.waitKey()

# save output
cv2.imwrite("output.jpg", image)

# release resources
cv2.destroyAllWindows()
