from random import random
import cv2
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import PIL.ImageOps
import os
import ssl
import time

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

if(not os.environ.get("PYTHONHTTPSVERIFY", "") and getattr(ssl, "create_unverified_context", None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
nClasses = len(classes)

xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, random_state=9, train_size=7500, test_size=2500)

xtrainScaled = xtrain/255.0
xtestScaled = xtest/255.0

clf = LogisticRegression(
    solver="saga", multi_class="multinomial").fit(xtrainScaled, ytrain)
yPred = clf.predict(xtestScaled)
accuracy = accuracy_score(ytest, yPred)
print("Accuracy: ", accuracy)

cap = cv2.VideoCapture(0)

while(True):
    try:
        rect, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        upperLeft = (int(width/2 - 56), int(height/2-56))
        bottomRight = (int(width/2+56), int(height/2+56))

        cv2.rectangle(gray, upperLeft, bottomRight, (0, 255, 0), 2)

        roi = gray[upperLeft[1]:bottomRight[1], upperLeft[0]:bottomRight[0]]

        im_pil = Image.fromarray(roi)
        image_bw = im_pil.convert("L")
        image_bw_resize = image_bw.resize((28, 28), Image.Resampling.LANCZOS)
        image_bw_resize_inverted = PIL.ImapeOps.invert(image_bw_resize)

        pixel_filter = 20
        min_pixel = np.percentile(image_bw_resize_inverted, pixel_filter)
        image_bw_resize_inverted_scaled = np.clip(
            image_bw_resize_inverted - min_pixel, 0, 255)
        max_pixel = np.max(image_bw_resize_inverted)

        image_bw_resize_inverted_scaled = np.asarray(
            image_bw_resize_inverted_scaled) / max_pixel

        test_sample = np.array(image_bw_resize_inverted_scaled).reshape(1, 784)

        test_pred = clf.predict(test_sample)
        print("Predicted Classes: ", test_pred)
        cv2.imshow("frame", gray)

        if(cv2.waitKey(1) and 0xFF == ord("q")):
            break

    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()
