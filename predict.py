import os

# Try running on CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import cv2
from keras.models import load_model

MODEL_NAME = './model.h5'
model = load_model(MODEL_NAME)

# Dont forget to delete gitkeep
for root, dirs, files in os.walk('./input', topdown=False):
    for name in files:
        print(os.path.join(root, name))

        im = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)

        im_predict = np.reshape(im, (1, im.shape[0], im.shape[1], 1))
        im_predict = im_predict.astype(np.float32) / 255 * 2 - 1

        result = model.predict(im_predict)

        im_res = (result[0] + 1) / 2 * 255

        cv2.imwrite(os.path.join('./output', name), im_res)
