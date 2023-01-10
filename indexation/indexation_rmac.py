import numpy as np
import tensorflow as tf
import os 
from rmac import RMAC

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda

# load the pretrained network from Keras Applications
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


folder = 'web_server/app/static/images/'
def generateRmac():
    # load the base model
    base_model = MobileNetV2()
    # create the new model consisting of the base model and a RMAC layer
    layer = "out_relu"
    base_out = base_model.get_layer(layer).output
    rmac = RMAC(base_out.shape, levels=5, norm_fm=True, sum_fm=True)
    # add RMAC layer on top
    rmac_layer = Lambda(rmac.rmac, input_shape=base_model.output_shape, name="rmac_"+layer)
    out = rmac_layer(base_out)
    #out = Dense(1024)(out) # fc to desired dimensionality
    model = Model(base_model.input, out)
    input_dim = 224
    base_model = model
    for path in os.listdir(folder):
      if '.jpg' in path:
        data = os.path.join(folder, path)
        file_name = os.path.basename(data) 
        image = load_img(data, target_size=(input_dim, input_dim))
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        if image is not None:
            feature = base_model.predict(image) # predict the probability
            feature = np.array(feature[0])
            np.savetxt("index/RMAC"+"/"+os.path.splitext(file_name)[0]+".txt", feature)
    return data
generateRmac()