import keras.applications as ka
from keras.layers import Dense
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import os
import smtplib
import keras.callbacks
from keras.layers import Input
import time, datetime
from keras.models import load_model
from keras.applications.vgg16 import  s