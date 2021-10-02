import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow

(X_train, y_train),(X_test, y_test) = mnist.load_data() 
