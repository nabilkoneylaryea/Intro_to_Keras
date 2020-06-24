import tensorflow
import keras

from keras.models import Sequential #dot notation after 'keras' indicates a package
                                    #to access actual functions need to say from keras.package import funciton

from keras.layers import Dense, Dropout #from layers package what layers we want
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

#Initializing model
model = Sequential() #sequential represents the layers we are making and what of type

#Layers: Convulutional, Max Pooling, Dense, Dropout
conv_layer = Conv2D(filter=32, kernel_size=(3, 3), activation='relu')
max_pool_layer = MaxPooling2D(pool_size=(2, 2))
dense_layer = Dense(units=10, activation='softmax')
dropout_layer = Dropout(rate=0.5)

#Adding layers to model
model.add(conv_layer)
model.add(max_pool_layer)
model.add(dense_layer)
model.add(dropout_layer)

#Training model
model.compile(optimizer="SGD", loss=0.01, metrics=['accuracy'])
model.fit()

#Other stuff
model.evaluate()
model.predict()