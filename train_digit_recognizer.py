#importing libraries.
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.losses import categorical_crossentropy

#Loading the dataset and splitting it into test and train.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshaping the image in the form it is there.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

#Dividing by 255 because the maximum pixel is 255 and dividing by 255 will give a small range of [0,1].
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Building of CNN.
model = Sequential()
#Adding the first convoluted layer with a 5 x 5 feature detector.
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
#Adding a pooling layer with a kernel of 3 x 3.
model.add(MaxPooling2D(pool_size=(3, 3)))

#Adding the second convoluted layer and pooling layer.
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#Making of the fully connected layer.
model.add(Flatten())
#Adding the first hidden layer.
model.add(Dense(128, activation='relu'))
#Adding a dropout of 0.3, to prevent overfitting.
model.add(Dropout(0.3))
#Adding second hidden layer.
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

#Using softmax function because we need probabilistic output for each class.
model.add(Dense(10, activation='softmax'))
#compiling of the model.
model.compile(loss=categorical_crossentropy,optimizer="adam",metrics=['accuracy'])

#Saving of the model.
digit_recognition = model.fit(x_train, y_train,batch_size=100,epochs=15,verbose=1,validation_data=(x_test, y_test))
print("The model has successfully trained")

#Printing the test accuracy and the test loss.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Saving the model
model.save('digit_recognition.h5')
print("Saving the model as digit_recognition.h5")

