import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import tensorflow

(X_train, y_train),(X_test, y_test) = mnist.load_data() 

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

img_rows, img_cols = 28, 28

if keras.backend.image_data_format() == 'channel_first':
  X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols) 
  X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols) 
  input_shape = (1, img_rows, img_cols) 
else:
  X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
  X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
  input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /=255
X_test /=255

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=10) 
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=10)

model= Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1))) 
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Flatten()) 
model.add(Dense(units=128,activation='relu')) 
model.add(Dense(units=10, activation='softmax'))

model.summary()

model.compile(optimizer='adadelta',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,  
                    batch_size=128, 
                    epochs=20, 
                    validation_data=(X_test, y_test))

def make_accuracy_plot(history):
  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.set()
  acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
  epochs = range(1, len(acc) + 1)

  plt.figure(figsize=(10,8))
  plt.plot(epochs, acc, label="Training accuracy", marker="o")
  plt.plot(epochs, val_acc, label="validation accuracy", marker="o")
  plt.legend()
  plt.title("Accuracy of training and validation")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.show()

def make_loss_lot(history):
  import matplotlib.pyplot as plt
  import seaborn as sns
  sns.set()
  loss, val_loss = history.history['loss'], history.history['val_loss']
  epochs = range(1, len(loss) + 1)

  plt.figure(figsize=(10,8))
  plt.plot(epochs, loss , label="Training loss", marker="o")
  plt.plot(epochs, val_loss, label="Validation loss", marker="o")
  plt.legend()
  plt.title("Loss of trainig and validation")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.show()
  
make_accuracy_plot(history) 
make_loss_lot(history)
