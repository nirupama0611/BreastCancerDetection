from google.colab import drive
drive.mount('/content/drive')

!pip install keras-preprocessing

import cv2
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPool2D, Activation, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_directory = '/content/drive/My Drive/mammography_images/train/'
test_directory = '/content/drive/My Drive/mammography_images/test/'

df = pd.read_csv('/content/drive/My Drive/mammography_images/Training_set.csv')
df.head()

img = cv2.imread('/content/drive/My Drive/mammography_images/train/Image_1.jpg')
plt.imshow(img)

img.shape

train_datagen = ImageDataGenerator(
rescale=1./255,
validation_split=0.15,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True
)

df['label'] = df['label'].astype(str)
train_generator = train_datagen.flow_from_dataframe(
df,
directory = train_directory,
subset = 'training',
x_col = 'filename',
y_col = 'label',
target_size = (224,224),
class_mode = 'categorical'
)
val_generator = train_datagen.flow_from_dataframe(
df,
directory = train_directory,
subset = 'validation',
x_col = 'filename',
y_col = 'label',
target_size = (224,224),
class_mode = 'categorical'
)

model = Sequential()
model.add(Conv2D(32, (3,3) ,activation = 'relu', input_shape = (224,224,3)))
model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation = 'softmax'))

model.compile(
loss = 'categorical_crossentropy',
optimizer = Adam(),
metrics = ['accuracy']
)

history = model.fit(
train_generator,
steps_per_epoch = 150,
epochs = 25,
validation_data = val_generator,
validation_steps = 26
)

!pip install natsort

from natsort import natsorted

ids = []
X_test = []
for image in natsorted(os.listdir(test_directory)):
  ids.append(image.split('.')[0])
  path = os.path.join(test_directory, image)
  X_test.append(cv2.imread(path))
  print(path)

X_test = np.array(X_test)
X_test = X_test.astype('float32') / 255
predictions = model.predict(X_test)

label_map = (train_generator.class_indices)
print(label_map)
print(np.argmax(predictions[1907],axis=-1))

submission = pd.read_csv('/content/drive/My Drive/mammography_images/Testing_set.csv')
submission['label'] = ''
print(submission.columns)
index=0
for i in predictions:
  class_int = np.argmax(i,axis=-1)
  if (class_int==0):
    submission['label'][index]="Density1Benign"
  elif (class_int==1):
    submission['label'][index]="Density1Malignant"
  elif (class_int==2):
    submission['label'][index]="Density2Benign"
  elif (class_int==3):
    submission['label'][index]="Density2Malignant"
  elif (class_int==4):
    submission['label'][index]="Density3Benign"
  elif (class_int==5):
    submission['label'][index]="Density3Malignant"
  elif (class_int==6):
    submission['label'][index]="Density4Benign"
  elif (class_int==7):
    submission['label'][index]="Density4Malignant"
  index=index+1

submission.to_csv('/content/drive/My Drive/mammography_images/Testing_set.csv', index = False)

df = pd.read_csv('/content/drive/My Drive/mammography_images/Testing_set.csv')
df.head()

