
import numpy as np # linear algebra
import pandas as pd # data processing, csv file I/O
import math

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Lambda, Flatten, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib
import seaborn as sns
get_ipython().magic('matplotlib inline')




sns.set(color_codes = True, palette='muted')





Input_file = 'G:/fashion-mnist_train.csv'
df_train = pd.read_csv(Input_file)




print('Training data:', df_train.info())
#print(df_train)




target = df_train['label']
y = to_categorical(target)
y.shape
print(target)




features = df_train.iloc[:,1:]
print(features)
features.shape





X_train, X_test, y_train,y_test = train_test_split(features,y, test_size=0.2, random_state=1)#splitting t data in 2 sets




X_test, X_val, y_test, y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=1)#splitting teh data in 2 sets




X_train.shape



y_train.shape
print(y_train)


X_val.shape



n_features = len(X_train.values[0])
n_pixels =int (math.sqrt(n_features))
n_classes = y_train.shape[1]
print('We\'ve got {} feature rows and {} labels'.format(len(X_train), len(y_train)))
print('Each row has {} features'.format(len(X_train.values[0])))
print('and we have {} classes'.format(n_classes))
assert(len(y_train) == len(X_train))
assert(len(X_train == len(X_test)))
#assert(n_features == n_pixels**2)
print('Input images have {0} x {0} px shape'.format(n_pixels))
print('So far, so good')




X_train = X_train.values.reshape(X_train.shape[0], n_pixels, n_pixels, 1)
X_test = X_test.values.reshape(X_test.shape[0], n_pixels, n_pixels, 1)
X_train, X_valid, y_train, y_valid = train_test_split(*shuffle(X_train, y_train), test_size=0.1)

print('X_train.shape:', X_train.shape)
print('y_train.shape:', y_train.shape)
print('X_valid.shape:', X_val.shape)
print('y_valid.shape:', y_val.shape)
print('X_test.shape:', X_test.shape)



sns.distplot(df_train['label'].values, kde=False, vertical=False, bins=10)





def create_my_model(shape=(28, 28, 1)):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=shape))
    
    model.add(Conv2D(32, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dropout(0.6))
    
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(n_classes, activation='softmax'))
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
    
model = create_my_model(shape=(n_pixels, n_pixels, 1))
model.summary()




X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)

imgen_train = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=False,
    vertical_flip=False
)

imgen_valid = ImageDataGenerator()

imgen_train.fit(X_train)
imgen_valid.fit(X_valid)




EPOCHS = 4 # actually needs to run for much longer to achieve the >99.4% validation accuracy I got
BATCH = 80

history = model.fit_generator(
    imgen_train.flow(*shuffle(X_train, y_train), batch_size=BATCH),
    steps_per_epoch=X_train.shape[0]//(4*BATCH),
    epochs=EPOCHS,
    validation_data=imgen_valid.flow(*shuffle(X_valid, y_valid)),
    validation_steps=BATCH
)




model.save('model.h5')




y_test = model.predict_classes(X_test)
print(y_test.shape)




result = list(enumerate(y_test))
result = pd.DataFrame.from_dict({'ImageId': range(1, len(y_test)+1), 'Label': y_test})

result.to_csv('submission.csv', index=False)

