import tensorflow as tf
tf.executing_eagerly()
#tf.enable_eager_execution()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)

# Load training data
train_generator = train_datagen.flow_from_directory('Data', target_size=(28,28), batch_size=1, class_mode='categorical')

# Load validation data
validation_generator = train_datagen.flow_from_directory('Data', target_size=(28,28), batch_size=1, class_mode='categorical')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras import optimizers
# CNN model
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00001), metrics=['accuracy'])
# training the model
batch_size = 1
model.fit_generator(train_generator, validation_data = validation_generator, epochs = 10)
model.save("DEV.model")