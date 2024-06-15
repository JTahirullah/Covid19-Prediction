from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np  

# Model Architecture
Classifier = Sequential()
Classifier.add(Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 3)))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))
Classifier.add(Conv2D(32, (3, 3), activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2, 2)))
Classifier.add(Flatten())
Classifier.add(Dense(units=128, activation='relu'))
Classifier.add(Dropout(0.5))  # Adding Dropout for regularization
Classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
optimizer = Adam(learning_rate=0.001)  # Use learning_rate instead of lr
Classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
training_set = train_datagen.flow_from_directory(r'E:\BTECH\Projects\COVID\Covid19-dataset\train',
                                                 target_size=(64, 64), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory(r'E:\BTECH\Projects\COVID\Covid19-dataset\test',
                                            target_size=(64, 64), batch_size=32, class_mode='binary')

# Training the model
history = Classifier.fit(training_set, steps_per_epoch=len(training_set),
                         epochs=20, validation_data=test_set, validation_steps=len(test_set),
                         callbacks=[EarlyStopping(monitor='val_loss', patience=3),
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2)])

# Evaluating the model
loss, accuracy = Classifier.evaluate(test_set)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Making predictions
test_image = image.load_img(r'E:\BTECH\Projects\COVID\TEST\Test (5).jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = Classifier.predict(test_image)

# Displaying Prediction
if result[0][0] == 1:
    prediction = 'COVID - Negative'
    print(prediction)
else:
    prediction = 'COVID - POSITIVE'
    print(prediction)
