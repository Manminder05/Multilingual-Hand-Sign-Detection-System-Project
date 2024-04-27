import numpy as np
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator

# Load preprocessed dataset
X_train_isl = np.load('X_train_isl.npy')
X_test_isl = np.load('X_test_isl.npy')
y_train_isl = np.load('y_train_isl.npy')
y_test_isl = np.load('y_test_isl.npy')

# Define data generator for data augmentation
datagen_isl = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define MobileNetV2 model with additional layers
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(400, 400, 3))
model_isl = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')
])

# Unfreeze some layers of the base model for fine-tuning
for layer in base_model.layers[:100]:
    layer.trainable = False

# Compile the model
optimizer = Adam(lr=0.0001)  # Reduce the learning rate
model_isl.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('model_isl.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

# Train the model with data augmentation
model_isl.fit(datagen_isl.flow(X_train_isl, y_train_isl, batch_size=32),
              epochs=30,  # Increase the number of epochs
              validation_data=(X_test_isl, y_test_isl),
              callbacks=[checkpoint, reduce_lr])

# Save the trained model
model_isl.save('model_isl.h5')

# Evaluate the model
y_pred_isl = np.argmax(model_isl.predict(X_test_isl), axis=1)
print("ISL Classification Report:")
print(classification_report(np.argmax(y_test_isl, axis=1), y_pred_isl))

