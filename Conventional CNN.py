import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict labels for the test dataset

# Set the paths to your dataset directory
train_dir = 'C:/Users/Deklin/Documents/UTS SHIT/Professional A/Dataset/Train'
test_dir = 'C:/Users/Deklin/Documents/UTS SHIT/Professional A/Dataset/Test'

# Define the image generators using the above directories
# and setting the focus area and batch size
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
target_size = (150, 150)  # Adjust the target size to match your model's input size
batch_size = 50



# Data generator construction and classes
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # Assumes a multi-class classification problem
    classes=['Benign', 'Malignant']
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    classes=['Benign', 'Malignant']
)
num_classes = 2  # Two classes for binary classification

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 5, strides=1, padding='same', activation='relu', input_shape=(150, 150, 3)),  # Changed activation
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(64, 5, strides=1, padding='same', activation='relu'),  # Changed activation
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Conv2D(128, 5, strides=1, padding='same', activation='relu'),  # Changed activation
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Use 'softmax' for multi-class classification
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model and save the best epoch
best_epoch = tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=False)
model.fit(
    train_generator,
    epochs=4,
    validation_data=test_generator,
    callbacks=[best_epoch]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)


predicted_labels = model.predict(test_generator)
predicted_labels = np.argmax(predicted_labels, axis=1)  # Convert one-hot encoded labels to class indices

# Get the true labels from the test generator
true_labels = test_generator.classes

# Compute the confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix
labels = ['Benign', 'Malignant']
disp = ConfusionMatrixDisplay(confusion, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')





