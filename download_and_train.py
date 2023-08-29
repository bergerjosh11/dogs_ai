import os
import urllib.request
import tarfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Download and extract the Stanford Dogs Dataset
dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
dataset_path = "dogs_dataset.tar"
extract_path = "dogs_dataset"
if not os.path.exists(extract_path):
    urllib.request.urlretrieve(dataset_url, dataset_path)
    with tarfile.open(dataset_path, "r") as tar:
        tar.extractall()

# Define paths and parameters
train_data_dir = "train_data_directory"
batch_size = 32
epochs = 10
image_size = (224, 224)
num_classes = len(os.listdir(extract_path))

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Load the MobileNetV2 model
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)

# Add a global spatial average pooling layer and a fully connected layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)

# Combine the base model and custom layers to create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Prepare data generator
train_generator = train_datagen.flow_from_directory(
    extract_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
)

# Save the trained model
model.save("dog_breed_recognition_model.h5")
