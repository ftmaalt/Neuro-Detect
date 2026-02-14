import os
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image(path):
    """Load an image from a specified path."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {path}")
    return image

def resize_image(image, width, height):
    """Resize the image to a specified width and height."""
    return cv2.resize(image, (width, height))

def normalize_image(image):
    """Normalize the image data to the range [0, 1]."""
    scaler = MinMaxScaler()
    image = image.astype(np.float32)
    image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return image

def remove_noise(image):
    """Remove noise from the image using Gaussian Blur."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def enhance_contrast(image):
    """Enhance the contrast of the image."""
    return cv2.convertScaleAbs(image, alpha=1.5, beta=0)

def create_data_generators(train_dir, test_dir, img_size, batch_size):
    """Create image data generators for training and testing."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator

def preprocess_image(path, width, height):
    """Load, preprocess, and augment the image for single image use."""
    image = load_image(path)
    image = resize_image(image, width, height)
    image = normalize_image(image)
    image = remove_noise(image)
    image = enhance_contrast(image)
    return image

def get_class_map(generator):
    """Get a dictionary mapping classes to human-readable names."""
    return {v: k for k, v in generator.class_indices.items()}