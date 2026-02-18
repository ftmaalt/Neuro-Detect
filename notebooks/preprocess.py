"since moved to notebooks incase fails running-might remove it"
import sys
sys.path.append(".")

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(train_dir, test_dir, img_size, batch_size):
    """Create image data generators for training and testing with augmentation."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, test_generator

def get_class_map(generator):
    """Get a dictionary mapping classes to human-readable names."""
    return {v: k for k, v in generator.class_indices.items()}