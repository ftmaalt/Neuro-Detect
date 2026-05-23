import random
import shutil
from pathlib import Path

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Datasets split
def split_dataset(src_dir, out_dir, train_ratio=0.70, val_ratio=0.15, seed=42):

    src_dir = Path(src_dir)
    out_dir = Path(out_dir)

    if (out_dir / "train").exists():
        print("Dataset already split — skipping.")
        return {
            "train": str(out_dir / "train"),
            "val":   str(out_dir / "val"),
            "test":  str(out_dir / "test"),
        }

    print("Creating dataset split...")
    random.seed(seed)

    for class_folder in sorted(src_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        images = list(class_folder.glob("*.*"))
        random.shuffle(images)

        total = len(images)
        n_train = int(total * train_ratio)
        n_val   = int(total * val_ratio)

        splits = {
            "train": images[:n_train],
            "val":   images[n_train : n_train + n_val],
            "test":  images[n_train + n_val :],
        }

        for split_name, files in splits.items():
            dest = out_dir / split_name / class_folder.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)

        print(
            f"  {class_folder.name}: "
            f"train={len(splits['train'])}  "
            f"val={len(splits['val'])}  "
            f"test={len(splits['test'])}"
        )

    print("Split done ✔️")
    return {
        "train": str(out_dir / "train"),
        "val":   str(out_dir / "val"),
        "test":  str(out_dir / "test"),
    }


    # img data generators for training
def create_data_generators(train_dir, val_dir, test_dir, img_size, batch_size):

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=8,
        width_shift_range=0.04,
        height_shift_range=0.04,
        zoom_range=0.08,
        horizontal_flip=True,
        brightness_range=[0.95, 1.05],
        fill_mode='nearest',
    )

    # Eval generators: no augmentation, only EfficientNet preprocessing.
    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
    )

    val_generator = eval_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )

    test_generator = eval_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


# util
def get_class_map(generator):
    """Return a dict mapping integer class index → class name."""
    return {v: k for k, v in generator.class_indices.items()}