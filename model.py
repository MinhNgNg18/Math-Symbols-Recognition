import tensorflow as tf
import pathlib
from collections import Counter


def load_data():
    target_dir = 'dataset'
    data_dir = pathlib.Path(target_dir)
    class_names = list(sorted([item.name for item in data_dir.glob("*")]))

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                              rotation_range=0.2,
                                                              shear_range=0.2,
                                                              zoom_range=0.2,
                                                              width_shift_range=0.2,
                                                              height_shift_range=0.3,
                                                              brightness_range=[0.7, 1.3],
                                                              horizontal_flip=True,
                                                              validation_split=0.2
                                                              )
    train_data = datagen.flow_from_directory(directory=target_dir,
                                             batch_size=32,
                                             target_size=(28, 28),
                                             color_mode='grayscale',
                                             class_mode='categorical',
                                             seed=42,
                                             shuffle=True,
                                             subset='training')

    val_data = datagen.flow_from_directory(
        directory=target_dir,
        batch_size=32,
        target_size=(28, 28),
        color_mode='grayscale',
        class_mode='categorical',
        seed=42,
        shuffle=True,
        subset='validation'
    )

    class_labels = train_data.class_indices
    train_labels = train_data.labels
    train_class_distribution = dict(Counter(train_labels))

    train_class_counts = {class_name: train_class_distribution[idx] for class_name, idx in class_labels.items()}
    print("Number of images in each class (training):", train_class_counts)

    val_labels = val_data.labels
    val_class_distribution = dict(Counter(val_labels))

    val_class_counts = {class_name: val_class_distribution[idx] for class_name, idx in class_labels.items()}
    print("Number of images in each class (validation):", val_class_counts)
    return train_data, val_data, class_names


def build_model(train_data, val_data, class_names):
    model = tf.keras.Sequential()

    # First Conv Block
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Second Conv Block
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Third Conv Block
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    # Flatten the output of the Conv layers
    model.add(tf.keras.layers.Flatten())

    # Fully connected layer
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    # Output layer
    model.add(tf.keras.layers.Dense(len(class_names), activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    model.fit(train_data,
              epochs=35,
              steps_per_epoch=len(train_data),
              validation_data=val_data,
              callbacks=[reduce_lr])
    return model


def main():
    train_data, val_data, class_names = load_data()
    model = build_model(train_data, val_data, class_names)
    model.save('model.h5')


if __name__ == '__main__':
    main()
