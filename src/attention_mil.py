import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
)
from simclr import augment


def load_and_augment(img_path):
    # Load and preprocess a single image with SimCLR augmentations
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels=3)
    image = augment(image)
    image = preprocess_input(image)
    return image


def get_images(directory):
    labels = []
    all_images = []

    # One-hot encoding for classes
    mapping = {
        "epithelioid": [1, 0, 0],
        "sarcomatoid": [0, 1, 0],
        "biphasic": [0, 0, 1],
        "ood": [0, 0, 0]  # Out of distribution
    }

    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        class_name = class_dir.split('_')[1].lower()
        if os.path.isdir(class_path):
            for wsi_dir in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
                images = []
                wsi_path = os.path.join(class_path, wsi_dir)
                if os.path.isdir(wsi_path):
                    for img_name in os.listdir(wsi_path):
                        img_path = os.path.join(wsi_path, img_name)
                        images.append(img_path)
                all_images.append(images)
                labels.append(mapping[class_name])

    return all_images, labels


def extract_features(patches_dir, backbone_model, batch_size):
    all_features = []
    wsi_list, labels = get_images(patches_dir)
    strategy = tf.distribute.MirroredStrategy()  # Use multiple GPUs if available

    with strategy.scope():
        @tf.function
        def extract_step(batch):
            # Forward pass through frozen backbone
            features = backbone_model(batch, training=False)
            return features

        for wsi_images in tqdm(wsi_list, desc="Extracting features"):
            features_list = []
            path_ds = tf.data.Dataset.from_tensor_slices(wsi_images)
            image_ds = path_ds.map(load_and_augment, num_parallel_calls=tf.data.AUTOTUNE)
            image_ds = image_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            dist_ds = strategy.experimental_distribute_dataset(image_ds)

            for dist_batch in dist_ds:
                per_replica_features = strategy.run(extract_step, args=(dist_batch,))
                batch_features = tf.concat(strategy.gather(per_replica_features, axis=0), axis=0)
                features_list.extend(batch_features.numpy())

            all_features.append(np.array(features_list))

    return np.array(all_features, dtype=object), np.array(labels, dtype=np.float32)


class MultiHeadAttentionMIL(tf.keras.Model):
    def __init__(self, num_heads=4, attention_dim=128, projection_dim=128, num_classes=3, dropout_rate=0.2, **kwargs):
        super(MultiHeadAttentionMIL, self).__init__(**kwargs)
        # Attention + projection + classifier
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=attention_dim)
        self.norm = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.projection = tf.keras.layers.Dense(projection_dim, activation='relu')
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        # Forward pass with residual connection and average pooling
        attn_output = self.mha(x, x, training=training)
        x_residual = x + attn_output
        x_norm = self.norm(x_residual)
        x_norm = self.dropout(x_norm, training=training)
        pooled = tf.reduce_mean(x_norm, axis=1)
        proj = self.projection(pooled)
        return self.classifier(proj)


def generate_dataset(features, labels, num_classes=3, batch_size=1, seed=10):
    # Build dataset from extracted features and labels
    def generator():
        for x, y in zip(features, labels):
            if x.shape[0] > 0:
                yield x, y

    output_signature = (
        tf.TensorSpec(shape=(None, features[0].shape[-1]), dtype=tf.float32),
        tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    dataset = dataset.shuffle(buffer_size=len(labels), seed=seed).batch(batch_size)
    return dataset


def crossfolding(features, labels, validation_split=0.2):
    # Simple random split into training and validation sets
    n_samples = len(features)
    n_val = int(n_samples * validation_split)
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    train_features = [features[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_features = [features[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    train_dataset = generate_dataset(train_features, train_labels)
    val_dataset = generate_dataset(val_features, val_labels)
    return train_dataset, val_dataset


def train_attention_mil(patches_dir, backbone_model_dir=None, num_epochs=20, initial_lr=1e-4):
    # Load or initialize backbone
    if backbone_model_dir is None:
        backbone = ResNet50(include_top=False, weights='imagenet', pooling="avg")
    else:
        backbone = tf.keras.models.load_model(backbone_model_dir)
    backbone.trainable = False  # Freeze the backbone model

    model = MultiHeadAttentionMIL(num_classes=3)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-5)

    # Compile model with metrics
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            'accuracy', 
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.F1Score(name='f1_score')
        ]
    )

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_attention_mil.keras',
        save_best_only=True,
        monitor='loss',
        mode='min',
    )

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Extract new augmented features at each epoch
        features, labels = extract_features(patches_dir, backbone, batch_size=256)
        train_ds, val_ds = crossfolding(features, labels)

        # Train for one epoch
        model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=[checkpoint_callback])
