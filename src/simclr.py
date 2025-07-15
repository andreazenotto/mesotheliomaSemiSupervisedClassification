import os
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    preprocess_input,
)

# Add Gaussian noise to the image
def add_gaussian_noise(image, mean=0.0, stddev=10.0):
    image = tf.cast(image, tf.float32)  # keep values in [0, 255]
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev, dtype=tf.float32)
    noisy_image = image + noise
    noisy_image = tf.clip_by_value(noisy_image, 0.0, 255.0)
    return noisy_image

# Apply a series of random augmentations
def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.resize(image, [256, 256])
    image = tf.image.random_crop(image, size=[224, 224, 3])
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = add_gaussian_noise(image)
    return image

# Read and preprocess an image into two augmented views
def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    view1 = augment(image)
    view2 = augment(image)
    view1 = preprocess_input(view1)
    view2 = preprocess_input(view2)
    return view1, view2

# Shuffle, batch and prefetch the dataset
def shuffle_and_batch(dataset, batch_size):
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create a dataset of (view1, view2) pairs from a directory of images
def create_dataset(directory):
    all_images = []

    for class_dir in os.listdir(directory):
        class_path = os.path.join(directory, class_dir)
        if os.path.isdir(class_path):
            for wsi_dir in os.listdir(class_path):
                wsi_path = os.path.join(class_path, wsi_dir)
                if os.path.isdir(wsi_path):
                    for img_name in os.listdir(wsi_path):
                        img_path = os.path.join(wsi_path, img_name)
                        all_images.append(img_path)

    path_ds = tf.data.Dataset.from_tensor_slices(all_images)
    image_ds = path_ds.map(lambda x: process_image(x), num_parallel_calls=tf.data.AUTOTUNE)
    return image_ds

# Build the encoder + projection head model
def build_model():
    base_model = ResNet50(weights='imagenet', include_top=False)
    inputs = base_model.input
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2048, activation='relu')(x)
    outputs = tf.keras.layers.Dense(128)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model

# Custom training loop for SimCLR
class SimCLRTrainer(tf.keras.Model):
    def __init__(self, encoder, temperature):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature

    def call(self, inputs, training=False):
        return self.encoder(inputs, training=training)

    def compile(self, optimizer):
        super().compile()
        self.optimizer = optimizer

    def train_step(self, data):
        view1, view2 = data

        with tf.GradientTape() as tape:
            proj1 = self.encoder(view1, training=True)
            proj2 = self.encoder(view2, training=True)
            loss = nt_xent_loss(proj1, proj2, self.temperature)

        grads = tape.gradient(loss, self.encoder.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))
        return {"loss": loss}

# Normalized temperature-scaled cross-entropy loss (NT-Xent)
def nt_xent_loss(proj_1, proj_2, temperature):
    batch_size = tf.shape(proj_1)[0]
    proj_1 = tf.math.l2_normalize(proj_1, axis=1)
    proj_2 = tf.math.l2_normalize(proj_2, axis=1)
    projections = tf.concat([proj_1, proj_2], axis=0)
    similarity_matrix = tf.matmul(projections, projections, transpose_b=True)
    logits = similarity_matrix / temperature

    # Remove self-similarity from logits
    mask = tf.eye(2 * batch_size)
    logits = logits * (1. - mask) - 1e9 * mask

    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)

# Train the SimCLR model across multiple epochs (supports resume)
def train_simclr(dataset_dir, start_epoch = 0, end_epoch = 20, total_epochs = 40, batch_size=128, temperature=0.5, lr=2e-4, lr_decay=True):
    strategy = tf.distribute.MirroredStrategy()
    dataset = create_dataset(dataset_dir)
    dataset = shuffle_and_batch(dataset, batch_size)
    model_path = f'simclr_model_epoch{start_epoch}.weights.h5'

    def lr_scheduler(epoch):
        factor = pow((1 - (epoch / total_epochs)), 0.9)
        return lr * factor
    lr = lr_scheduler(start_epoch) if lr_decay else lr

    with strategy.scope():
        backbone = build_model()
        simclr_model = SimCLRTrainer(backbone, temperature)

        if start_epoch > 0:
            print(f"Resuming from epoch {start_epoch} with LR={lr:.6f}")
            simclr_model.build(input_shape=(None, 224, 224, 3))
            simclr_model.load_weights(model_path)

        optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-5)
        simclr_model.compile(optimizer=optimizer)
        dist_dataset = strategy.experimental_distribute_dataset(dataset)

        # Save best model based on loss
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_simclr_model.weights.h5',
            save_best_only=True,
            monitor='loss',
            mode='min',
            save_weights_only=True,
        )

        callbacks = [checkpoint_callback]
        if lr_decay:
            callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_scheduler))

        simclr_model.fit(dist_dataset, initial_epoch=start_epoch, epochs=end_epoch, callbacks=[checkpoint_callback])

    if end_epoch == total_epochs:
        print("Saving the best backbone weights...")
        simclr_model.build(input_shape=(None, 224, 224, 3))
        simclr_model.load_weights('best_simclr_model.weights.h5')
        new_model = tf.keras.models.Model(inputs=simclr_model.encoder.inputs, outputs=simclr_model.encoder.layers[-3].output)
        new_model.save("best_backbone.h5")
    else:
        print("Training completed. Saving the final model weights...")
        simclr_model.save_weights(f'simclr_model_epoch{end_epoch}.weights.h5')
