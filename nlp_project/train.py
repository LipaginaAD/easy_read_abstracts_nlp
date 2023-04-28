import tensorflow as tf
import utils, preprocess_data, engine
NUM_EPOCHS = 6
BATCH_SIZE = 32
CHECK_PATH = 'nlp_project_check/checkpoint'
MODEL_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'
OUTPUT_SHAPE_EMBED_LAYER = 512
OUTPUT_CHAR_LEN = 290
SAVED_MODEL_DIR = 'nlp_model'

import argparse

my_parser = argparse.ArgumentParser()
my_parser.add_argument("--train_dir", help='directory of text file with train data')
my_parser.add_argument("--valid_dir", help='directory of text file with valid data')
my_parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help='num of epochs, default=5')
my_parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help='batch size, default=32')
my_parser.add_argument("--check_path", type=str, default=CHECK_PATH, help="directory to save checkpoint data, default=CHECK_PATH")
my_parser.add_argument("--model_url", type=str, default=MODEL_URL, help="Model's URL from TensorFlow Hub for transfer learning, default=MODEL_URL")
my_parser.add_argument("--output_shape_embed_layer", type=int, default=OUTPUT_SHAPE_EMBED_LAYER, help="output shape of embed layer, depends from using transfer learning mode, default=512")
my_parser.add_argument("--output_char_len", type=str, default=OUTPUT_CHAR_LEN, help="Length of output sequences from character vectorizer layer, default=290")
my_parser.add_argument("--saved_model_dir", type=str, default=SAVED_MODEL_DIR, help='directory to save model')

args = my_parser.parse_args()


# Create datasets
print("Create prefetch datasets...")
train_ds = preprocess_data.create_dataset(filename=args.train_dir, batch_size=args.batch_size)
val_ds = preprocess_data.create_dataset(filename=args.valid_dir, batch_size=args.batch_size)

# Create Callbacks
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.check_path,
                                                        save_weights_only=True,
                                                        save_best_only=True,
                                                        save_freq='epoch',
                                                        verbose=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience = 3)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",  
                                                factor=0.2,
                                                patience=2,
                                                verbose=1,
                                                min_lr=1e-5)

# Create embed layer of model using transfer learning 
embed_layer_tr_learning = utils.embed_layer_transfer_learning(use_url=args.model_url,
                                                        output_shape=args.output_shape_embed_layer)
# Create character vectorizer and character embedding layers
char_vectorizer, char_embed = preprocess_data.char_vectorize_embed(output_char_len=args.output_char_len,
                                                   train_dir=args.train_dir)


train_labels_le, class_names = preprocess_data.create_label_encoder(filename=args.train_dir, classes=True)
val_labels_le = preprocess_data.create_label_encoder(filename=args.valid_dir, classes=False)


# Create the model
print("Create the model...")
model = engine.create_model(embed_layer_tr_learning, char_vectorizer, char_embed, len(class_names))

# Compile the model
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
              optimizer = tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit the model
print("Train our model...")
history_model = model.fit(train_ds,
                          steps_per_epoch=len(train_ds), 
                          epochs=args.num_epochs,
                          validation_data=val_ds, 
                          validation_steps=int(0.1 * len(val_ds)),
                          callbacks=[checkpoint_callback, early_stopping, reduce_lr])

# Give the model the best weights from checkpoint path
model.load_weights(args.check_path)

print(f"Save model to {args.saved_model_dir}")
model.save(args.saved_model_dir)

# Calculate results
print("Calculate model's predictions...")
model_prob = model.predict(val_ds)
model_pred = tf.argmax(model_prob, axis=1)

model_metrics = utils.calculate_results(y_true=val_labels_le,
                                        y_pred=model_pred)

print("Print model's metrics after training")
print(model_metrics)
