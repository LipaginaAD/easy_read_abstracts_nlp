import tensorflow as tf
from tensorflow.keras import layers
def create_model(embed_layer, 
                 char_vectorizer, 
                 char_embed, 
                 num_classes:int):
  """
  Create an NLP model with token embeddings from transfer learning model,
  character vectorizer, character embeddings and position embeddings

  Args:
    embed_layer - layer creates with model from TensorFlow Hub
    char_vectorizer - Character vectorizer
    char_embed - Character embeddings
    num_classes - Number of classes

  Usage example:
    model = create_model(embed_layer=embed_layer_tr_learning, 
                        char_vectorizer=char_vectorizer, 
                        char_embed=char_embed, 
                        num_classes=len(classes))

  """

  # 1. Setup token inputs/model
  token_inputs = layers.Input(shape=[], dtype='string', name='token_inputs')
  token_embeddings = embed_layer(token_inputs)
  token_outputs = layers.Dense(296, activation='relu')(token_embeddings)
  token_model = tf.keras.Model(token_inputs, token_outputs)

  # 2. Setup char inputs/model
  char_inputs = layers.Input(shape=(1,), dtype='string', name='char_inputs')
  char_vectors = char_vectorizer(char_inputs)
  char_embeddings = char_embed(char_vectors)
  char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
  char_model = tf.keras.Model(char_inputs, char_bi_lstm)

  # 3. line_number feature
  line_number_inputs = layers.Input(shape=(15,), dtype=tf.float32, name='line_number_inputs')
  line_number_outputs = layers.Dense(32, activation='relu')(line_number_inputs)
  line_number_model = tf.keras.Model(line_number_inputs, line_number_outputs)

  # 4. total_line feature
  total_line_inputs = layers.Input(shape=(20,), dtype=tf.float32, name='total_line_inputs')
  total_line_outputs = layers.Dense(32, activation='relu')(total_line_inputs)
  total_line_model = tf.keras.Model(total_line_inputs, total_line_outputs)

  # 5. Concatenate token and char 
  token_char_concat = layers.Concatenate(name='token_char_concat')([token_model.output, char_model.output])
  combined_dropout = layers.Dropout(0.5)(token_char_concat)

  # 6. Bidirectional layer
  combined_reshape = layers.Reshape((1, 344), input_shape=(344,))(combined_dropout)
  token_char_bi_lstm = layers.Bidirectional(layers.LSTM(96))(combined_reshape)

  # 7. Concatenate line_number feature, total_line feature and token_char_concat 
  concat_layer = layers.Concatenate(name='concat_layer')([line_number_model.output, total_line_model.output, token_char_bi_lstm])

  # 8. Create output layers + dropout
  final_dropout = layers.Dropout(0.5)(concat_layer)
  output_layer = layers.Dense(num_classes, activation='softmax')(final_dropout)

  # 9. Create a model
  model = tf.keras.Model(inputs=[line_number_inputs,
                                  total_line_inputs,
                                  token_inputs, 
                                  char_inputs],
                        outputs = output_layer)
  return model
