"""
Contains function to create datasets
"""
import re
import string
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Embedding, TextVectorization
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def get_lines(filename: str):
  """
  A function to read the lines of text to a list

  Args:
    filename: str - directory of data

  Returns:
    A list of sentenses

  Usage example:
    lines = get_lines('data_dir/text.txt')
  """
  with open(filename, 'r') as f:
      
    return f.readlines()

def preprocess_text_with_line_numbers(filename: str):
  """
  Separate a text into list of abstracts and create a dictionary to each abstract with id abstract, 
  label of sentense, sentetnse, line number of sentence and total lines in abstract.

   in the form:{'id': [...],
                'target': [...],
                'text': [...],
                'line_number': [...],
                'total_lines': [...]
                }

  Args:
    filename: str - directory of data

  Returns:
    A DataFrame with columns 'id' 'target' 'text' 'line_number' 'total_lines'

  Usage example:
    dataframe = preprocess_text_with_line_numbers(filename='data_dir/text.txt')
  """
  input_lines = get_lines(filename)
  abstract_lines = ""
  abstract_samples = []

  for line in input_lines:
    if line.startswith('###'): 
      abstract_id = line
      abstract_lines = ""
    elif line.isspace():  
      abstract_line_split = abstract_lines.splitlines()

      for line_number, abstract_line in enumerate(abstract_line_split):
        line_data = {}
        target, text = abstract_line.split('\t')
        line_data['id'] = re.search('\d+', abstract_id).group(0)
        line_data['target'] = target
        line_data['text'] = text.lower()
        line_data['line_number'] = line_number 
        line_data['total_lines'] = len(abstract_line_split) - 1
        abstract_samples.append(line_data)

    else:
      abstract_lines += line

  return  pd.DataFrame(abstract_samples)

def split_char(text):
  """
  Split text of sentense into list of characters
  """
  return " ".join(list(text))

def create_label_encoder(filename:str, classes:bool):
  """
  Create label encoder and makes list of class names if it's nessesary

  Args:
    filename - A directory of text file with data
    classes - bool, True - makes list of class names, False - doesn't make
  
  Returns:
   Labels encoded by integers (labels_le) or a tuple of labels encoded and class names (labels_le, class_names)

   Usage example:
    labels_le, class_names = create_label_encoder(filename=train_dir, classes=True)
  """
  df = preprocess_text_with_line_numbers(filename)
  label_enc = LabelEncoder()
  labels_le = label_enc.fit_transform(df.target.to_numpy())
  if classes:
    class_names = label_enc.classes_
    return labels_le, class_names
  else:
    pass
    return labels_le

def get_data_ready(filename:str):

  """
  Get data ready to create dataset. 
  Use One Hot Encoder to encode labels, number of sentence's posinion
  
  Args:
    filename - Directory of text file with data
  Return:
    A tuple of sentence's text, OHE labels, OHE position number of sentence,
    OHE total amount of sentences in target abstract
    (sent, labels_ohe, line_number_ohe, total_lines_ohe)
  Usage example:
    sent, labels_ohe, line_number_ohe, total_lines_ohe = get_data_ready(filename=train_dir)

  """
  ohe = OneHotEncoder(sparse=False)
  import tensorflow as tf
  df = preprocess_text_with_line_numbers(filename)
  sent = df.text.tolist()
  labels_ohe = ohe.fit_transform(df.target.to_numpy().reshape(-1, 1))
  line_number_ohe = tf.one_hot(df.line_number.to_numpy(), depth=15)
  total_lines_ohe = tf.one_hot(df.total_lines.to_numpy(), depth=20)

  return sent, labels_ohe, line_number_ohe, total_lines_ohe
 

def create_dataset(filename:str,
                   batch_size:int):
  """
  Create a Prefetch dataset with position embeddings,
  token embeddings and character embeddings

  Args:
    filename - Directory of text file with data
    batch_size - Number of batches per epoch

  Return:
    Prefetch dataset

  Usage example:

  dataset = create_dataset(directory, 32)

  """
  print(tf.__version__)
  sent, labels_ohe, line_number_ohe, total_lines_ohe = get_data_ready(filename)
  chars = [split_char(text) for text in sent]

  token_char_lines_data = tf.data.Dataset.from_tensor_slices((line_number_ohe, total_lines_ohe, sent, chars))
  token_char_lines_labels = tf.data.Dataset.from_tensor_slices(labels_ohe)
  token_char_lines_ds = tf.data.Dataset.zip((token_char_lines_data, token_char_lines_labels))
  return token_char_lines_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def char_vectorize_embed(output_char_len: int,
                         train_dir: str):
  """
  Create character vectorizer and character embeddings

  Args:
    output_char_len: int - A length of output sequences 
    train_dir: - A directory of text file with train data

  Returns:
    A tuple (char_vectorizer, char_embed)

  Usage example:
    char_vectorizer, char_embed = char_vectorize_embed(output_char_len=290,
                                                       train_dir=train_dir)
  """
  df = preprocess_text_with_line_numbers(train_dir)
  sent = df.text.tolist()
  train_char = [split_char(text) for text in sent]
  alphabet = string.ascii_lowercase + string.digits + string.punctuation
  # Create char-level token vectorizer
  num_char = len(alphabet) + 2 # for space and [UNK]
  char_vectorizer = TextVectorization(max_tokens=num_char,
                                      output_sequence_length=output_char_len,
                                      pad_to_max_tokens=True)
  char_vectorizer.adapt(train_char)
  char_vocab = char_vectorizer.get_vocabulary()

  # Create Embeding layer
  char_embed = Embedding(input_dim=len(char_vocab),
                        output_dim = 24,
                        mask_zero=True)
  return char_vectorizer, char_embed
