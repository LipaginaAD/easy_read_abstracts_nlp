from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow_hub as hub

def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.
  
  Return:
    A dictionary {"accuracy": ...,
                  "precision": ...,
                  "recall": ...,
                  "f1": ...}

  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) 
  # Calculate model precision, recall and f1 score using "weighted" average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": round(model_accuracy, 2),
                  "precision": round(model_precision, 2),
                  "recall": round(model_recall, 2),
                  "f1": round(model_f1, 2)}
  return model_results


def embed_layer_transfer_learning(use_url:str,
                                  output_shape:int):
  """
  Take a URL of model from Tensorflow Hub and create a layer

  Args:
    use_url - a URL of model from Tensorflow Hub
    output_shape - output shape of layer, depends from using model

  Returns:
    hub.KerasLayer
  """
  print('Load model from TensorFlow Hub')
  model_embed = hub.load(use_url)
  embed_layer = hub.KerasLayer(use_url,
                                  output_shape=(output_shape),
                                  dtype='string')
  return embed_layer
