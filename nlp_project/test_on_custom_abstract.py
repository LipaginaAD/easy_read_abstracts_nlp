ABSTRACT_NUM = 0
import argparse
import pandas as pd
import tensorflow as tf
import custom_abstract_preprocess, preprocess_data

TRAIN_DIR = '/content/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/train.txt'

parser = argparse.ArgumentParser()
parser.add_argument("--custom_abstract_dir", type=str, help='directory of text file with custom abstracts')
parser.add_argument("--abstract_num", type=int, default=ABSTRACT_NUM, help='number of custom abstract, default=0')
parser.add_argument("--saved_model_dir", type=str, help='directory of saved_model')
parser.add_argument("--train_dir", type=str, help='directory of train data text file', default=TRAIN_DIR)

args = parser.parse_args()

print(f'Preprocess custom abstract number {args.abstract_num} from file {args.custom_abstract_dir}')
# Split abstracts
abstracts = custom_abstract_preprocess.split_text_file_to_abstracts(args.custom_abstract_dir)

# Choose the abstract and preprocess it
chosen_abstr = abstracts[args.abstract_num]
abstract = pd.DataFrame(custom_abstract_preprocess.preprocess_abstract(chosen_abstr))

test_line_number_ohe = tf.one_hot(abstract.line_num.to_numpy(), depth=15)
test_total_lines_ohe = tf.one_hot(abstract.total_lines.to_numpy(), depth=20)
test_sent = abstract.text.tolist()
test_char = [preprocess_data.split_char(text) for text in test_sent]

# Load the model
print(f'Load the model from {args.saved_model_dir}')
loaded_model = tf.keras.models.load_model(args.saved_model_dir)

# Get predictions
custom_prob = loaded_model.predict(x=(test_line_number_ohe,
                                      test_total_lines_ohe,
                                      tf.constant(test_sent),
                                      tf.constant(test_char)))
custom_pred = tf.argmax(custom_prob, axis=1)

# Setup class names
train_labels_le, class_names = preprocess_data.create_label_encoder(filename=args.train_dir, classes=True)

# Visualize models' predictions
print("Visualize model's predictions:")
for i, pred in enumerate(custom_pred):
  label = class_names[pred]
  sent = test_sent[i]
  print(f'{label.upper()} :\n {sent}\n')
