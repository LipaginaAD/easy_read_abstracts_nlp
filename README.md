# easy_read_abstracts_nlp
This classification NLP model makes reading medical papers' abstracts easier. It assigns a label (BACKGROUND, OBJECTIVE, METHODS, RESULTS or CONCLUSIONS) to each sentence of abstracts.  

It's replicate this paper: https://arxiv.org/abs/1710.06071

In model used token embeddings, character embeddings and position embeddings.

## Data
Data taken from authors of paper, that replicate: https://github.com/Franck-Dernoncourt/pubmed-rct

## Usage

To train model:

```
!python nlp_project/train.py --train_dir='your_directory/train.txt' --valid_dir='your_directory/dev.txt'
```

You can also change the following settings:


--num_epochs - number of epochs, default=5


--batch_size - Number of samples per batch, default=32

--check_path - directory to save checkpoint data

--model_url - Model's URL from TensorFlow Hub for transfer learning embed layer, USE model as default

--output_shape_embed_layer - output shape of embed layer, depends from using transfer learning mode

--output_char_len - Length of output sequences from character vectorizer layer, default=290

--saved_model_dir - directory to save model


To use model on custom abstracts 

```
!python nlp_project/test_on_custom_abstract.py --custom_abstract_dir='your_directory/test_abstracts.txt' --saved_model_dir='/content/nlp_model'
```

You can also change the following settings:

--abstract_num - number of custom abstract in text file

--train_dir - directory of text file with train data
