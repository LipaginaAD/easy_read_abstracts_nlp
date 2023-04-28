"""
Preprocess custom abstract to test model
"""
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
    
def split_text_file_to_abstracts(filedir:str):
  """
  Read text file with custom abstracts and split it into abstracts 
  """
  text = get_lines(filedir)
  text = ' '.join(text)
  return text.split('###')

def preprocess_abstract(abstract):
  """
  Create a dictionary to each abstract with id abstract, 
  label of sentense, sentetnse, line number of sentence and total lines in abstract.

   in the form:{'text': ...,
                'line_number': ...,
                'total_lines': ...
                }

  Args:
    abstract - abstract to preprocess

  Returns:
    A dictionary with columns 'text' 'line_number' 'total_lines'

  Usage example:
    abstract = preprocess_abstract(abstract = abstracts[2])

  """
  # Replace digits by @
  remove_digits = str.maketrans('0123456789', '@@@@@@@@@@')
  abstract = abstract.translate(remove_digits)
  abstract = abstract.replace('\n', '')
  
  # Split abctract into sentences
  sentences = abstract.split(".")
  abstract_samples = []
  for num, sent in enumerate(sentences):
    line_data = {}
    line_data['line_num'] = num
    line_data['total_lines'] = len(sentences)
    line_data['text'] = sent.lower()
    abstract_samples.append(line_data)

  return abstract_samples
