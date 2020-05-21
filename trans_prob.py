import re

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.layers import common_hparams
from tensor2tensor.utils import registry

# Use register_model for a new T2TModel
# Use register_problem for a new Problem
# Use register_hparams for a new hyperparameter set


@registry.register_hparams
def my_very_own_hparams():
  # Start with the base set
  hp = common_hparams.basic_params1()
  # Modify existing hparams
  hp.num_hidden_layers = 2
  return hp


@registry.register_problem
class TranslitProb(text_problems.Text2TextProblem):
  """Predict next line of poetry from the last line. From Gutenberg texts."""
  
  @property
  def vocab_type(self):
    return "character"
  
  # @property
  # def approx_vocab_size(self):
  #   """Approximate vocab size to generate. Only for VocabType.SUBWORD."""
  #   return 2**8  # ~32k
  
  @property
  def is_generate_per_split(self):
    # generate_data will shard the data into TRAIN and EVAL for us.
    return False

  @property
  def dataset_splits(self):
    """Splits of data to produce and number of output shards for each."""
    # 10% evaluation data
    return [{
        "split": problem.DatasetSplit.TRAIN,
        "shards": 50,
    }, {
        "split": problem.DatasetSplit.EVAL,
        "shards": 1,
    }]

  def generate_samples(self, data_dir, tmp_dir, dataset_split):
    del data_dir
    del tmp_dir
    del dataset_split

    f = open('/content/t2t-translit/train.txt',encoding='utf8')
    en = []
    bn = []
    
    for line in f:
      [b,e]=line.split("\t")
      e = e.strip()
      if len(b.split(" ")) == len(e.split(" ")):
        for i in range(len(b.split(" "))):
          # print(' '.join(list(e.split(" ")[i].strip())),' '.join(list(b.split(" ")[i].strip())))
          yield {
              "inputs": tokenize_english(e.split(" ")[i]),
              "targets": ' '.join(list(b.split(" ")[i].strip())),
          }
  
def tokenize_english(text):
  tokens = []
  vowels = 'aeiou'
  semivowels = 'wy'
  current_token = ''
  for char in text:
    if char is 'h' and (current_token != '' and current_token[-1] not in vowels and current_token[-1] not in semivowels):
      tokens.append(current_token + 'h')
      current_token = ''
    else:
      if current_token != '':
        tokens.append(current_token)
        current_token = char
  if current_token != '':
    tokens.append(current_token)

  return ' '.join(tokens)
