"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import urllib

from tensorflow.python.platform import gfile

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


"""
    Basic Tokenizer: Splits the given sentence into list of words.
"""
def basic_tokenizer(sentence):
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
  return [w for w in words if w]

'''
    Creates a vocabulary from the path.
'''
def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1      
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      print('>> Full Vocabulary Size :',len(vocab_list))
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")

"""
    Initializes vocabulary from the given path.
"""
def initialize_vocabulary(vocabulary_path):

  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

"""
    Converts given sentences into token ids.
    If no tokenizer provided, we will be using the basic tokenizer.
"""
def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]

"""
    Converts the data into token ids.
    First we initialize the vocabulary.
    Then we process the tokenization for each sentence in the data files.
"""
def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):

  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                             normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


"""
    Initially the vocabulary will not be exist. 
    So it creates a vocabulary of given size.
    Then it tokenizes the training and testing datasets 
    and returns the path.
"""
def wrangle_data(wranglig_directory, train_enc, train_dec, test_enc, test_dec, enc_vocabulary_size, dec_vocabulary_size, tokenizer=None):

    #Create vocabulary if not exist
    encode_vocab_path = os.path.join(wranglig_directory, "vocab%d.enc" % enc_vocabulary_size)
    decode_vocab_path = os.path.join(wranglig_directory, "vocab%d.dec" % dec_vocabulary_size)
    create_vocabulary(encode_vocab_path, train_enc, enc_vocabulary_size, tokenizer)
    create_vocabulary(decode_vocab_path, train_dec, dec_vocabulary_size, tokenizer)

    #Generate the token ids for the training set.
    #Saves tokenized data into data wrangling directory
    encode_train_ids_path = os.path.join(wranglig_directory, "train.enc.ids%d" % enc_vocabulary_size)
    decode_train_ids_path = os.path.join(wranglig_directory, "train.dec.ids%d" % dec_vocabulary_size)
    data_to_token_ids(train_enc, encode_train_ids_path, encode_vocab_path, tokenizer)
    data_to_token_ids(train_dec, decode_train_ids_path, decode_vocab_path, tokenizer)
    
    #Generate the token ids for the training set.
    #Saves tokenized data into data wrangling directory
    encode_test_ids_path = os.path.join(wranglig_directory, "test.enc.ids%d" % enc_vocabulary_size)
    decode_test_ids_path = os.path.join(wranglig_directory, "test.dec.ids%d" % dec_vocabulary_size)
    data_to_token_ids(test_enc, encode_test_ids_path, encode_vocab_path, tokenizer)
    data_to_token_ids(test_dec, decode_test_ids_path, decode_vocab_path, tokenizer)

    return (encode_train_ids_path, decode_train_ids_path, encode_test_ids_path, decode_test_ids_path, encode_vocab_path, decode_vocab_path)
