
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import datawrangler.data_wrangler as data_wrangler
import model.enc_dec_model as encdec


try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser # because In Python 3, ConfigParser has been renamed to configparser for PEP 8 compliance.
    
config_params = {}

'''
Reads the configuration file from the given path
and returns a dictionary with configuration parameters
'''
def getConfiguration(config_file='setModelParams.rc'):
    parser = SafeConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)

#Refer the report for the details about bucketing
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]

"""
    Uses tensorflow to read the data into buckets 
"""
def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_wrangler.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


"""
    Creates encode decode Model.
"""
def create_model(session,forward_only):

  model = encdec.encDec_Model( config_params['enc_vocab_size'], config_params['dec_vocab_size'], _buckets, config_params['layer_size'], config_params['num_layers'], config_params['max_gradient_norm'], config_params['batch_size'], config_params['learning_rate'], config_params['learning_rate_decay_factor'],forward_only=forward_only)
  if 'pretrained_model' in config_params:
      model.saver.restore(session,config_params['pretrained_model'])
      return model

  ckpt = tf.train.get_checkpoint_state(config_params['checkpoints'])
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Creating model with initial parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  #Data Wrangling --> Tokenizes the data set.
  print("Creating tokenized data in %s" % config_params['wrangling_directory'])
  encode_train_ids_path, decode_train_ids_path, encode_test_ids_path, decode_test_ids_path, _, _ = data_wrangler.wrangle_data(config_params['wrangling_directory'],config_params['train_enc'],config_params['train_dec'],config_params['test_enc'],config_params['test_dec'],config_params['enc_vocab_size'],config_params['dec_vocab_size'])

  #setup config to use BFC allocator
  config = tf.ConfigProto()  
  config.gpu_options.allocator_type = 'BFC'

  with tf.Session(config=config) as sess:
    
    print("Number of Layers :%d, Layer Size:%d" % (config_params['num_layers'], config_params['layer_size']))
    # Create sequence to sequence model.
    model = create_model(sess,False)

    test_set = read_data(encode_test_ids_path, decode_test_ids_path)
    train_set = read_data(encode_train_ids_path, decode_train_ids_path, config_params['max_train_data_size'])
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    
    step_time, loss = 0.0, 0.0
    cur_step = 0
    prev_losses = []
    #Event logs will be stored in tensorboard/logs folder
    writer = tf.train.SummaryWriter("tensorboard/logs",sess.graph)
    #Run the traing loop
    while True:
      #Pick a random bucket between 0 and 1.
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / config_params['steps_per_checkpoint']
      loss += step_loss / config_params['steps_per_checkpoint']
      cur_step += 1

      #Save Checkpoints and print stats.
      if cur_step % config_params['steps_per_checkpoint'] == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("step > %d,  learning rate > %.4f, step-time > %.2f, perplexity > "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        # If no improvement seen in the last two steps, then decrease learning rate.
        if len(prev_losses) > 2 and loss > max(prev_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        prev_losses.append(loss)
        checkpoint_path = os.path.join(config_params['checkpoints'], "encdec.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        for bucket_id in xrange(len(_buckets)):
          if len(test_set[bucket_id]) == 0:
            print("  Empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              test_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("bucket> %d,  perplexity> %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()


def test():
  with tf.Session() as sess:
    # initilazie encode-decode model
    model = create_model(sess,True)
    model.batch_size = 1

    # Initialize vocabularies
    encode_vocab_path = os.path.join(config_params['wrangling_directory'],"vocab%d.enc" % config_params['enc_vocab_size'])
    decode_vocab_path = os.path.join(config_params['wrangling_directory'],"vocab%d.dec" % config_params['dec_vocab_size'])

    encode_vocab, _ = data_wrangler.initialize_vocabulary(encode_vocab_path)
    _, rev_dec_vocab = data_wrangler.initialize_vocabulary(decode_vocab_path)

    # Decode from standard input.
    sys.stdout.write("Query> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token_ids
      token_ids = data_wrangler.sentence_to_token_ids(tf.compat.as_bytes(sentence), encode_vocab)
      # Identify the bucket
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_wrangler.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_wrangler.EOS_ID)]
      #print the output queries
      print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
      print("Query> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()

def init_session(sess, conf='setModelParams.rc'):
    global config_params
    config_params = getConfiguration(conf)
 
    # Create model and load parameters.
    model = create_model(sess,True)
    model.batch_size = 1  # We test one sentence at a time.

    # Load vocabularies.
    encode_vocab_path = os.path.join(config_params['wrangling_directory'],"vocab%d.enc" % config_params['enc_vocab_size'])
    decode_vocab_path = os.path.join(config_params['wrangling_directory'],"vocab%d.dec" % config_params['dec_vocab_size'])

    encode_vocab, _ = data_wrangler.initialize_vocabulary(encode_vocab_path)
    _, rev_decode_vocab = data_wrangler.initialize_vocabulary(decode_vocab_path)

    return sess, model, encode_vocab, rev_decode_vocab

"""
    Main Function: Initializes the model parameters from the configuration.
    Runs in either training or testing mode. 
"""
if __name__ == '__main__':
    if len(sys.argv) - 1:
        # Reads parameter values from user provided configuration file.
        config_params = getConfiguration(sys.argv[1])
    else:
        # Reads parameter values from setModerParams.rc file and initializes it.
        config_params = getConfiguration()

    print('\n>> Initialized Garrula in %s mode\n' %(config_params['mode']))

    if config_params['mode'] == 'train':
        # Train our application
        train()
    elif config_params['mode'] == 'test':
        # Invoke Interactive Decode
        test()
    else:
        # Error - Invalid mode provided
        print('# uses setModelParams.rc as conf file')
