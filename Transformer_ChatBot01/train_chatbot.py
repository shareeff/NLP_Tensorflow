from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.random.set_seed(1234)
import tensorflow_datasets as tfds

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import time

from models import Transformer
from run_demo import predict
from data_utils import create_masks, preprocess_sentence
from configuration import *


path_to_zip = tf.keras.utils.get_file(
    'cornell_movie_dialogs.zip',
    origin=
    'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
    extract=True)

path_to_dataset = os.path.join(
    os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
path_to_movie_conversations = os.path.join(path_to_dataset,
                                           'movie_conversations.txt')

def load_conversations():
  # dictionary of line id to text
  id2line = {}
  with open(path_to_movie_lines, errors='ignore') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    id2line[parts[0]] = parts[4]

  inputs, outputs = [], []
  with open(path_to_movie_conversations, 'r') as file:
    lines = file.readlines()
  for line in lines:
    parts = line.replace('\n', '').split(' +++$+++ ')
    # get conversation in a list of line ID
    conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
    for i in range(len(conversation) - 1):
      inputs.append(preprocess_sentence(id2line[conversation[i]]))
      outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
      
  return inputs, outputs




# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs, tokenizer):

  tokenized_inputs, tokenized_outputs = [], []
  # Define start and end token to indicate the start and end of a sentence
  START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
  
  for (sentence1, sentence2) in zip(inputs, outputs):
    # tokenize sentence
    sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
    sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
    # check tokenized sentence max length
    if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
      tokenized_inputs.append(sentence1)
      tokenized_outputs.append(sentence2)
  
  # pad tokenized sentences
  tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
  tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
  return tokenized_inputs, tokenized_outputs

def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  accuracy = tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)
  return accuracy

@tf.function
def train_step(inp, tar, model, optimizer, train_loss, train_accuracy):
  inputs = inp['inputs']
  print(type(tar))
  dec_inputs = tar['outputs'][:, :-1]
  outputs = tar['outputs'][:, 1:]  
  enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inputs, dec_inputs)
  #inputs, dec_inputs, enc_padding_mask, look_ahead_mask, dec_padding_mask, training
  with tf.GradientTape() as tape:
    predictions = model(
      inputs=inputs,
      dec_inputs=dec_inputs,
      enc_padding_mask=enc_padding_mask,
      look_ahead_mask=look_ahead_mask,
      dec_padding_mask=dec_padding_mask,
      training=True
    )
    loss = loss_function(outputs, predictions)

  gradients = tape.gradient(loss, model.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  train_loss(loss)
  train_accuracy(outputs, predictions)


def main():
  questions, answers = load_conversations()
  # Build tokenizer using tfds for both questions and answers
  tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

  tokenizer.save_to_file(vocab_filename)

  # Vocabulary size plus start and end token
  VOCAB_SIZE = tokenizer.vocab_size + 2


  questions, answers = tokenize_and_filter(questions, answers, tokenizer)
  print('Vocab size: {}'.format(VOCAB_SIZE))
  print('Number of samples: {}'.format(len(questions)))
  # decoder inputs use the previous target as input
  # remove START_TOKEN from targets
  dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions
    },
    {
        'outputs': answers
    },
  ))

  dataset = dataset.cache()
  dataset = dataset.shuffle(BUFFER_SIZE)
  dataset = dataset.batch(BATCH_SIZE)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  print(dataset)
 
  model = Transformer(
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    vocab_size=VOCAB_SIZE,
    dropout=DROPOUT, 
    name='transformer'
  )

 
  learning_rate = CustomSchedule(D_MODEL)

  optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  ckpt = tf.train.Checkpoint(model=model,
                           optimizer=optimizer)
                        
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

  for epoch in range(EPOCHS):
    start = time.time()
  
    train_loss.reset_states()
    train_accuracy.reset_states()
   
  
    for (batch, (inp, tar)) in enumerate(dataset):
     
      train_step(inp, tar, model, optimizer, train_loss, train_accuracy)
    
      if batch % 500 == 0:
        print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))

      
    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


  
  
  
  model.save_weights(save_weight_path)
  #model.summary()
  input_sentence = 'Where have you been?'
  predict(input_sentence, tokenizer, model)

  sentence = 'I am not crazy, my mother had me tested.'
  for _ in range(5):
    sentence = predict(sentence, tokenizer, model)
    print('')




if __name__ == "__main__":
  main()
  