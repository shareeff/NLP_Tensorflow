from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import re
import sys

import tensorflow as tf
tf.random.set_seed(1234)
import tensorflow_datasets as tfds


from models import Transformer
from data_utils import create_masks, preprocess_sentence
from configuration import *

#checkpoint_path1 = "./data/chatbot/ckpt-1"
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


def evaluate(sentence, tokenizer, model):
  START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)
  
    
  enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(sentence, output)

  for _ in range(MAX_LENGTH):

    #inputs, dec_inputs, enc_padding_mask, look_ahead_mask, dec_padding_mask, training
    predictions = model(
      inputs=sentence,
      dec_inputs=output,
      enc_padding_mask=enc_padding_mask,
      look_ahead_mask=look_ahead_mask,
      dec_padding_mask=dec_padding_mask,
      training=False
    )
    #print('model output shape: {}'.format(tf.shape(predictions)))
    
    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]
    #print(tf.shape(predictions))
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    #print(tf.shape(predicted_id))
    #print('model prediction ID: {}'.format(predicted_id))
   
   

    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, END_TOKEN[0]) is None:
      break

    # concatenated the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)


def predict(sentence, tokenizer, model, initialize=False):
  prediction = evaluate(sentence, tokenizer, model)
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  if not initialize:
    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

  return predicted_sentence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default=None,
                        type=str,
                        help="run demo chatbot")

    args = parser.parse_args()

    input_sentence = args.input

    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)
    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2

    model = Transformer(
      num_layers=NUM_LAYERS,
      units=UNITS,
      d_model=D_MODEL,
      num_heads=NUM_HEADS,
      vocab_size=VOCAB_SIZE,
      dropout=DROPOUT, 
      name='transformer'
    )
    demo_sentense = 'How are you'
    predict(demo_sentense, tokenizer, model, True)

    model.load_weights(save_weight_path)

    model.summary()
    tf.keras.utils.plot_model(
    model, to_file='transformer.png', show_shapes=True)



    predict(input_sentence, tokenizer, model)












if __name__ == "__main__":
  main()
  