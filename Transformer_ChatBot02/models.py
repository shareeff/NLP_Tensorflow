from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.random.set_seed(1234)
import sys
import json
import copy

from data_utils import create_padding_mask, create_look_ahead_mask

class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=self.d_model)
        self.key_dense = tf.keras.layers.Dense(units=self.d_model)
        self.value_dense = tf.keras.layers.Dense(units=self.d_model)

        self.dense = tf.keras.layers.Dense(units=self.d_model)

    def split_heads(self, inputs, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth)
        )
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, query, key, value, mask):
    
        matmul_qk = tf.matmul(query, key, transpose_b=True)

        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)

        return output

    def call(self, query, key, value, mask):

        #query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        #linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        #split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = self.scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

    def get_config(self):
        
        return {
            "num_heads" : self.num_heads,
            "d_model" : self.d_model,
        }


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(self.position, self.d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model = d_model
        )
        
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])

        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        
        return {
            "position" : self.position,
            "d_model" : self.d_model,
        }


class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, units, d_model, num_heads, dropout=0.1, name="encoder_layer"):
        super(EncoderLayer, self).__init__(name=name)

        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention = MultiHeadAttention(self.d_model, self.num_heads, name="attention")
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout)

        self.ffn1 = tf.keras.layers.Dense(units=self.units, activation='relu')
        self.ffn2 = tf.keras.layers.Dense(units=self.d_model)

    
    def call(self, inputs, padding_mask, training):

        #query, key, value, mask
        attention = self.attention(inputs, inputs, inputs, padding_mask)
           
        attention = self.dropout1(attention, training=training)
        attention = self.layernorm1(inputs + attention)

        outputs = self.ffn1(attention)
        outputs = self.ffn2(outputs)
        outputs = self.dropout2(outputs)
        outputs = self.layernorm2(attention + outputs)

        return outputs
    

    def get_config(self):
        return {
            "units" : self.units,
            "d_model" : self.d_model,
            "num_heads": self.num_heads,
            "dropout" : self.dropout
        }

    
class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, units, d_model, num_heads, dropout=0.1, name="decoder_layer"):
        super(DecoderLayer, self).__init__(name=name)

        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.attention1 = MultiHeadAttention(self.d_model, self.num_heads, name="attention_1")
        self.attention2 = MultiHeadAttention(self.d_model, self.num_heads, name="attention_2")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout)
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout)
        self.dropout3 = tf.keras.layers.Dropout(rate=self.dropout)

        self.ffn1 = tf.keras.layers.Dense(units=self.units, activation='relu')
        self.ffn2 = tf.keras.layers.Dense(units=self.d_model)

    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask, training):

        attention1 = self.attention1(inputs, inputs, inputs, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        attention1 = self.layernorm1(attention1 + inputs)
        attention2 = self.attention2(attention1, enc_outputs, enc_outputs, padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        attention2 = self.layernorm2(attention2 + attention1)

        outputs = self.ffn1(attention2)
        outputs = self.ffn2(outputs)
        outputs = self.dropout3(outputs, training=training)
        outputs = self.layernorm3(outputs + attention2)

        return outputs

    def get_config(self):
        return {
            "units" : self.units,
            "d_model" : self.d_model,
            "num_heads": self.num_heads,
            "dropout" : self.dropout
        }



class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, units, d_model, num_heads, input_vocab_size, dropout=0.1, name="encoder"):
        super(Encoder, self).__init__(name=name)

        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.vocab_size = input_vocab_size
        self.dropout = dropout

        self.embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)
        self.pos_embeddings = PositionalEncoding(self.vocab_size, self.d_model)

        self.enc_layers = [EncoderLayer(
            units=self.units,
            d_model = self.d_model,
            num_heads = self.num_heads,
            dropout= self.dropout,
            name="encoder_layer_{}".format(i)

        ) for i in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate=self.dropout)


    def call(self, inputs, padding_mask, training):

        embeddings = self.embeddings(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.pos_embeddings(embeddings)

        outputs = self.dropout(embeddings, training=training)
        #inputs, padding_mask, training
        for i in range(self.num_layers):
            outputs = self.enc_layers[i](
                inputs=outputs, 
                padding_mask=padding_mask, 
                training=training)

        return outputs

    def get_config(self):
        return {
            "num_layers" : self.num_layers,
            "units" : self.units,
            "d_model" : self.d_model,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "dropout" : self.dropout
        }


class Decoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, units, d_model, num_heads, target_vocab_size, dropout=0.1, name="decoder"):
        super(Decoder, self).__init__(name=name)

        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.vocab_size = target_vocab_size
        self.dropout = dropout

        self.embeddings = tf.keras.layers.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.vocab_size, self.d_model)

        self.dec_layers = [DecoderLayer(
            units = self.units,
            d_model = self.d_model,
            num_heads = self.num_heads,
            dropout= self.dropout,
            name="decoder_layer_{}".format(i)
        ) for i in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, inputs, enc_outputs, look_ahead_mask, padding_mask, training):

        embeddings = self.embeddings(inputs)
        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        embeddings = self.pos_encoding(embeddings)

        outputs = self.dropout(embeddings, training=training)
        #inputs, enc_outputs, look_ahead_mask, padding_mask, training

        for i in range(self.num_layers):
            outputs = self.dec_layers[i](
                inputs=outputs, 
                enc_outputs=enc_outputs, 
                look_ahead_mask=look_ahead_mask, 
                padding_mask=padding_mask, 
                training=training)

        return outputs

    def get_config(self):
        return {
            "num_layers" : self.num_layers,
            "units" : self.units,
            "d_model" : self.d_model,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "dropout" : self.dropout
        }

        

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, units, d_model, num_heads, vocab_size, dropout=0.1, name="transformer_model"):
        super(Transformer, self).__init__(name=name)
        self.num_layers = num_layers
        self.units = units
        self.d_model = d_model
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.dropout = dropout
        

        
        self.enc_outputs = Encoder(
            num_layers=self.num_layers,
            units = self.units,
            d_model = self.d_model,
            num_heads = self.num_heads,
            input_vocab_size = self.vocab_size,
            dropout= self.dropout
        )

        self.dec_outputs = Decoder(
            num_layers = self.num_layers,
            units = self.units,
            d_model = self.d_model,
            num_heads = self.num_heads,
            target_vocab_size = self.vocab_size,
            dropout= self.dropout
        )

        self.final_layer = tf.keras.layers.Dense(self.vocab_size, name="outputs")

    def call(self, inputs, dec_inputs, enc_padding_mask, look_ahead_mask, dec_padding_mask, training):
       
        
        #inputs, padding_mask, training
        enc_outputs = self.enc_outputs(
            inputs=inputs, 
            padding_mask=enc_padding_mask, 
            training=training)
        dec_outputs = self.dec_outputs(
            inputs=dec_inputs, 
            enc_outputs=enc_outputs, 
            look_ahead_mask= look_ahead_mask, 
            padding_mask= dec_padding_mask,
            training=training)

        outputs = self.final_layer(dec_outputs)

        return outputs

    def get_config(self):
        return {
            "num_layers" : self.num_layers,
            "units" : self.units,
            "d_model" : self.d_model,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "dropout" : self.dropout
        }



def create_transformer_model(num_layers, units, d_model, num_heads, vocab_size, dropout, training, name):

  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")
  enc_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
  # mask the future tokens for decoder inputs at the 1st attention block
  look_ahead_mask = tf.keras.layers.Lambda(
      create_look_ahead_mask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  # mask the encoder outputs for the 2nd attention block
  dec_padding_mask = tf.keras.layers.Lambda(
      create_padding_mask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  internal_model =Transformer(
    num_layers=num_layers,
    units=units,
    d_model=d_model,
    num_heads=num_heads,
    vocab_size=vocab_size,
    dropout=dropout, 
    name='transformer_model'
  )

  logits = internal_model(
      inputs = inputs,
      dec_inputs = dec_inputs, 
      enc_padding_mask = enc_padding_mask, 
      look_ahead_mask = look_ahead_mask, 
      dec_padding_mask = dec_padding_mask,
      training= training)

  logits = tf.keras.layers.Lambda(lambda x: x, name="outputs")(logits)

  return tf.keras.Model(
      inputs={
          'inputs': inputs, 
          'dec_inputs': dec_inputs
      }, 
      outputs=logits, 
      name=name
  )


























        



