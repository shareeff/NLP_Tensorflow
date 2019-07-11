# Transformer_ChatBot01




## Dependencies

* Tensorflow 2.0


## Usage

For Training Run:

```
$ python3 train_chatbot.py 

```

For Demo Run:

``` 
$ python3 run_demo.py --input='input sentence'

```

As tensorflow subclass method doesn't have any DAG of layers. It doesn't know about the input. So the model is wraped with keras functional api. As the result the model can directly load weight without initializing the model.

```
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
```

## Reference

[1] https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/transformer.ipynb

[2] https://github.com/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb

