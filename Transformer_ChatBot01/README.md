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

As tensorflow subclass method doesn't have any DAG of layers. It doesn't know about the input. So before loading weights(model.load_weights(save_weight_path)), you'll need to initilize the model. In this demo (run_demo.py) the model is initilized with fake demo sentense.

```
demo_sentense = 'How are you'
predict(demo_sentense, tokenizer, model, True)

```

## Reference

[1] https://github.com/tensorflow/docs/blob/master/site/en/r2/tutorials/text/transformer.ipynb

[2] https://github.com/tensorflow/examples/blob/master/community/en/transformer_chatbot.ipynb

