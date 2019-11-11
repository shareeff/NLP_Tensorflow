import tensorflow as tf
import tensorflow_datasets
import os
import logging
import time
from transformers import BertTokenizer
from bert_model import Config, Transformer
from data_utils import CornellMovieDialogsProcessor, cornell_movie_convert_examples_to_features
from utils import create_masks
from configuration import *
from run_demo import chatbot

config = Config(num_layers=NUM_LAYERS, d_model=D_MODEL, dff=DFF, num_heads=NUM_HEADS)
start_time = time.time()
logging.basicConfig(filename='training_{}.log'.format(start_time), level=logging.INFO)
logger = logging.getLogger('bert_transformer_training')


os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
transformer = Transformer(config, VOCAB_SIZE, BERT_MODEL_NAME)

processor = CornellMovieDialogsProcessor()
train_examples = processor.get_train_examples()
valid_examples = processor.get_dev_examples()

train_dataset = cornell_movie_convert_examples_to_features(train_examples, tokenizer, MAX_SEQ_LENGTH)
valid_dataset = cornell_movie_convert_examples_to_features(valid_examples, tokenizer, MAX_SEQ_LENGTH)
train_dataset = train_dataset.shuffle(128).batch(BATCH_SIZE)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)

opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
ckpt = tf.train.Checkpoint(model=transformer,
                           optimizer=opt)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
  # if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')





loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
#model.compile(optimizer=opt, loss=loss, metrics=[metric])

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function 
def train_step(inp, tar):
    tar = tar.get('output_ids')
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp['input_ids'], tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    opt.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


for epoch in range(EPOCHS):

    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        if batch % 500 == 0:
            logger.info('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 1 == 0:
      ckpt_save_path = ckpt_manager.save()
      logger.info('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
      print('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    


    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))
    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

    logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    

    if(train_loss.result() < 0.2):
        break


#transformer.save(save_model_path)
transformer.save_weights(save_weight_path)

demo_sentense = 'How are you'
out, _ = chatbot(demo_sentense, tokenizer, transformer)
logger.info('input: {}\n output: {}'.format(demo_sentense, out))










    





