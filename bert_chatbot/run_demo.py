import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BertTokenizer
from bert_model import Config, Transformer
from configuration import *
from data_utils import preprocess_sentence
from utils import create_masks

START_TOKEN = 101
STOP_TOKEN = 102


def evaluate(sentence, 
            tokenizer, 
            model, 
            max_length=MAX_SEQ_LENGTH,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            mask_padding_with_zero=True):

    sentence = preprocess_sentence(sentence)

    inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=max_length,
        )

    input_ids, input_token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    input_attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    input_padding_length = max_length - len(input_ids)

    if pad_on_left:
        input_ids = ([pad_token] * input_padding_length) + input_ids
        input_attention_mask = ([0 if mask_padding_with_zero else 1] * input_padding_length) + input_attention_mask 
        input_token_type_ids = ([pad_token_segment_id] * input_padding_length) + input_token_type_ids

    else: 
        input_ids = input_ids + ([pad_token] * input_padding_length)
        input_attention_mask = input_attention_mask + ([0 if mask_padding_with_zero else 1] * input_padding_length)
        input_token_type_ids = input_token_type_ids + ([pad_token_segment_id] * input_padding_length)

    
    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(input_attention_mask) == max_length, "Error with input length {} vs {}".format(len(input_attention_mask), max_length)
    assert len(input_token_type_ids) == max_length, "Error with input length {} vs {}".format(len(input_token_type_ids), max_length)

    input_ids = tf.expand_dims(input_ids, 0)
    input_attention_mask = tf.expand_dims(input_attention_mask, 0)
    input_token_type_ids = tf.expand_dims(input_token_type_ids, 0)

    bert_input = {'input_ids': input_ids,
                    'attention_mask': input_attention_mask,
                    'token_type_ids': input_token_type_ids}

    decoder_input = [START_TOKEN]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                                bert_input['input_ids'], output)

        predictions, attention_weights = model(bert_input,
                                                         output,
                                                         False,
                                                         enc_padding_mask,
                                                         combined_mask,
                                                         dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        output = tf.concat([output, predicted_id], axis=-1)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, STOP_TOKEN):
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
        

    return tf.squeeze(output, axis=0), attention_weights


def chatbot(sentence, tokenizer, model, demo=True):
    output, attention_weights = evaluate(sentence, tokenizer, model)
    print(output)
    if demo:
        output = output.numpy().tolist()
        predicted_sentence = tokenizer.decode(output[1:-1])
        print('Input: {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))
        return predicted_sentence, attention_weights



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default=None,
                        type=str,
                        help="run demo chatbot")

    args = parser.parse_args()

    input_sentence = args.input

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    config = Config(num_layers=NUM_LAYERS, d_model=D_MODEL, dff=DFF, num_heads=NUM_HEADS)
    transformer = Transformer(config, VOCAB_SIZE, BERT_MODEL_NAME)
    demo_sentense = 'How are you'
    chatbot(input_sentence, tokenizer, transformer, demo=False)
    transformer.load_weights(save_weight_path)

    #model = load_model(save_model_path)
    
    chatbot(input_sentence, tokenizer, transformer)












if __name__ == "__main__":
    main()


        




