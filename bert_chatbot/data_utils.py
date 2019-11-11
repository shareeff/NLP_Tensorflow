import csv
import sys 
import os
import copy
import json
import logging
from random import shuffle
import re
import tensorflow as tf
from transformers import is_tf_available

logger = logging.getLogger(__name__)

def preprocess_sentence(sentence):
  sentence = sentence.lower().strip()
  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = re.sub(r'[" "]+', " ", sentence)
  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
  sentence = sentence.strip()
  # adding a start and an end token to the sentence
  return sentence

def cornell_movie_convert_examples_to_features(examples, tokenizer,
                                               max_length=512,
                                               pad_on_left=False,
                                               pad_token=0,
                                               pad_token_segment_id=0,
                                               mask_padding_with_zero=True):

    is_tf_dataset = False
    if is_tf_available():
        is_tf_dataset = True
    
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))

        inputs = tokenizer.encode_plus(
            example.text_a,
            add_special_tokens=True,
            max_length=max_length,
        )

        outputs = tokenizer.encode_plus(
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )

        input_ids, input_token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        output_ids = outputs["input_ids"]

        input_attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
    
        input_padding_length = max_length - len(input_ids)
        output_padding_length = max_length - len(output_ids)

        if pad_on_left:
            input_ids = ([pad_token] * input_padding_length) + input_ids
            input_attention_mask = ([0 if mask_padding_with_zero else 1] * input_padding_length) + input_attention_mask 
            input_token_type_ids = ([pad_token_segment_id] * input_padding_length) + input_token_type_ids

            output_ids = ([pad_token] * output_padding_length) + output_ids

        else:
            input_ids = input_ids + ([pad_token] * input_padding_length)
            input_attention_mask = input_attention_mask + ([0 if mask_padding_with_zero else 1] * input_padding_length)
            input_token_type_ids = input_token_type_ids + ([pad_token_segment_id] * input_padding_length)

            output_ids = output_ids + ([pad_token] * output_padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(input_attention_mask) == max_length, "Error with input length {} vs {}".format(len(input_attention_mask), max_length)
        assert len(input_token_type_ids) == max_length, "Error with input length {} vs {}".format(len(input_token_type_ids), max_length)
        assert len(output_ids) == max_length, "Error with output length {} vs {}".format(len(output_ids), max_length)
        

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_attention_mask: %s" % " ".join([str(x) for x in input_attention_mask]))
            logger.info("input_token_type_ids: %s" % " ".join([str(x) for x in input_token_type_ids]))
            logger.info("output_ids: %s" % " ".join([str(x) for x in output_ids]))


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_attention_mask=input_attention_mask,
                              input_token_type_ids=input_token_type_ids,
                              output_ids=output_ids))

    
    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield  ({'input_ids': ex.input_ids,
                         'attention_mask': ex.input_attention_mask,
                         'token_type_ids': ex.input_token_type_ids},
                        {'output_ids': ex.output_ids})

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             {'output_ids': tf.int32}),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             {'output_ids': tf.TensorShape([None])}))

    return features
            







    

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """Gets an example from a dict with tensorflow tensors

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class CornellMovieDialogsProcessor(DataProcessor):

    def __init__(self):
        self._prepare_dataset()


    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence1'].numpy().decode('utf-8'),
                            tensor_dict['sentence2'].numpy().decode('utf-8'))

    def get_train_examples(self):
        """See base class."""
        #logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self.train_dataset

    def get_dev_examples(self):
        """See base class."""
        return self.test_dataset

    def get_labels(self):
        """See base class."""
        pass

    def _create_examples(self, inputs, outputs):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, (input, output)) in enumerate(zip(inputs, outputs)):
            if i == 0:
                continue
            guid = "%s" % (i)
            text_a = input
            text_b = output
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b))

        return examples

    def split_train_and_test_data(self, dataset):
        train_size = int(0.8 * len(dataset))
        #test_size = len(dataset) - train_size
        shuffle(dataset)
        train_dataset = dataset[:train_size]
        test_dataset = dataset[train_size:]
        print("\ntain dataset length {}".format(len(train_dataset)))
        print("\ntest dataset length {}".format(len(test_dataset)))
        return train_dataset, test_dataset
                
    def _prepare_dataset(self):
        path_to_zip = tf.keras.utils.get_file(
            'cornell_movie_dialogs.zip',
        origin='http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip',
        extract=True)

        path_to_dataset = os.path.join(
            os.path.dirname(path_to_zip), "cornell movie-dialogs corpus")

        path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
        path_to_movie_conversations = os.path.join(path_to_dataset,
                                           'movie_conversations.txt')

        inputs, outputs = self._load_conversations(path_to_movie_lines, path_to_movie_conversations)
        
        dataset = self._create_examples(inputs, outputs)

        self.train_dataset, self.test_dataset = self.split_train_and_test_data(dataset)
        


    def _load_conversations(self, path_to_movie_lines, path_to_movie_conversations):
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

        




class InputExample(object):

    def __init__(self, guid, text_a, text_b=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):

    r"""
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        input_attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        input_token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        output_ids: Indices of output sequence tokens in the vocabulary.
        output_attention_mask:  Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
    """

    def __init__(self, input_ids, input_attention_mask, input_token_type_ids, output_ids):
        self.input_ids = input_ids
        self.input_attention_mask = input_attention_mask
        self.input_token_type_ids = input_token_type_ids
        self.output_ids = output_ids

    
    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

