
USE_XLA = False
USE_AMP = False
BERT_MODEL_NAME = 'bert-base-uncased'
VOCAB_SIZE = 30522
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16
EVAL_BATCH_SIZE = BATCH_SIZE * 2
EPOCHS = 10  #30
NUM_LAYERS = 6
D_MODEL = 256
DFF = 1024
NUM_HEADS = 8

checkpoint_path = "./save/checkpoint/bertchatbot"
save_model_path = "./save/model/bertchatbot"
save_weight_path = "./save/model/bertchatbot_weights.h5"