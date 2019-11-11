__version__ = "2.1.1"

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')
    absl.logging._warn_preinit_stderr = False
except:
    pass

import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

# Files and general utilities
from .file_utils import (TRANSFORMERS_CACHE, PYTORCH_TRANSFORMERS_CACHE, PYTORCH_PRETRAINED_BERT_CACHE,
                         cached_path, add_start_docstrings, add_end_docstrings,
                         WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME, CONFIG_NAME,
                         is_tf_available, is_torch_available)

# Tokenizers
from .tokenization_utils import (PreTrainedTokenizer)
from .tokenization_bert import BertTokenizer, BasicTokenizer, WordpieceTokenizer


# Configurations
from .configuration_utils import PretrainedConfig
from .configuration_bert import BertConfig, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

# TensorFlow
if is_tf_available():
    from .modeling_tf_utils import TFPreTrainedModel, TFSharedEmbeddings, TFSequenceSummary

    from .modeling_tf_bert import (TFBertPreTrainedModel, TFBertMainLayer, TFBertEmbeddings,
                                   TFBertModel, TFBertForPreTraining,
                                   TFBertForMaskedLM, TFBertForNextSentencePrediction,
                                   TFBertForSequenceClassification, TFBertForMultipleChoice,
                                   TFBertForTokenClassification, TFBertForQuestionAnswering,
                                   TF_BERT_PRETRAINED_MODEL_ARCHIVE_MAP)

# TF 2.0 <=> PyTorch conversion utilities
from .modeling_tf_pytorch_utils import (convert_tf_weight_name_to_pt_weight_name,
                                        load_pytorch_checkpoint_in_tf2_model,
                                        load_pytorch_weights_in_tf2_model,
                                        load_pytorch_model_in_tf2_model,
                                        load_tf2_checkpoint_in_pytorch_model,
                                        load_tf2_weights_in_pytorch_model,
                                        load_tf2_model_in_pytorch_model)

if not is_tf_available():
    logger.warning("TensorFlow >= 2.0 have been found."
                   "Models won't be available and only tokenizers, configuration"
                   "and file/data utilities can be used.")

    



