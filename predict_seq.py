# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import modeling
import optimization
import collections
import random
import tokenization
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_dir", None,
    "Output dir for writing the output files")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")


flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for eval.")


# TPU specific
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def create_instances_from_document(document,max_seq_length):
    """document: list of tokens for which to create instance [list]
       max_seq_length: maximum sequence length.

       If document is ["hello","this","is","[MASK]","string"]
       The document is processed to ["[CLS]","hello","this","is","[MASK]","string","[SEP]","[SEP]"]
       """

    max_num_tokens = max_seq_length - 3 #taking care of [CLS], and the two [SEP]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in document[:max_num_tokens]:
      tokens.append(token)
      segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(0)

    tokens.append("[SEP]")
    segment_ids.append(1)

    is_random_next = False

    masked_lm_positions = []
    masked_lm_labels = []
    for i,t in enumerate(tokens):
      if t=="[MASK]":
        masked_lm_positions.append(i)
        masked_lm_labels.append('a')


    instance = TrainingInstance(
        tokens=tokens,
        segment_ids=segment_ids,
        is_random_next=is_random_next,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)

    tf.logging.info("*******GOT DATA POINT********")
    tf.logging.info(instance)
    return [instance]


def create_training_instances(input_files, tokenizer, max_seq_length):
  """Create `TrainingInstance`s from raw text."""
  all_documents = []

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        tokens = []
        for segment in re.split(r"(\[MASK\])",line):
          if segment=="[MASK]":
            tokens.append(segment)
            continue
          _toks = tokenizer.tokenize(segment)
          if _toks:
            tokens.extend(_toks)
        all_documents.append(tokens)

  # Remove empty documents
  all_documents = [x for x in all_documents if x]

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  for document_index in range(len(all_documents)):
    instances.extend(
        create_instances_from_document(
            all_documents[document_index], max_seq_length))
  return instances


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def get_features(instances,tokenizer,max_seq_length):
  examples = []
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_seq_length:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = (input_ids)
    features["input_mask"] = (input_mask)
    features["segment_ids"] = (segment_ids)
    features["masked_lm_positions"] = (masked_lm_positions)
    features["masked_lm_ids"] = (masked_lm_ids)
    features["masked_lm_weights"] = (masked_lm_weights)
    features["next_sentence_labels"] = ([next_sentence_label])


    examples.append(features)
  return examples


def input_fn_builder(features, seq_length):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_segment_ids = []
  all_input_ids = []
  all_input_mask = []
  all_masked_lm_positions = []
  all_masked_lm_weights = []
  all_masked_lm_ids = []
  all_next_sentence_labels = []



  for feature in features:
    all_segment_ids.append(feature["segment_ids"])
    all_input_ids.append(feature["input_ids"])
    all_input_mask.append(feature["input_mask"])
    all_masked_lm_positions.append(feature["masked_lm_positions"])
    all_masked_lm_weights.append(feature["masked_lm_weights"])
    all_masked_lm_ids.append(feature["masked_lm_ids"])
    all_next_sentence_labels.append(feature["next_sentence_labels"])

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "segment_ids":
            tf.constant(all_segment_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "masked_lm_positions":
            tf.constant(
                all_masked_lm_positions,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "masked_lm_weights":
            tf.constant(
                all_masked_lm_weights,
                shape=[num_examples, seq_length],
                dtype=tf.float32),
        "masked_lm_ids":
            tf.constant(
                all_masked_lm_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "next_sentence_labels":
            tf.constant(
                all_next_sentence_labels,
                shape=[num_examples, 1],
                dtype=tf.int32),
    })

    d = d.batch(batch_size=batch_size, drop_remainder=False)
    return d

  return input_fn

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_new_input_ids(input_ids,
              masked_lm_probs,masked_lm_positions,masked_lm_weights):
  
  #Reshaping the masked_lm_probs
  sequence_shape = modeling.get_shape_list(masked_lm_positions, expected_rank=2) 
  #for getting the shape info
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]

  # Gathering the first masks only. shape = (batch_size,vocab_size)
  mask_lab_pred = tf.gather(masked_lm_probs,
                            tf.range(0,batch_size*seq_length,delta=seq_length),axis=0)
  #Getting the indexes for the predictions. shape = (batch_size,1)
  #mask_lab_pred = tf.random.multinomial(mask_lab_pred,1,output_dtype=tf.int32)
  mask_lab_pred = tf.reshape(tf.cast(tf.argmax(mask_lab_pred,
                  axis=-1),dtype=tf.int32),(batch_size,1))
  # Gathering positions of the first mask. shape=(batch_size,)
  mask_positions = tf.gather(masked_lm_positions,
                          tf.constant(0,dtype=tf.int32),axis=-1)
  #Assigning the first masks in input_ids
  #--First converting mask_positions to one hot, shape=(batch_size,seq_length)
  mask_positions = tf.one_hot(mask_positions,depth=seq_length,axis=1,dtype=tf.int32)
  #--Removing those positions there is no mask left
  mask_positions = tf.cast(tf.reduce_max(masked_lm_positions,axis=-1,keepdims=True)>0
                        ,dtype=tf.int32)*mask_positions

  #--Next multiplying mask_positions_one_hot to mask_lab_pred and forming new_ids
  input_ids = mask_positions*mask_lab_pred+(1-mask_positions)*input_ids

  #Setting the first mask lm_weights to be zero

  masked_lm_weights = tf.slice(masked_lm_weights,(0,1),(-1,-1))
  masked_lm_weights = tf.concat([masked_lm_weights,tf.zeros(shape=(batch_size,1),
                                                dtype=tf.float32)],axis=-1)

  #Setting the masked_lm_positions first masks to be zero
  masked_lm_positions = tf.slice(masked_lm_positions,(0,1),(-1,-1))
  masked_lm_positions = tf.concat([masked_lm_positions,tf.zeros(shape=(batch_size,1),
                                                dtype=tf.int32)],axis=-1)

  
  return input_ids,masked_lm_positions,masked_lm_weights
      

def model_fn_builder(bert_config, init_checkpoint, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = False

    # Running while loop for prediction

    def body(new_input_ids,new_masked_lm_positions,new_masked_lm_weights ):

      model = modeling.BertModel(
          config=bert_config,
          is_training=is_training,
          input_ids=new_input_ids,
          input_mask=input_mask,
          token_type_ids=segment_ids,
          use_one_hot_embeddings=use_one_hot_embeddings)

      (_,_, masked_lm_log_probs) = get_masked_lm_output(
           bert_config, model.get_sequence_output(), model.get_embedding_table(),
           new_masked_lm_positions, masked_lm_ids, new_masked_lm_weights)

      return get_new_input_ids(new_input_ids,
              masked_lm_log_probs,new_masked_lm_positions,new_masked_lm_weights) 
      

    def cond(input_ids,masked_lm_positions,masked_lm_weights):
      # If there is no masked element, we are done
      return tf.reduce_max(masked_lm_weights)>0


    ids,pos,wt = tf.while_loop(
          cond=cond,
          body=body,
          loop_vars=[
                input_ids,
                masked_lm_positions,
                masked_lm_weights
          ],
          back_prop=False
      )


    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)
    
    # Prediction mode
    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions={"ids": ids},
          scaffold_fn=scaffold_fn)

    return output_spec

  return model_fn



def get_index_for_prediction(instances):
  done = [True]*len(instances)
  for i,f in enumerate(instances):
    if len(instances[i].masked_lm_positions)==0:
      done[i]=False

  return done


def write_output(writer,tokens):
  tf.logging.info("***** Pred results *****")
  to_print=""
  for t in tokens:
    if t in ["[CLS]","[SEP]","[PAD]"]:
      continue
    if t.startswith("##"):
      to_print+=t[2:]
    else:
      to_print+=" "+t

  tf.logging.info(to_print)
  writer.write(to_print+"\n")


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  


  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      predict_batch_size=FLAGS.predict_batch_size)

  tf.logging.info("***** Running prediction *****")
  tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

  itern = 0
  output_pred_file = os.path.join(FLAGS.output_dir, "pred_results.txt")
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length
      )

  todo = get_index_for_prediction(instances)

  features = get_features([instances[i] for i in range(len(todo)) if todo[i]], tokenizer, FLAGS.max_seq_length)

  pred_input_fn = input_fn_builder(features=features, seq_length=FLAGS.max_seq_length)

  num_examples = len(features)

  with tf.gfile.GFile(output_pred_file, "w") as writer:
    for result in estimator.predict(pred_input_fn, yield_single_examples=False):
      for i in range(len(result['ids'])):
            write_output(writer,tokenizer.convert_ids_to_tokens(result['ids'][i]))
        
      

if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
