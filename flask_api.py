import numpy as np

from flask import Flask
from flask import request
from flask import abort
from time import time

import tensorflow as tf

from extract_features import *



def read_examples(string):
  """Read a list of `InputExample`s from an input file."""
  examples = []
  unique_id = 0
  for texline in string.split('\n'): 
      line = tokenization.convert_to_unicode(texline)
      if not line:
        break
      line = line.strip()
      text_a = None
      text_b = None
      m = re.match(r"^(.*) \|\|\| (.*)$", line)
      if m is None:
        text_a = line
      else:
        text_a = m.group(1)
        text_b = m.group(2)
      examples.append(
          InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
      unique_id += 1
  return examples

class RUN():
    
    
    def __init__(self):
        self.app = Flask(__name__)
        self.initialize_routes()
        pass

    def initialize_routes(self):

        @self.app.route('/docvec',methods=["GET"])
        def docvec():
            string = request.args.get("q")
            tf.logging.info(f"Got string:\n {string}")
            if not string:
                return abort(400)

            t0 = time()
            examples = read_examples(string)

            t1 = time()
            tf.logging.info("time for read_examples: "+str(t1-t0))

            features = convert_examples_to_features(
              examples=examples, seq_length=FLAGS.max_seq_length, tokenizer=self.tokenizer)
            t5 = time()
            tf.logging.info("time for convert_examples_to_features: "+str(t5-t1))

            unique_id_to_feature = {}
            for feature in features:
                unique_id_to_feature[feature.unique_id] = feature

            input_fn = input_fn_builder(
              features=features, seq_length=FLAGS.max_seq_length)
            t10 = time()
            tf.logging.info("time for input_fn_builder: "+str(t10-t5))

            all_outputs = []
            for result in self.estimator.predict(input_fn, yield_single_examples=True):
              unique_id = int(result["unique_id"])
              feature = unique_id_to_feature[unique_id]
              output_json = collections.OrderedDict()
              output_json["linex_index"] = unique_id
              all_features = []
              for (i, token) in enumerate(feature.tokens):
                if token=="[SEP]":
                    break
                j = 0
                layer_index = -1
                layer_output = result["layer_output_%d" % j]
                layers = collections.OrderedDict()
                layers["index"] = layer_index
                layers["values"] =  layer_output[i:(i + 1)].flatten()
                all_features.append(layers["values"])
              output = np.mean(all_features,axis=0)
              output = [round(float(x), 6) for x in output]
              all_outputs.append(output)
            
            return json.dumps(all_outputs)




def main(_):
  
  run = RUN()

  tf.logging.set_verbosity(tf.logging.INFO)

  run.layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

  run.bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  run.tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run.run_config = tf.contrib.tpu.RunConfig(
      master=FLAGS.master,
      tpu_config=tf.contrib.tpu.TPUConfig(
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))


  run.model_fn = model_fn_builder(
      bert_config=run.bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      layer_indexes=run.layer_indexes,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_one_hot_embeddings)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  run.estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=run.model_fn,
      config=run.run_config,
      predict_batch_size=FLAGS.batch_size)

  run.app.run()

  

 


if __name__ == '__main__':
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("init_checkpoint")
  flags.mark_flag_as_required("output_file")
  tf.app.run()
