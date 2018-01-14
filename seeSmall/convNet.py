from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf 


#GLOBALS
tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN.
     from tensorflow.org tutorial"""
  # Input Layer
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)



##
class SimpleModel:
    """
    """

    _train_data=None
    _train_labels=None

    _eval_data=None
    _eval_labels=None

    _classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, 
        model_dir="/tmp/seeSmall_convnet_model")

    _tensors_to_log = {"probabilities": "softmax_tensor"}
    _logging_hook = tf.train.LoggingTensorHook(
      tensors=_tensors_to_log, every_n_iter=50)

    @classmethod
    def set_train_data(cls, train_data):
        """
        """
        cls._train_data=train_data
    
    @classmethod
    def set_train_label(cls, train_label):
        """
        """
        cls._train_label=train_label
    
    @classmethod
    def set_eval_data(cls, eval_data):
        """
        """
        cls._eval_data=eval_data
    
    @classmethod
    def set_eval_label(cls, eval_label):
        """
        """
        cls._eval_label=eval_labe
    
    @classmethod
    def train(cls):
        """
        """
        _train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": cls._train_data},
            y=cls._train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)
        cls._classifier.train(
            input_fn=_train_input_fn,
            steps=20000,
            hooks=[cls._logging_hook])
    
    @classmethod
    def evaluate(cls):
        """
        """
        _eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": cls._eval_data},
            y=cls._eval_labels,
            num_epochs=1,
            shuffle=False)
        cls._eval_results = cls._classifier.evaluate(
                             input_fn=_eval_input_fn)
        print(cls._eval_results)
    
    @classmethod
    def get_evaluate_results(cls):
        """
        """
        return cls._eval_results
    
    @classmethod
    def classify(cls, toPredict):
        """
        """
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": toPredict},
            num_epochs=1,
            shuffle=False)
        cls._predictions = cls._classifier.predict(input_fn=predict_input_fn)
        for i, p in enumerate(cls._predictions):print(i,p)
        
