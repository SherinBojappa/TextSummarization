import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers.core import Dense

import nltk

import numpy as np
import pickle

from tensorflow.python.platform import flags

flags.DEFINE_string("attention", "yes", "attention (yes/no)")

FLAGS = flags.FLAGS

print(FLAGS.attention)

batch_size = 100
num_hidden_units_for_proj = 200
rnn_hidden_size = 200

# load preprocessed data
#N = 50000
#preprocessed_data = pickle.load(open('preprocessed_data_'+str(N)+'.pkl', 'r'))
preprocessed_data = pickle.load(open('preprocessed_data.pkl', 'r'))

cn_embedding_matrix_numpy = preprocessed_data["cn_embedding_matrix_numpy"]
word_to_num = preprocessed_data["word_to_num"]
num_to_word = preprocessed_data["num_to_word"]
sorted_text_indices = preprocessed_data["sorted_text_indices"]
review_text_to_num = preprocessed_data["review_text_to_num"]
review_summary_to_num = preprocessed_data["review_summary_to_num"]

N = len(sorted_text_indices)
sorted_text_indices_train = sorted_text_indices[sorted_text_indices<int(N*0.95)]
sorted_text_indices_test = sorted_text_indices[sorted_text_indices>int(N*0.95)]
vocab_size = cn_embedding_matrix_numpy.shape[0]
EOS = word_to_num["<EOS>"]
GO = word_to_num["<GO>"]

embedding = tf.get_variable(name="embedding", shape=cn_embedding_matrix_numpy.shape, initializer=tf.constant_initializer(cn_embedding_matrix_numpy), trainable=False)

# Input placeholders

encoder_ip_seq = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_ip_seq')
decoder_ip_seq = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_ip_seq')

encoder_ip_seq_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_ip_seq_length')
decoder_ip_seq_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_ip_seq_length')

# Add GO in the beginnig of the decoder training ips
decoder_ip_seq_train = tf.concat([tf.ones(shape=[batch_size, 1], dtype=tf.int32) * GO,
                                  decoder_ip_seq], axis=1, name='decoder_ip_seq_train')
decoder_ip_seq_length_train = decoder_ip_seq_length + 1

# Add EOS in the end of the decoder training ops
decoder_op_seq_train = tf.concat([decoder_ip_seq,
                                  tf.ones(shape=[batch_size, 1], dtype=tf.int32) * EOS],
                                 axis=1, name='decoder_op_seq_train')

# Encoder
with tf.variable_scope("encoder") as scope:
  # encoder cell
  encoder_cell = LSTMCell(rnn_hidden_size)
  encoder_ip_embedded = tf.nn.embedding_lookup(embedding, encoder_ip_seq)
  encoder_ip_embedded = Dense(num_hidden_units_for_proj, dtype=tf.float32, name='ip_proj')(encoder_ip_embedded)

  (encoder_op, encoder_last_state) = (
    tf.nn.dynamic_rnn(cell=encoder_cell,
                      inputs=encoder_ip_embedded,
                      sequence_length=encoder_ip_seq_length,
                      time_major=False,
                      dtype=tf.float32))

# Decoder
with tf.variable_scope("decoder") as scope:
  # reduce the dimension of the input layer from vocab_size to num_hidden_units
  input_layer = Dense(num_hidden_units_for_proj, dtype=tf.float32, name='in_proj')
  # Increase the size of the output from num_hidden units to vocab_size
  output_layer = Dense(vocab_size, dtype=tf.float32, name='out_proj')

  # building attention model
  attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=num_hidden_units_for_proj, memory=encoder_op,
                                                             memory_sequence_length=encoder_ip_seq_length)

  # decoder cell
  decoder_cell = LSTMCell(rnn_hidden_size)

  if FLAGS.attention == "yes":
    print("here")
    attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                  attention_layer_size=num_hidden_units_for_proj,
                                                  initial_cell_state=encoder_last_state)

    attn_zero = attn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    decoder_initial_state = attn_zero.clone(cell_state=encoder_last_state)
  else:
    attn_cell = decoder_cell
    decoder_initial_state = encoder_last_state

  decoder_ip_embedded_train = tf.nn.embedding_lookup(embedding, decoder_ip_seq_train)
  decoder_ip_embedded_train = input_layer(decoder_ip_embedded_train)

  helper_for_training = seq2seq.TrainingHelper(inputs=decoder_ip_embedded_train,
                                               sequence_length=decoder_ip_seq_length_train,
                                               time_major=False,
                                               name='helper_for_training')

  decoder_for_training = seq2seq.BasicDecoder(attn_cell,
                                              helper=helper_for_training,
                                              initial_state=decoder_initial_state,
                                              output_layer=output_layer)

  (decoder_op_train, decoder_last_state_train, decoder_op_length_train) = (seq2seq.dynamic_decode(
    decoder=decoder_for_training,
    output_time_major=False,
    impute_finished=True,
    maximum_iterations=tf.reduce_max(decoder_ip_seq_length_train)))

  # decoder_logits_train is nothing but the RNN value
  decoder_logits_train = tf.identity(decoder_op_train.rnn_output)
  # Pick the index of the max value predicted by the RNN
  decoder_preds_train = tf.argmax(decoder_logits_train, axis=-1, name='decoder_preds_train')


  def decoder_embed_proj(x):
    return input_layer(tf.nn.embedding_lookup(embedding, x))


  helper_for_inference = seq2seq.GreedyEmbeddingHelper(start_tokens=tf.ones([batch_size, ], tf.int32) * GO,
                                                       end_token=EOS,
                                                       embedding=decoder_embed_proj)

  decoder_for_inference = seq2seq.BasicDecoder(cell=attn_cell,
                                               helper=helper_for_inference,
                                               initial_state=decoder_initial_state,
                                               output_layer=output_layer)

  (decoder_op_inference, decoder_last_state_inference, decoder_op_length_inference) = (seq2seq.dynamic_decode(
    decoder=decoder_for_inference,
    output_time_major=False,
    maximum_iterations=10))
  decoder_preds_inference = tf.expand_dims(decoder_op_inference.sample_id, -1)

# Time major and other formatting
def get_batch(ip):
  sequence_lengths = [len(seq) for seq in ip]
  max_seq_length = max(sequence_lengths)

  inputs_batch_major = np.ones(shape=[batch_size, max_seq_length], dtype=np.int32) * word_to_num["<PAD>"]

  for i, seq in enumerate(ip):
    for j, element in enumerate(seq):
      inputs_batch_major[i, j] = element

  return inputs_batch_major, sequence_lengths

# Define loss and training operation
masks = tf.sequence_mask(lengths=decoder_ip_seq_length_train,
                         maxlen=tf.reduce_max(decoder_ip_seq_length_train),
                         dtype=tf.float32,
                         name='masks')
loss = seq2seq.sequence_loss(logits=decoder_logits_train,
                             targets=decoder_op_seq_train,
                             weights=masks,
                             average_across_timesteps=True,
                             average_across_batch=True, )

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.001, global_step, 2000, 0.9, staircase=True)


train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)


# training
with tf.Session() as session:
  session.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  num_batches_train = len(sorted_text_indices_train) / batch_size
  num_batches_test = len(sorted_text_indices_test) / batch_size

  for epoch in range(100):

    avg_train_loss = 0.0

    for batch in range(num_batches_train):
      text_ids_train = sorted_text_indices_train[batch * batch_size:(batch + 1) * batch_size]
      batch_review_data_train = [review_text_to_num[i] for i in text_ids_train]
      batch_summary_data_train = [review_summary_to_num[i] for i in text_ids_train]
      inputs_train_, inputs_length_train_ = get_batch(batch_review_data_train)
      targets_train_, targets_length_train_ = get_batch(batch_summary_data_train)

      fd_train = {
        encoder_ip_seq: inputs_train_,
        encoder_ip_seq_length: inputs_length_train_,
        decoder_ip_seq: targets_train_,
        decoder_ip_seq_length: targets_length_train_,
      }

      _, l = session.run([train_op, loss], fd_train)
      avg_train_loss += l / num_batches_train

      if batch == 0 or batch % 1000 == 0:
        print('train batch {}'.format(batch))
        print('minibatch loss: {}'.format(l))
        for i, (e_in, d_in, dt_pred) in enumerate(zip(
            fd_train[encoder_ip_seq], fd_train[decoder_ip_seq],
            session.run(decoder_preds_inference, fd_train).tolist()
        )):
          # print(dt_pred)
          e_in_words = " ".join([num_to_word[e] for e in e_in])
          d_in_words = " ".join([num_to_word[d] for d in d_in if num_to_word[d]!="<PAD>"])
          dt_pred_words = " ".join([num_to_word[p[0]] for p in dt_pred if num_to_word[p[0]]!="<PAD>"])
          print('TRAIN:: sample number {}:'.format(i + 1))
          print('enc input: {}'.format(e_in_words))
          print('dec train true:  {}'.format(d_in_words))
          print('dec train predicted:  {}'.format(dt_pred_words))
          if i >= 2:
            break

    print('epoch: {}'.format(epoch))
    print('epoch train loss: {}'.format(avg_train_loss))
    saver.save(session, 'model_full_attention_'+FLAGS.attention)

    avg_bleu = 0.0
    avg_test_loss = 0.0

    for batch in range(num_batches_test):

      text_ids_test = sorted_text_indices_test[batch * batch_size:(batch + 1) * batch_size]
      batch_review_data_test = [review_text_to_num[i] for i in text_ids_test]
      batch_summary_data_test = [review_summary_to_num[i] for i in text_ids_test]
      inputs_test_, inputs_length_test_ = get_batch(batch_review_data_test)
      targets_test_, targets_length_test_ = get_batch(batch_summary_data_test)

      fd_test = {
        encoder_ip_seq: inputs_test_,
        encoder_ip_seq_length: inputs_length_test_,
        decoder_ip_seq: targets_test_,
        decoder_ip_seq_length: targets_length_test_,
      }

      l = session.run(loss, fd_test)
      avg_test_loss += l / num_batches_test
      preds_test_ = session.run(decoder_preds_inference, fd_test)

      for i, (e_in, d_in, dt_pred) in enumerate(zip(
          fd_test[encoder_ip_seq], fd_test[decoder_ip_seq],
          preds_test_.tolist()
      )):
        # print(dt_pred)
        e_in_words = " ".join([num_to_word[e] for e in e_in])
        d_in_words = " ".join([num_to_word[d] for d in d_in if num_to_word[d]!="<PAD>"])
        dt_pred_words = " ".join([num_to_word[p[0]] for p in dt_pred if num_to_word[p[0]]!="<PAD>"])

        blue_score = nltk.translate.bleu_score.sentence_bleu([d_in_words], dt_pred_words, weights=(0.5, 0.5))
        avg_bleu += blue_score / (num_batches_test * batch_size)

        if i < 2 and (batch == 0 or batch % 50 == 0):
          print('batch {}'.format(batch))
          print('TEST:: sample number {}:'.format(i + 1))
          print('enc test input           > {}'.format(e_in_words))
          print('dec test true:  {}'.format(d_in_words))
          print('dec test predicted:  {}'.format(dt_pred_words))

    print('epoch: {}'.format(epoch))
    print('epoch train loss: {}'.format(avg_train_loss))
    print('epoch test loss: {}'.format(avg_test_loss))
    print('epoch test bleu: {}'.format(avg_bleu))
