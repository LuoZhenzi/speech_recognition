import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wav
import string
import os
import time
from python_speech_features import mfcc

WAV_PATH = './TIMIT_DATA/train'
LABEL_PATH = './TIMIT/TRAIN'
NUM_FEATURES = 13
FIRST_INDEX = ord('a') - 1 
NUM_CLASSES = ord('z') - ord('a') + 1 + 1 + 1
NUM_EPOCHS = 10000
NUM_HIDDEN = 128
NUM_LAYERS = 1
BATCH_SIZE = 20
INITIAL_LEARNING_RATE = 0.001
DECAY_STEPS = 5000
LEARNING_RATE_DECAY_FACTOR = 0.9
#NUM_EXAMPLES = 1
#NUM_BATCHES_PER_EPOCH = int(NUM_EXAMPLES/BATCH_SIZE)

def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape

def get_wav_files(wav_path=WAV_PATH):
    wav_files = []
    for (dir_path, dir_names, file_names) in os.walk(wav_path):
        for file_name in file_names:
            if file_name.endswith('.wav') or file_name.endswith('.WAV'):
                file_path = os.sep.join([dir_path, file_name])
                wav_files.append(file_path)
    return wav_files

wav_files = get_wav_files()

def get_wav_labels(wav_files=wav_files, label_path=LABEL_PATH):
    labels_dict = {}
    for (dir_path, dir_names, file_names) in os.walk(label_path):
        for file_name in file_names:
            if file_name.endswith('.txt') or file_name.endswith('.TXT'):
                label_id = file_name.split('.')[0]
                with tf.gfile.Open(os.sep.join([dir_path, file_name]), 'r') as f:
                    label = f.readlines()[0]
                    label = label.strip('\n')
                    label_text = label.split(' ', 2)[2]
                    for c in string.punctuation:
                        label_text = label_text.replace(c, '')
                    label_text = label_text.lower()
                    labels_dict[label_id] = label_text
    labels = []
    new_wav_files = []
    for wav_file in wav_files:
        wav_id = os.path.basename(wav_file).split('.')[0]
        if wav_id in labels_dict:
            new_wav_files.append(wav_file)
            labels.append(labels_dict[wav_id])
    return new_wav_files, labels

wav_files, labels = get_wav_labels()

def label_to_vector(labels=labels):
    labels_vector = []
    for label in labels:
        label = label.replace(' ', '  ')
        label = label.split(' ')
        label = np.hstack([' ' if x == '' else list(x) for x in label])
        label_vector = np.asarray([0 if x == ' ' else ord(x) - FIRST_INDEX for x in label])
        labels_vector.append(label_vector)
    return labels_vector

labels_vector = label_to_vector()

def feature_extract(wav_files=wav_files):
    wav_mfcc = []
    for wav_file in wav_files:
        fs, audio = wav.read(wav_file)
        inputs = mfcc(audio, samplerate=fs)
        wav_mfcc.append(inputs)
    return wav_mfcc

wav_mfcc = feature_extract(wav_files)

wav_max_len = 0
for inputs in wav_mfcc:
    if len(inputs) > wav_max_len:
        wav_max_len = len(inputs)

label_max_len = 0
for label in labels_vector:
    if len(label) > label_max_len:
        label_max_len = len(label)

print(wav_files[0], labels[0])
print(len(wav_files), len(labels))
print(wav_files[231], labels[231])
print(labels_vector[0])
print(labels_vector[231])
print(wav_max_len, label_max_len)
print(wav_mfcc[0].shape)

NUM_EXAMPLES = 100
NUM_BATCHES_PER_EPOCH = int(NUM_EXAMPLES/BATCH_SIZE)

pointer = 0
def get_next_batch(batch_size):
    global pointer
    batch_wav = []
    batch_label = []
    batch_seq_len = []
    for i in range(batch_size):
        inputs = wav_mfcc[pointer].tolist()
        label = labels_vector[pointer].tolist()
        batch_wav.append(inputs)
        batch_label.append(label)
        batch_seq_len.append(wav_max_len)
        pointer += 1
    for feature in batch_wav:
        while len(feature) < wav_max_len:
            feature.append([0]*13)
    for label in batch_label:
        while len(label) < label_max_len:
            label.append(0)
    batch_label = sparse_tuple_from(batch_label)
    return batch_wav, batch_label, batch_seq_len

def get_train_model(inputs, seq_len):
    cell_fw = tf.contrib.rnn.LSTMCell(NUM_HIDDEN,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
        state_is_tuple=True)
    cells_fw = [cell_fw] * NUM_LAYERS
    cell_bw = tf.contrib.rnn.LSTMCell(NUM_HIDDEN,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
        state_is_tuple=True)
    cells_bw = [cell_bw] * NUM_LAYERS
    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
        cells_fw,
        cells_bw,
        inputs,
        dtype=tf.float32,
        sequence_length=seq_len)
    shape = tf.shape(inputs)
    batch_s, max_timesteps = shape[0], shape[1]
    outputs = tf.reshape(outputs, [-1, NUM_HIDDEN])
    w = tf.Variable(tf.truncated_normal([NUM_HIDDEN, NUM_CLASSES], stddev=0.1))
    b = tf.Variable(tf.constant(0., shape=[NUM_CLASSES]))
    logits = tf.matmul(outputs, w) + b
    logits = tf.reshape(logits, [batch_s, -1, NUM_CLASSES])
    logits = tf.transpose(logits, (1, 0, 2))
    return logits

def train():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        DECAY_STEPS,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    inputs = tf.placeholder(tf.float32, [None, None, NUM_FEATURES])
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])
    logits = get_train_model(inputs, seq_len)
    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)
    optimizer = tf.train.MomentumOptimizer(INITIAL_LEARNING_RATE, 0.9).minimize(cost)
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        #cost,
        #global_step=global_step)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    #decoded, log_prob = tf.nn.ctc_beam_search_decoder(
        #logits,
        #seq_len,
        #merge_repeated=False)
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        for curr_epoch in range(NUM_EPOCHS):
            train_cost = train_ler = 0
            start = time.time()
            global pointer
            pointer = 0
            for batch in range(NUM_BATCHES_PER_EPOCH):
                train_inputs, train_targets, train_seq_len = get_next_batch(BATCH_SIZE)
                feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len}
                batch_cost, _ = session.run([cost, optimizer], feed)
                train_cost += batch_cost * BATCH_SIZE
                train_ler += session.run(ler, feed_dict=feed)*BATCH_SIZE
            train_cost /= NUM_EXAMPLES
            train_ler /= NUM_EXAMPLES
            if train_ler < 0.1:
                saver.save(session, './save/sr.model', global_step=curr_epoch)
                break
            if (curr_epoch + 1) % 10 == 0:
                saver.save(session, './save/sr.model', global_step=curr_epoch)
            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, time = {:.3f}"
            print(log.format(curr_epoch+1,
                NUM_EPOCHS, train_cost, train_ler, time.time() - start))

def recognition():
    inputs = tf.placeholder(tf.float32, [None, None, NUM_FEATURES])
    targets = tf.sparse_placeholder(tf.int32)
    seq_len = tf.placeholder(tf.int32, [None])
    logits = get_train_model(inputs, seq_len)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
    global pointer
    pointer = 0
    print(wav_files[pointer], labels[pointer])
    saver = tf.train.Saver()
    with tf.Session() as session:
        saver.restore(session, './save/sr.model-199')
        test_inputs, test_targets, test_seq_len = get_next_batch(1)
        feed = {inputs: test_inputs,
            targets: test_targets,
            seq_len: test_seq_len}
        d = session.run(decoded[0], feed)
        str_decoded = ''.join([chr(x + FIRST_INDEX) for x in np.asarray(d[1])])
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        print(str_decoded)

train()
