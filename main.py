import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

if "-h" in sys.argv:
    print("python3 main.py [-t] [-e float] [-n int] [-s string] [-l bool]")
    print("## -t :")
    print("Specify whether the model should be trained")
    print("## -e float :")
    print("Specify the temperature of the tweets generated")
    print("## -n int :")
    print("Specify the number of tweets generated")
    print("## -s string :")
    print("Specify the staring string for all tweets generated")
    print("## -l bool :")
    print("Specify whether the generated tweets should be saved in the lib")
    exit(0)

training = ("-t" in sys.argv)
num = 10
start = "Trump "
temp = 0.5
addToLibrary = True
if "-l" in sys.argv:
    lib_index = sys.argv.index("-l")
    try:
        addToLibrary = bool(sys.argv[lib_index + 1])
    except:
        print("-l has to have a following string to specify true or false whether generated tweets should be saved")
        exit(1)
if "-e" in sys.argv:
    tmp_index = sys.argv.index("-e")
    try:
        temp = float(sys.argv[tmp_index + 1])
    except:
        print("-e has to have a following float to specify the temperature of generated tweets")
        exit(1)
if "-n" in sys.argv:
    num_index = sys.argv.index("-n")
    try:
        num = int(sys.argv[num_index + 1])
    except:
        print("-n has to have a following int to specify the number of generated tweets")
        exit(1)
if "-s" in sys.argv:
    start_index = sys.argv.index("-s")
    start = ""
    try:
        start = sys.argv[start_index + 1]
        if start[0] == "-" and len(start) == 2:
            raise ValueError
    except:
        print("-s has to have a following string to specify the staring string of generated tweets")
        exit(1)

import tensorflow as tf
import numpy as np

path_to_file = "../resources/TrumpTweetsText.txt"

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))

char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 280
examples_per_epoch = len(text) / (seq_length + 1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = sequences.map(split_input_target)

BATCH_SIZE = 64 if training else 1

BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


def generate_text(model, start_string=start, nums=num, addToLib=True, temperature=temp):
    lib_file = "./Tweets.txt"
    tweets = []
    for nums_i in range(nums):
        num_generate = 280
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        model.reset_states()
        for i in range(num_generate):
            predictions = model.predict(input_eval)

            predictions = tf.squeeze(predictions, 0)
            predictions = predictions / temperature

            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            input_eval = tf.expand_dims([predicted_id], 0)
            char = idx2char[predicted_id]
            if char == "ß":
                break
            text_generated.append(char)

        tweet = start_string + ''.join(text_generated)
        tweets.append(tweet)
        print(tweet)
    if not addToLib:
        return
    f = open(lib_file, "a")
    f.write("\n########### Temperature: " + str(temperature) + " - Starting-Chars: \"" + start_string + "\"\n")
    for line in tweets:
        f.write(line + "\n")
    f.close()


checkpoint_dir = './training_checkpoints'
if not training:
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    generate_text(model)
    exit(0)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


model.compile(optimizer='adam', loss=loss)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 5
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])