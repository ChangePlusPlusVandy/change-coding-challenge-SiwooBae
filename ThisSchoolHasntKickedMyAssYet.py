import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from Transformer_Decoder import *
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.keras.callbacks import TensorBoard
from bpe import Encoder
import time
import random
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

##########################################################
#
# Who said Vanderbilt Engineering had a lot of workload??????????
# It couldn't even stop me from creating a deep neural network
# Allow me to introduce you the Elon "Cyber" Musk
# This trains a neural network with 10000 Elon Musk tweets and simulates its writing style.
# You will need tensorflow 2.0 and all the packages above to run this program. Good Luck
#
##########################################################

elon = []
raw_csv = pd.read_csv("./elonmusk.csv")['tweet']
for tweet in raw_csv:
    if not (("http://" in tweet) or ("https://" in tweet) or (".com" in tweet) or ("@" in tweet)):
        elon.append((tweet.lower()).encode("utf8").decode("ascii", 'ignore'))


def group_by_sentence(corp):
    sentence_stack = []
    sentences = []
    for line in corp:
        sentence_stack.append(line)
        if sentence_stack[-1][-1] == ".":  # if the last txt ended a sentence
            sentences.append(" ".join(sentence_stack))
            sentence_stack.clear()
    return sentences


def chunk_sentences(data, chunk_size):
    temp = []
    for i in range(len(data) - chunk_size):
        chunk = " ".join(data[i:i + chunk_size])
        temp.append(chunk)
    return temp


print(len(elon))
tokenizer_en = Encoder(vocab_size=8192, ngram_max=5)
tokenizer_en.fit(elon)

num_words = tokenizer_en.vocab_size
vocab_size = tokenizer_en.vocab_size + 2

max_len = len(next(tokenizer_en.transform([max(elon, key=len)])))


def encode(lang):
    lang = next(tokenizer_en.transform([lang]))
    lang = [num_words] + lang + [num_words + 1]
    return lang


elon = [encode(tweet) for tweet in elon]  # encode the whole dataset

plt.hist([len(sen) for sen in elon])
plt.show()

plt.hist([num for tweet in elon for num in tweet])  # this line is so disgusting but okay
plt.show()


def batch_pad(data):
    max_length = max([len(poem) for poem in data])
    for i in range(len(data)):
        data[i] = data[i] + [0 for _ in range(max_length - len(data[i]))]
    return np.array(data)


batch_size = 16


def create_batch(data, batch_size):
    new = []
    i = 0
    while i < len(data) - batch_size:
        new.append(data[i:i + batch_size])
        i += batch_size
    return new


sentences = batch_pad(elon)
sentences = create_batch(sentences, batch_size)

#################################################

num_layers = 4
d_model = 64
d_feedforward = 256
num_heads = 8
dropout_rate = 0.1

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


transformer = Transformer_Decoder(num_layers, d_model, num_heads, d_feedforward, max_len=max_len + 300,
                                  target_vocab_size=vocab_size,
                                  rate=dropout_rate)
# Visualize Model


checkpoint_path = "./SavedModel/CyberElonMusk"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
    print('Latest checkpoint restored!!')

EPOCHS = 30

train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64)]


@tf.function(input_signature=train_step_signature)
def train_step(tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    with tf.GradientTape() as tape:
        predictions, _ = transformer(tar_inp, training=True)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


# train phase. uncomment to train.
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    random.shuffle(sentences)
    for (batch, tar) in enumerate(sentences):
        train_step(tar)
        if batch % 50 == 0:
            print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


def evaluate(inp_sentence):
    start_token = [num_words]

    inp_sentence = start_token + next(tokenizer_en.transform([inp_sentence]))
    input = tf.expand_dims(inp_sentence, 0)

    for i in range(max_len):

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(input, False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :].numpy()  # (batch_size, 1, vocab_size)

        # predictions[:, -1:, 1] = -20

        # plt.plot(np.squeeze(predictions))
        # plt.show()

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # this part makes sure that the AI does not pick the same word twice
        # while np.any(np.equal(input, np.squeeze(predicted_id))): #and not np.any(np.equal(mfw_ids, np.squeeze(predicted_id))):
        #     id = np.squeeze(predicted_id)
        #     predictions[:, -1:, id] = -20
        #     predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if np.squeeze(predicted_id) == num_words + 1:
            return tf.squeeze(input, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        input = tf.concat([input, predicted_id], axis=-1)

    return tf.squeeze(input, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))

    sentence = tokenizer_en.encode(sentence)

    attention = tf.squeeze(attention[layer], axis=0)

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(result) - 1.5, -0.5)

        ax.set_xticklabels(
            ['<start>'] + [tokenizer_en.decode([i]) for i in sentence] + ['<end>'],
            fontdict=fontdict, rotation=90)

        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                            if i < tokenizer_en.vocab_size],
                           fontdict=fontdict)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()


def compose(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = next(tokenizer_en.inverse_transform([result[1:].numpy()]))

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)


compose("shorted")
