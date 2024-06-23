import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TextVectorization, LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
from matplotlib import pyplot as plt


data = pd.read_csv('test_dataframe.csv', nrows=40000)


comments = data['comment_text']
labels = data[data.columns[3:]].values

MAX_VOCAB_SIZE = 200000  

text_vectorizer = TextVectorization(max_tokens=MAX_VOCAB_SIZE, output_sequence_length=1800, output_mode='int')
text_vectorizer.adapt(comments.values)

vectorized_comments = text_vectorizer(comments.values)


tf_dataset = tf.data.Dataset.from_tensor_slices((vectorized_comments, labels))
tf_dataset = tf_dataset.cache()
tf_dataset = tf_dataset.shuffle(160000)
tf_dataset = tf_dataset.batch(16)
tf_dataset = tf_dataset.prefetch(8)  

train_dataset = tf_dataset.take(int(len(tf_dataset) * .7))
val_dataset = tf_dataset.skip(int(len(tf_dataset) * .7)).take(int(len(tf_dataset) * .2))
test_dataset = tf_dataset.skip(int(len(tf_dataset) * .9)).take(int(len(tf_dataset) * .1))


model = Sequential()
model.add(Embedding(MAX_VOCAB_SIZE + 1, 32))
model.add(Bidirectional(LSTM(32, activation='tanh')))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='sigmoid'))

model.compile(loss='BinaryCrossentropy', optimizer='Adam')
model.summary()


model = tf.keras.models.load_model('model_weight.h5')
history = model.fit(train_dataset, epochs=5, validation_data=val_dataset)


model.save('model_weight.h5')


plt.figure(figsize=(8, 5))
pd.DataFrame(history.history).plot()
plt.show()


sample_text = text_vectorizer('I hate you! You are an idiot!')
prediction = model.predict(np.expand_dims(sample_text, 0))
print((prediction > 0.5).astype(int))


test_batch_X, test_batch_y = test_dataset.as_numpy_iterator().next()
print((model.predict(test_batch_X) > 0.5).astype(int))


output_labels = ['toxic', 'very toxic', 'obscene', 'threat', 'insult', 'identity_hate']
precision_metrics = {label: Precision() for label in output_labels}
recall_metrics = {label: Recall() for label in output_labels}
accuracy_metrics = {label: CategoricalAccuracy() for label in output_labels}

for batch in test_dataset.as_numpy_iterator():
    true_X, true_y = batch
    predictions = model.predict(true_X)

    for i, label in enumerate(output_labels):
        true_label_output = true_y[:, i]
        pred_label_output = predictions[:, i]

        precision_metrics[label].update_state(true_label_output, pred_label_output)
        recall_metrics[label].update_state(true_label_output, pred_label_output)
        accuracy_metrics[label].update_state(true_label_output, pred_label_output)

for label in output_labels:
    precision = precision_metrics[label].result().numpy()
    recall = recall_metrics[label].result().numpy()
    accuracy = accuracy_metrics[label].result().numpy()

    print(f"{label}:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
