import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import tensorflow as tf

# Increase the field size limit
csv.field_size_limit(500 * 1024 * 1024)  # Set a larger limit if needed

# Step 1: Load and preprocess the data
true_data_path = r'D:\Bloodcast\archive\DataSet_Misinfo_TRUE.csv'
false_data_path = r'D:\Bloodcast\archive\DataSet_Misinfo_FAKE.csv'

# Read true articles data
true_data = []
with open(true_data_path, 'r', encoding='ISO-8859-1') as file:
    csv_reader = csv.reader(file, delimiter=";")
    for row in csv_reader:
        if len(row) >= 2:
            text = row[1].strip().replace("â", "").replace("â", "").replace("â", "'").replace("â", "-")
            true_data.append([row[0], text, 1])

# Read false articles data
false_data = []
with open(false_data_path, 'r', encoding='ISO-8859-1') as file:
    csv_reader = csv.reader(file, delimiter=",")
    for row in csv_reader:
        if len(row) >= 2:
            text = row[1].strip().replace("â", "").replace("â", "").replace("â", "'").replace("â", "-")
            false_data.append([row[0], text, 0])

# Convert to pandas DataFrame
true_df = pd.DataFrame(true_data, columns=["ID", "text", "label"])
false_df = pd.DataFrame(false_data, columns=["ID", "text", "label"])

# Concatenate the dataframes
data_df = pd.concat([true_df, false_df])

# Shuffle the data
data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display 10 random lines from the dataset
random_lines = data_df.sample(n=10)
print("Random Lines from the Dataset:")
#print(random_lines) #comented because it is crashing program if data is corrupted in some way

# Save the random lines to a file
random_lines.to_csv("123.txt", index=False)

# Extract text and labels
text_data = data_df["text"]
labels = data_df["label"]

# Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(text_data)

# Get indices and values of the sparse matrix
indices = features.nonzero()
values = features.data

# Convert to tf.SparseTensor
features = tf.sparse.SparseTensor(indices=np.column_stack(indices),
                                  values=tf.cast(values, tf.float32),
                                  dense_shape=features.shape)

# Reorder the sparse matrix
features = tf.sparse.reorder(features)

# Encode labels
label_encoder = LabelEncoder()

# Fit the label encoder on labels
label_encoder.fit(labels)

# Transform labels using the fitted encoder
labels = label_encoder.transform(labels)

# Step 2: Build the neural network model
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(features.get_shape()[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 3: Train the model
epochs = 6
batch_size = 32
model.fit(features, labels, epochs=epochs, batch_size=batch_size)

# Save the model
model.save('model.h5')

# Continuous prediction
loaded_model = load_model('model.h5')

while True:
    new_data = input("Enter new text data (or 'q' to quit): ")
    if new_data == 'q':
        break

    new_text_data = [new_data]

    # Convert text to numerical features using the existing vectorizer
    new_features = vectorizer.transform(new_text_data)

    # Get indices and values of the sparse matrix
    new_indices = new_features.nonzero()
    new_values = new_features.data

    # Convert to tf.SparseTensor
    new_features = tf.sparse.SparseTensor(indices=np.column_stack(new_indices),
                                          values=tf.cast(new_values, tf.float32),
                                          dense_shape=new_features.shape)

    # Reorder the sparse matrix
    new_features = tf.sparse.reorder(new_features)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(new_features)

    # Transform the prediction to the original label format
    predicted_label = label_encoder.inverse_transform([round(prediction[0][0])])

    print("Predicted Label:", predicted_label)
