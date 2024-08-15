import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

col_names = ["feature {}".format(i) for i in range(0,61)]

df = pd.read_csv('sonar_data.csv',header=None)
'''
df.columns = col_names

df.head()

data = df.rename(columns={"feature 60":"y"})
data.head()

df

data.describe()

data.info()

data.y.value_counts()

data.groupby("y").mean()

data.y = pd.get_dummies(data.y)["R"].astype(int)

data

data.corr()
'''
# Separate features and target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Convert the target to categorical (one-hot encoding) if it's not numerical
y = pd.get_dummies(y).values

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=50)

# Reshape the data to fit the model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
'''
X[10:20]

X[1]

X_trains, X_tests, y_trains, y_tests = train_test_split(X, y, test_size=0.2, random_state=42)

X_tests=X_tests.reshape(X_tests.shape[0], X_tests.shape[1], 1)

X_tests[1].shape
'''
model = Sequential()

# Block 1

model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3))

model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3))

model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=3))

model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(y.shape[1], activation='sigmoid'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Fit the model
# History= model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), batch_size=64)
history = model.fit(X_train, y_train, epochs=50, shuffle=True, validation_data=(X_test, y_test), batch_size=64).history

model.save("cnn_model.h5")
# import pickle
# pickle_out = open("CNN.pkl","wb")
# pickle.dump(m, pickle_out)
# pickle_out.close()

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')

index = int(input('Enter a index number to pick the set of test data:'))
input_data = X_val[index]
#print(input_data)
input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

input_data_reshaped=input_data_reshaped.reshape(input_data_reshaped.shape[0], input_data_reshaped.shape[1], 1)

input_data_reshaped.shape

prediction = model.predict(input_data_reshaped)

print(prediction[0][0])

if prediction.shape:
    # Check if the element is equal to 1
    if prediction[0][0]> prediction[0][1]:
        print("It is a ROCK")
    else:
        print("It is a MINE")
else:
    # Raise an error if the prediction array is not a single element array
    raise ValueError("The prediction array must be a single element array")

# Calculate F1 score
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
f1 = f1_score(y_true, y_pred_classes, average=None)

print(f'Test F1 Score: {f1[0]}')

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Extract TP, TN, FP, FN
TN, FP, FN, TP = conf_matrix.ravel()

print("Confusion Matrix:")
print(conf_matrix)

# Create a table to display TP, TN, FP, FN
table_data = {
    '': ['True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'],
    'Count': [TP, TN, FP, FN]
}

table_df = pd.DataFrame(table_data)
print("\nTP, TN, FP, FN Table:")
print(table_df)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plotting training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

#print(y_true, y_pred_classes, y_pred)

#input_data = (0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)

