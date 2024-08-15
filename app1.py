import numpy as np
import pickle
import pandas as pd
from random import randint
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, MaxPooling1D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

#from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in_logistic = open("logistic_reg.pkl","rb")
logistic_reg = pickle.load(pickle_in_logistic)

#pickle_in_CNN = open("CNN.pkl","rb")
#cnn_model = pickle.load(pickle_in_CNN)
loaded_model = load_model("sonar_model_latest.h5")

def predict_logistic(sonar_data):
    X = sonar_data.drop(columns=60, axis=1)
    Y = sonar_data[60]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)
    logistic_reg.fit(X_train, Y_train)
    X_train_prediction = logistic_reg.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    
    X_test_prediction = logistic_reg.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 
    #print('Accuracy on test data : ', test_data_accuracy)
    
    X_test_np = np.asarray(X_test)
    X_test_np_len = X_test_np.shape[0]
    X_test_1 = X_test_np[randint(0, X_test_np_len)].reshape(1, -1)
    prediction=logistic_reg.predict(X_test_1)
    #print(prediction)
    return training_data_accuracy, test_data_accuracy, prediction

def predict_CNN(sonar_data):
    # Enable eager execution
    tf.config.run_functions_eagerly(True)

    # Separate features and target
    X = sonar_data.iloc[:, :-1].values
    y = sonar_data.iloc[:, -1].values

    # Convert target values to numerical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Convert numerical labels to float32
    y = y.astype(np.float32)

    # Normalize input data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=50)
    
    # X = sonar_data.iloc[:, :-1].values
    # y = sonar_data.iloc[:, -1].values

    # Convert the target to categorical (one-hot encoding) if it's not numerical
    # y = pd.get_dummies(y).values
    
    # Splitting the dataset into training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=50)

    # Reshape the data to fit the model
    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    #history = cnn_model.fit(X_train, y_train, epochs=50, shuffle=True, validation_data=(X_test, y_test), batch_size=64).history
    
    loss, accuracy = loaded_model.evaluate(X_test, y_test)
    st.text('Accuracy on testing data: {}'.format(accuracy))
    st.text('Loss on testing data: {}'.format(loss))
    
    X_val_np = np.asarray(X_val)
    X_val_np_len = X_val_np.shape[0]
    input_data = X_val[randint(0,X_val_np_len)]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)   
    input_data_reshaped=input_data_reshaped.reshape(input_data_reshaped.shape[0], input_data_reshaped.shape[1], 1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction.shape[1] == 1:
        # For binary classification with sigmoid activation, prediction array has shape (n_samples, 1)
        # Here, prediction[0][0] represents the probability of class 1
        if prediction[0][0] > 0.5:
            output =  "It is a MINE"
        else:
            output =  "It is a ROCK"
    else:
        # For multi-class classification with softmax activation, prediction array has shape (n_samples, n_classes)
        # You can use argmax to find the predicted class
        predicted_class = np.argmax(prediction[0])
        if predicted_class == 0:
            output = "It is a MINE"
        else:
            output =  "It is a ROCK"
    
    # y_pred = loaded_model.predict(X_val)
    # y_pred_classes = np.argmax(y_pred, axis=1)
    # y_true = np.argmax(y_val, axis=1)

    # # Compute F1 score
    # f1 = f1_score(y_true, y_pred_classes, average='macro')
    y_pred = loaded_model.predict(X_test)
    y_pred_classes = np.round(y_pred).astype(int)
    f1 = f1_score(y_test, y_pred_classes)
    st.text('F1 score: {}'.format(f1))

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    st.text('Confusion Matrix: {}'.format(cm))
    return output

    #start
    #input_data = X_val[1]
    #print(input_data)
    # input_data_as_numpy_array = np.asarray(input_data)
    # input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    # input_data_reshaped=input_data_reshaped.reshape(input_data_reshaped.shape[0], input_data_reshaped.shape[1], 1)
    # prediction = loaded_model.predict(input_data_reshaped)
    #print(prediction[0][0])
    
    # if prediction.shape:
    #     # Check if the element is equal to 1
    #     if prediction[0][0]> prediction[0][1]:
    #         #print("It is a ROCK")
    #         st.text("It is a ROCK")
    #     else:
    #         st.text("It is a MINE")
    # else:
    #     # Raise an error if the prediction array is not a single element array
    #     raise ValueError("The prediction array must be a single element array")

def main():
    #st.title("Underwater Rock or Mine Prediction")
    html_temp = """
    <div style="background-color:orange;padding:10px">
    <h2 style="color:white;text-align:center;">Underwater Rock or Mine Prediction Using Machine Learning </h2>
    <h3 style="color:white;text-align:right;">SHSU </h3>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    data = st.file_uploader('Upload the data file')
    prediction = ""
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Predict using Logistic Regression"):
                sonar_data = pd.read_csv(data, header = None)
                #X_test_np = np.asarray(sonar_data)
                #st.text(X_test_np)
                #X_test_1 = X_test_np[0].reshape(1, -1)
                #st.text(X_test_1)
                #X_test = np.reshape(X_test_1, (1, -1))
                #st.text(X_test)
                #X_test_1 = X_test_np.reshape(1, -1)
                #st.dataframe(sonar_data[1:3])        
                
                training_data_accuracy, test_data_accuracy, prediction = predict_logistic(sonar_data)
                
                st.text('Accuracy on training data: {}'.format(training_data_accuracy))
                st.text('Accuracy on testing data: {}'.format(test_data_accuracy))
                if (prediction[0]=='R'):
                    st.success('The object is a Rock')
                else:
                    st.success('The object is a Mine')       
            #st.success('The output is {}'.format(prediction))
        with col2:
            if st.button("Predict using CNN"):
                sonar_data = pd.read_csv(data, header = None)
                prediction = predict_CNN(sonar_data)
                st.success(prediction)
                
if __name__=='__main__':
    main()
    
    
    
    