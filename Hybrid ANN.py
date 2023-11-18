import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import io
import datetime
from sklearn.metrics import accuracy_score, classification_report


final_df = pd.read_csv('final_csv.csv')

#1-2 Visualization of the Raw Dataset:

#1-2-1 Take a look at the dataset:
attribute_columns = ['AT', 'DPX', 'ASA', 'Average Bfactor', 'CX', 'RMSF', 'kdHydrophobicity', 'label']

correlation_matrix = final_df[attribute_columns].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Attributes')
plt.show()

# Define X, Y 
x= final_df.iloc[: ,3:10]
y= final_df.iloc[:, -1]



#Hybrid ANN

#PreProcessing phase: applying smote + Normalization+ PCA
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


smote = SMOTE(random_state=16)
X_smote_dnn, y_smote_dnn = smote.fit_resample(x, y)

# Standardize the data before applying PCA
scaler = MinMaxScaler(feature_range = (0,1))
X_scaled = scaler.fit_transform(X_smote_dnn)


# Applying PCA
num_components = 4
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_scaled)
# X_pca now contains the transformed features after applying SMOTE and PCA

#spliting DATA
X_train_pca, X_test_pca, y_train_dnn, y_test_dnn = train_test_split(X_pca,y_smote_dnn,test_size=0.2, random_state=16)


#Creating the ANN model 
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6,activation = 'relu'))
#ann.add(Dropout(rate=0.3))
ann.add(tf.keras.layers.Dense(units=50,activation = 'relu'))
#ann.add(Dropout(rate=0.3))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
#ann.add(Dropout(rate=0.3))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
#ann.add(Dropout(rate=0.3))
ann.add(tf.keras.layers.Dense(units=50, activation='relu'))
#ann.add(Dropout(rate=0.3))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#ann.add(Dropout(rate=0.3))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train_pca,y_train_dnn, batch_size=12,epochs=250, validation_data=(X_test_pca, y_test_dnn))

prediction = ann.predict(X_test_pca)
prediction_dnn = [1 if p > 0.5 else 0 for p in prediction]

print(classification_report(y_test_dnn, prediction_dnn))
accuracy_score(y_test_dnn, prediction_dnn)

# Save the trained model:
import os
from tensorflow.keras.models import load_model
project_path = os.getcwd()
model_save_path = os.path.join(project_path, 'trained_hybrid_ann')
ann.save(model_save_path)
loaded_model = load_model(model_save_path)


#predict a sample by user's inputs:

def predict_label(AT, DPX, ASA, Average_Bfactor, CX, RMSF, kdHydrophobicity):
    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'AT': [AT],
        'DPX': [DPX],
        'ASA': [ASA],
        'Average Bfactor': [Average_Bfactor],
        'CX': [CX],
        'RMSF': [RMSF],
        'kdHydrophobicity': [kdHydrophobicity]
    })

    user_input_array = user_input.values
    user_input_normalized = scaler.transform(user_input_array)
    user_input_pca = pca.transform(user_input_normalized)

    # Use the trained model for prediction
    prediction1 = loaded_model.predict(user_input_pca)
    prediction_dnn = [1 if p > 0.5 else 0 for p in prediction1]

    # Output the prediction
    if prediction_dnn[0] == 1:
        print("The model predicts a positive label (1).")
    else:
        print("The model predicts a negative label (0).")



# get new values by user
AT = float(input("Enter AT value: "))
DPX = float(input("Enter DPX value: "))
ASA = float(input("Enter ASA value: "))
Average_Bfactor = float(input("Enter Average Bfactor value: "))
CX = float(input("Enter CX value: "))
RMSF = float(input("Enter RMSF value: "))
kdHydrophobicity = float(input("Enter kdHydrophobicity value: "))

# Predict label based on user input
predict_label(AT, DPX, ASA, Average_Bfactor, CX, RMSF, kdHydrophobicity)

