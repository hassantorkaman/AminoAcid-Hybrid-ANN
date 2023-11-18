
from tensorflow.keras.models import load_model

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
