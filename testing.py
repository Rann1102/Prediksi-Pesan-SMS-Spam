# MELOAD program prediksi_spam.ipynb

from joblib import load

# Load the saved model and vectorizer
model = load('./model/model_prediksi_Pesanspam.joblib')
vectorizer = load('./model/vectorizer.joblib')

def predict_message(message):
    # Transform the input message using the loaded vectorizer
    message_transformed = vectorizer.transform([message])
    
    # Predict using the loaded model
    prediction = model.predict(message_transformed)
    
    # Return whether the message is ham or spam
    return 'Spam' if prediction[0] == 0 else 'Ham'

# Example usage
input_message = input("Tulis Pesan : ")
result = predict_message(input_message)
print(f"The message is predicted as: {result}")