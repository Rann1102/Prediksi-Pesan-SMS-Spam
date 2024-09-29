from flask import Flask, request, render_template
from joblib import load

app = Flask(__name__)

# Load the model and vectorizer
model = load('./model/model_prediksi_Pesanspam.joblib')
vectorizer = load('./model/vectorizer.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        message = request.form['message']
        message_transformed = vectorizer.transform([message])
        prediction = model.predict(message_transformed)
        result = 'Spam' if prediction[0] == 0 else 'Ham'
        return render_template('index.html', prediction_text=f'The message is predicted as: {result}')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)