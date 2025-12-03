from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    pH = float(request.form['pH'])
    EC = float(request.form['EC'])
    
    data = np.array([[N, P, K, pH, EC]])
    prediction = model.predict(data)[0]
    
    return render_template('result.html', result=prediction)

if __name__ == '__main__':
    app.run(debug=True)
