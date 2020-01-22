from flask import Flask, render_template, request
from chatbot import model,count_vect
from datacleaning import df
import numpy as np
app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    probability = model.predict_proba(count_vect.transform([userText]))
    
    if np.amax(probability) > 0.15:
        result = model.predict(count_vect.transform([userText])) 
        ans = df['answers'][result].item()
        return str(ans.capitalize())
    else:
        answer = "Sorry i didn't get you! for detail call on phone : +92 21 99217501-3"
        return str(answer.capitalize())
if __name__ == "__main__":
    app.run(debug=True)