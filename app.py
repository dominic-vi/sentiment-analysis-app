# import objects from the Flask model
import io
import random
import torch
import transformers
from flask import Response
from flask import Flask, render_template, request
from underthesea import word_tokenize


# creating flask app
app = Flask(__name__)

# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {
    "PhoBERT": transformers.RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment"),
} # feel free to add several models

#initialize tokenizer 
tokenizer = transformers.AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

listOfKeys = []
for key in dictOfModels :
        listOfKeys.append(key) 

# inference fonction
def get_prediction(text,model):
    # word_segment text
    text = word_tokenize(text, format="text")
    #encode text with tokenizer
    input_ids = torch.tensor([tokenizer.encode(text)])
    #convert them into a list of sentiment result
    with torch.no_grad():
        out = model(input_ids)
        list = out.logits.softmax(dim=-1).tolist()
    #convert them into result
    named_result = {
      'Negative': list[0][0],
      'Positive': list[0][1],
      'Neutral' : list[0][2],
      }
    return named_result

# get method
@app.route('/', methods=['GET'])
def get():
    # in the select we will have each key of the list in option
    return render_template("home.html", len = len(listOfKeys), listOfKeys = listOfKeys)

def translate(score):
    if score == 'Positive':
        result = "tích cực"
    elif score == 'Negative':
        result = 'tiêu cực'
    else:
        result = 'trung tính'
    return result 

# post method
@app.route('/', methods=['POST'])
def predict():
    message = request.form['message']
    # choice of the model
    results = get_prediction(message, dictOfModels[request.form.get("model_choice")])
    print(f'User selected model : {request.form.get("model_choice")}')
    my_prediction = f'Văn bản của bạn có cảm xúc {translate(max(results, key=results.get))} với tỉ lệ là {round(max(results.values())*100,2)}%'
    print(results)
        
    return render_template('result.html', text=f'{message}', prediction=my_prediction, POS = round(results['Positive'],5), NEU = round(results['Neutral'],5), NEG = round(results['Negative'],5))


if __name__ == '__main__':
    # starting app
    app.run(debug=True, host='0.0.0.0')
