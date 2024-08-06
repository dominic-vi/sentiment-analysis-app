# import objects from the Flask model
import torch
import torch.nn as nn
import transformers
from flask import Response
from flask import Flask, render_template, request
from underthesea import word_tokenize


# creating flask app
app = Flask(__name__)

#Class for custom PhoBERT
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.bert.config.hidden_size, out_features=512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(in_features=128, out_features=3),
            nn.Softmax(1)
        )
    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        x = self.drop(output)
        x = self.fc(x)
        return x

#MODEL_PATH to PhoBERT fine tuned on Vietnamese sentiment dataset
MODEL_PATH = r"weights\vn_sentiment_best.pt"

# create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
dictOfModels = {
    "Phở BERT": transformers.RobertaForSequenceClassification.from_pretrained("wonrax/phobert-base-vietnamese-sentiment"),
    "Phở Nhà Làm": torch.load(MODEL_PATH, map_location=torch.device('cpu')),
}

#initialize tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained("wonrax/phobert-base-vietnamese-sentiment", use_fast=False)

listOfKeys = []
for key in dictOfModels :
        listOfKeys.append(key)

# inference function for PhoBERT
def get_prediction_PhoBERT(text,model):
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

#inference function for custom PhoBERT
def get_prediction_custom_Pho(text,model):
    # word_segment text
    text = word_tokenize(text, format="text")
    #encode text with tokenizer
    batch = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=60,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
    #convert them into a list of sentiment result
    with torch.no_grad():
        output = model(batch['input_ids'], batch['attention_mask'])
        list = output.tolist()
    #convert them into result
    named_result = {
      'Negative': list[0][0],
      'Neutral': list[0][1],
      'Positive' : list[0][2],
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
    if request.form.get("model_choice") == "Phở BERT":
        results = get_prediction_PhoBERT(message, dictOfModels[request.form.get("model_choice")])
        print(f'User selected model : {request.form.get("model_choice")}')
        my_prediction = f'Văn bản của bạn có cảm xúc {translate(max(results, key=results.get))} với tỉ lệ là {round(max(results.values())*100,2)}%'
        print(results)
        return render_template('result.html', text=f'{message}', prediction=my_prediction, POS = round(results['Positive'],5), NEU = round(results['Neutral'],5), NEG = round(results['Negative'],5))

    #custom PhoBERT
    elif request.form.get("model_choice") == "Phở Nhà Làm":
        results = get_prediction_custom_Pho(message, dictOfModels[request.form.get("model_choice")])
        print(f'User selected model : {request.form.get("model_choice")}')
        my_prediction = f'Văn bản của bạn có cảm xúc {translate(max(results, key=results.get))} với tỉ lệ là {round(max(results.values())*100,2)}%'
        print(results)
        return render_template('result.html', text=f'{message}', prediction=my_prediction, POS = round(results['Positive'],5), NEU = round(results['Neutral'],5), NEG = round(results['Negative'],5))

if __name__ == '__main__':
    # starting app
    app.run(debug=True, host='0.0.0.0')
