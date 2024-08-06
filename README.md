## A Simple Sentiment Classification App using a finetuned PhoBERT

### Install modules needed to run the app
>pip install -r requirements.txt

### To run the application you're going to need to need the PhoBERT's finetuned weight file added to the weight directory
It should look something like this (if you don't have a weight directory feel free to create one)
``
+-- static
|   +-- style.css
+-- templates
|   +-- home.html
|   +-- result.html
+-- weights
|   +-- place your weight file (.pt) here
+-- .gitignore
+-- app.py
+-- requirements.txt
``