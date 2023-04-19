from django.shortcuts import render
from transformers import AutoModelForSequenceClassification, BertTokenizerFast
import torch

@torch.no_grad()
def predict_4(text):
    tokenizer = BertTokenizerFast.from_pretrained(
        '/home/mit34/Downloads/rubert_4poi_blanchefort_rubert-base-cased-sentiment-rusentiment/rubert-4poi')
    model = AutoModelForSequenceClassification.from_pretrained(
        '/home/mit34/Downloads/rubert_4poi_blanchefort_rubert-base-cased-sentiment-rusentiment/rubert-4poi',
        return_dict=True)
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = predicted.tolist()[0]
    return predicted

@torch.no_grad()
def predict_5(text):
    tokenizer = BertTokenizerFast.from_pretrained(
        '/home/mit34/Downloads/rubert_5poi_blanchefort_rubert-base-cased-sentiment-rusentiment/rubert-5poi')
    model = AutoModelForSequenceClassification.from_pretrained(
        '/home/mit34/Downloads/rubert_5poi_blanchefort_rubert-base-cased-sentiment-rusentiment/rubert-5poi',
        return_dict=True)
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = predicted.tolist()[0]
    return predicted

def sentiment(request):
    result = None  # define result variable with initial value None
    if request.method == 'POST':
        text = request.POST.get('text')
        model = request.POST.get('model')  # get selected model
        if model == '4':
            result = predict_4(text)  # use predict_4 function for model 4
        elif model == '5':
            result = predict_5(text)  # use predict_5 function for model 5
    context = {'result': result}
    return render(request, 'website/api.html', context)
