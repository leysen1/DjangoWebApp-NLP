#importing required packages
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
import requests
import json
from .formality_model import get_model_score
from .formality_model import clf
from sklearn.linear_model import Ridge,LinearRegression,BayesianRidge

#disabling csrf (cross site request forgery)
@csrf_exempt
def index(request):

    anger = 0
    fear = 0
    joy = 0
    sadness = 0
    analytical = 0
    confident = 0
    tentative = 0
    formal = 0

    #if post request came
    if request.method == 'POST':
        #getting values from post
        name = request.POST.get('name')
        
        tone_result = analyze_tone(name)
        tone_result = json.loads(tone_result)
        
        print("formality here")
        formal_result = get_model_score(name,clf)
        formal = int(round((formal_result[0]+3)/6*100))
        
        for key in tone_result['document_tone']['tones']:
            if key['tone_name'] == "Anger":
                anger = round(key['score']*100)
            elif key['tone_name'] == "Fear":
                fear = round(key['score']*100)
            elif key['tone_name'] == "Joy":
                joy = round(key['score']*100)
            elif key['tone_name'] == "Sadness":
                sadness = round(key['score']*100)
            elif key['tone_name'] == "Analytical":
                analytical = round(key['score']*100)
            elif key['tone_name'] == "Confident":
                confident = round(key['score']*100)
            elif key['tone_name'] == "Tentative":
                tentative = round(key['score']*100)
        
        print(anger, fear, joy, sadness, analytical, confident, tentative)

        #adding the values in a context variable
        context = {
            'anger': anger,
            'fear': fear,
            'joy': joy,
            'sadness': sadness,
            'analytical': analytical,
            'confident': confident,
            'tentative': tentative,
            'formal': formal,
            
        }
        #getting our showdata template
        template = loader.get_template('showdata.html')
            

        #returing the template
        return HttpResponse(template.render(context, request))
    else:
        #if post request is not true
        #returing the form template
        template = loader.get_template('index.html')
        return HttpResponse(template.render())

# function to analyse tones from IBM
def analyze_tone(text):
        
    username = "d394d72b-c5d8-4388-a4a5-1878cf729898"
    password = "VLYeoaHJlujV"
    watsonUrl = 'https://gateway.watsonplatform.net/tone-analyzer/api/v3/tone?version=2017-09-21'
    headers = {"content-type": "text/plain"}
    data = text
    try:
        r = requests.post(watsonUrl, auth=(username,password),headers = headers,
         data=data)
        return r.text
    except:
        return False
