# DjangoWebApp
The Formality Model is an NLP model recreated from the following paper
http://aclweb.org/anthology/Q16-1005
The purpose is to analyse text such as emails to rate the formality.

The MrAlfonso folder is a webapp created with Django that uses IBM's Bluemix Tone Analyser API in order to rate different text on their tones e.g. anger, sadness, joy, fear etc. The formality model is also incorporated.
https://tone-analyzer-demo.ng.bluemix.net/

In order to run the MrAlfonso web app follow these steps: 

1. Download the folder
2. Open Terminal / equivalent and go inside the MrAlfonso folder
3. Type command: python manage.py runserver
4. Wait for it to run and set up (may take 2 minutes)
5. Put the local URL given in your browser
