
"""
Created on Fri Apr 10 13:49:44 2020

@author: ARKADIP GHOSH
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

print(r.json())