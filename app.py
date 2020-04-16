
"""
Created on Fri Apr 10 13:48:41 2020

@author: ARKADIP GHOSH
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    #query_index = np.random.choice(movie_features_df.shape[0])
    query_index=request.form['Name of the movie']
    for i in range(0,len(model1)) : 
      if(model1.index[i]==query_index) :
          break
    distances, indices = model.kneighbors(model1.iloc[i,:].values.reshape(1, -1), n_neighbors = 11)

    output=['']*11
    
    if(i<9718):
        for j in range(0, len(distances.flatten())):
            if j == 0:
                output[j]='Top 10 Recommendations for {0}:\n'.format(model1.index[i])
            else:
                output[j]='{0}: {1}'.format(j, model1.index[indices.flatten()[j]])

        
        
        
        return render_template('index.html',prediction_text0=" Nice Choice:)",prediction_text1= output[0],prediction_text2=output[1],prediction_text3=output[2],prediction_text4=output[3],prediction_text5=output[4],prediction_text6=output[5],prediction_text7=output[6],prediction_text8=output[7],prediction_text9=output[8],prediction_text10=output[9],prediction_text11=output[10])

        
        
    else:
        a=('oops! No such movie found')
    
        return render_template('index.html', prediction_text1=a)
    
    
    
    #return render_template('index.html', prediction_text1=a,prediction_text2=b,prediction_text3=c,prediction_text4=d,prediction_text5=e,prediction_text6=f)



@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)
