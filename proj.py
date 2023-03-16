
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
data = pd.read_csv('waterQuality1.csv')
data=data.replace('#NUM!',0)
data['is_safe'] = pd.to_numeric(data['is_safe'])
X = data.drop(['is_safe'], axis = 1).values
y = data['is_safe'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

from sklearn.metrics import classification_report
from sklearn import metrics
acc = []
model = []
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(X_train,y_train)

predicted_values = RF.predict(X_test)

x = metrics.accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('RF')

RF_pkl_filename='./RandomForest.pkl'
RF_Model_pkl=open(RF_pkl_filename,'wb')
pickle.dump(RF,RF_Model_pkl)

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
app=Flask(__name__)
water_recommendation_model_path = 'RandomForest.pkl'
water_recommendation_model = pickle.load(open(water_recommendation_model_path, 'rb'))
@ app.route('/home')
def home():
    title='Water Safety Detection System'
    return render_template('form2.html',title=title)


@app.route('/water-predict', methods=['POST'])
def water_prediction():
    title = 'Water Safety Detection System'

    if request.method == 'POST':
        aluminium = float(request.form['aluminium'])
        ammonia = float(request.form['ammonia'])
        barium = float(request.form['barium'])
        arsenic = float(request.form['arsenic'])
        cadmium = float(request.form['cadmium'])
        chloramine = float(request.form['chloramine'])
        chromium = float(request.form['chromium'])
        copper = float(request.form['copper'])
        flouride = float(request.form['flouride'])
        bacteria = float(request.form['bacteria'])
        viruses = float(request.form['viruses'])
        lead = float(request.form['lead'])
        nitrites = float(request.form['nitrites'])
        nitrates = float(request.form['nitrates'])
        mercury = float(request.form['mercury'])
        perchlorate = float(request.form['perchlorate'])
        radium = float(request.form['radium'])
        selenium = float(request.form['selenium'])
        silver = float(request.form['silver'])
        uranium = float(request.form['uranium'])
        data = np.array([[aluminium, ammonia, barium, arsenic, cadmium, chloramine, copper, flouride, bacteria, viruses,
                          lead, nitrites, mercury, perchlorate, radium, selenium, silver, uranium]])
        my_prediction = water_recommendation_model.predict(data)
        final_prediction = my_prediction[0]
        output = final_prediction
        if output == [1]:
            return render_template('form2.html', final_prediction="Its  safe to drink")
        else:
            return render_template('form2.html', final_prediction="Its not safe to drink")
if __name__ == '__main__':
    app.run(debug=False)