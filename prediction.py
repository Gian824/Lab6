from sklearn import datasets
from joblib import load
import numpy as np
import json

#load the model

my_model = load('svc_model.pkl')


iris_data = datasets.load_iris()
class_names = iris_data.target_names

def my_prediction(id):
    dummy = np.array(id)
    dummyT = dummy.reshape(1,-1)
    r = dummy.shape
    t = dummyT.shape
    r_str = json.dumps(r)
    t_str = json.dumps(t)
    prediction = my_model.predict(dummyT)
    name = class_names[prediction]
    name = name.tolist()
    name_str = json.dumps(name)
    str = [t_str, r_str, name_str]
    return str
    
#    this is an example text for me to use and practice my typing speed onto. This is a very slow typing speed for me and I am not too sure why my typing can be any slower.
 
