import json
import numpy as np
import os
from sklearn.externals import joblib
from azureml.core import Model
import joblib
import pickle


def init():
    global model
    model_path = Model.get_model_path('bestHpModel')
    model = joblib.load(model_path)


def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
