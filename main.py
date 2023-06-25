from flask import Flask
from flask import request, Response
import json

import numpy as np

from nn_builder import NNBuilder

app = Flask(__name__)

build = NNBuilder
X = np.asarray([[-0.41675785, -0.05626683, -2.1361961], [1.64027081, -1.79343559, -0.84174737]]) 
Y = np.asarray([[.3, .4, .9]])

@app.route('/train_neural_network')
def train_neural_network():

    n_epochs = request.args.get('n_epochs')

    for i in n_epochs:
        parameters = build.initialise_parameters()
        Y1, Y2 = build.forward_propagation(X, Y, parameters)
        loss = build.cross_entropy_loss(Y2, Y)
        gradients = build.back_propagation(X, Y, parameters, Y1, Y2)
        parameters = build.update_parameters(parameters, gradients)

        para_as_str = json.dumps(parameters, default=str)
    
    return Response(response=para_as_str, status=200, mimetype="application/json")