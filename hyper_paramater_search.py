
from model import sequence_model
import torch
import os
import json
import numpy as np
from skopt.space import Integer,Real
from skopt.utils import use_named_args
from skopt import gp_minimize
import json

hyperparameter = {
                      'model_version' : 'Test',
                      "frezze": False,
                      "batch_size": 15,
                      "loss_weight": None,
                      "max_norm": 1.0,
                      "epochs": 10,
                      "learning_rate": 0.0001,
                      "weight_decay" : 0.01,
                      "warmup_steps": 6
                      }

learning_rate = Real(low=1e-5, high=1e-4, prior='log-uniform', name='learning_rate')
weight_decay = Real(low=1e-2, high=1e-1, prior='log-uniform', name='weight_decay')
search_space = [learning_rate,
                weight_decay
                ]

@use_named_args(search_space)
def evaluate_model(weight_decay,learning_rate):
    model = sequence_model()
    print("learning_rate : ",learning_rate)
    print("weight_decay : ",weight_decay)
    f1_score = model.train(train_batch_size=hyperparameter['batch_size'],
                max_norm=hyperparameter['max_norm'],
                epochs=hyperparameter['epochs'],
                frezze=hyperparameter['frezze'],
                learning_rate=learning_rate,
                warmup_steps=hyperparameter['warmup_steps'],
                model_version=hyperparameter['model_version'],
                weight=hyperparameter['loss_weight'],
                weight_decay=weight_decay,
                early_stop=True,
                save_performance = False,
                save_model = False,
                validation_batch_size = 128
                )

    return 1.0 - f1_score

result = gp_minimize(evaluate_model, search_space, n_calls=20)
# summarizing finding:
print('Best f1_score: %.3f' % (1.0 - result.fun))
print('Best Parameters: learning_rate=%.3f, weight_decay=%.3f' % (result.x[0], result.x[1]))

hyperparameterlog = {}
for i,(h,f) in enumerate(zip(result.x_iters,result.func_vals)):
    hyperparameterlog[f'{i}_times_search'] = {'learning_rate' : h[0],
                                              'weight_decay' : h[1],
                                              'f1_score' : 1.0 - f}
    

with open(f'./result/hyperparamaterlog.json', 'w', newline='') as jsonfile:
        json.dump(hyperparameterlog,jsonfile,indent = 4)