"""
訓練bert模型
可繼續訓練已訓練bert模型
"""

from model import sequence_model
import torch
import os


PRETRAINED_MODEL_NAME = "hfl/chinese-bert-wwm-ext"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model_version = input('model_version : ')
    model_name = f'model_v{model_version}.pt'
    weight = [
    10/2,#乳房外科
    10/1,#大腸直腸外科
    10/6,#婦產科
    10/2,#兒科
    10/5,#復健科
    10/3,#心臟科
    10/2,#感染科
    10/2,#新陳代謝科
    10/12,#泌尿科
    10/1,#牙科
    10/14,#皮膚科
    10/2,#眼科
    10/1,#神經內/外科
    10/10,#耳鼻喉科
    10/2,#胸腔科
    10/1,#腎臟內科
    10/3,#肝膽腸胃科
    10/2,#血液腫瘤科
    10/4,#身心科
    10/1,#免疫風濕科
    10/4,#骨科
    10/1,#口腔顎面外科
    10/2,#一般外科
    10/6,#家醫科          
    10/1#中醫科
    ]

    hyperparameter = {"model_version": model_version,
                      'PRETRAINED_MODEL_NAME': PRETRAINED_MODEL_NAME,
                      "frezze": False,
                      "batch_size": 15,
                      "loss_weight": weight,
                      "max_norm": 1.0,
                      "epochs": 25,
                      "learning_rate": 0.00001,
                      "weight_decay" : 0.05,
                      "warmup_steps": 6
                      }

    model = sequence_model()

    if model_name in os.listdir('./model'):
        print('='*10)
        print(f'retrain model {model_name}')
        print('=' * 10)
        model.load_model(model_version)
        '''
        try:
            with open(f'./model/model_log/modelv{model_version}.txt') as file:
                for lines in file:
                    line = lines.split(' : ')
                    if line == [' ']:
                        break
                    key = line[0]
                    val = line[1][:-1]
                    if key == 'loss_log' :
                        loss_log = val
                model.train(max_norm=hyperparameter['max_norm'],
                            epochs=hyperparameter['epochs'],
                            frezze=hyperparameter['frezze'],
                            learning_rate=hyperparameter['learning_rate'],
                            warmup_steps=hyperparameter['warmup_steps'],
                            model_version=hyperparameter['model_version'],
                            loss_log=loss_log
                            )
        except:
        '''
        print('no model log!')
        model.train(    batch_size = hyperparameter['batch_size'],
                        max_norm=hyperparameter['max_norm'],
                        epochs=hyperparameter['epochs'],
                        frezze=hyperparameter['frezze'],
                        learning_rate=hyperparameter['learning_rate'],
                        warmup_steps=hyperparameter['warmup_steps'],
                        model_version=hyperparameter['model_version'],
                        weight_decay = hyperparameter['weight_decay'],
                        early_stop = False
                        )

    else:
        '''
        model.set_model(frezze = True)
        model.validation()
        '''
        model.train(batch_size = hyperparameter['batch_size'],
                    max_norm=hyperparameter['max_norm'],
                    epochs=hyperparameter['epochs'],
                    frezze=hyperparameter['frezze'],
                    learning_rate=hyperparameter['learning_rate'],
                    warmup_steps=hyperparameter['warmup_steps'],
                    model_version=hyperparameter['model_version'],
                    weight = hyperparameter['loss_weight'],
                    weight_decay = hyperparameter['weight_decay'],
                    early_stop = True
                    )
    
    hyperparameter['loss_log'] = model.loss_log
    hyperparameter['threshold'] = model.threshold
    

    with open(f'./model/model_log/modelv{model_version}.txt', 'w') as file:
        for item, value in zip(hyperparameter.keys(),hyperparameter.values()):
            file.write('%s : %s \n ' %(item,value))

