
from model import sequence_model
import torch




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":


    PRETRAINED_MODEL_NAME = ["bert-base-chinese",
                             "hfl/chinese-bert-wwm-ext",
                             "voidful/albert_chinese_tiny"]
    model_version = ['bert-base','bert-wwm','albert']

    for pretrained_name,model_name in zip(PRETRAINED_MODEL_NAME,model_version):
        hyperparameter = {"model_name": model_name,
                          'PRETRAINED_MODEL_NAME': pretrained_name,
                          "frezze": False,
                          "batch_size": 15,
                          "loss_weight": None,
                          "max_norm": 1.0,
                          "epochs": 2,
                          "learning_rate": 7.919544235009059e-05,
                          "weight_decay": 0.08255158587968352,
                          "warmup_steps": 6
                          }

        model = sequence_model(
                               use_multiple_gpu = True
                               )
        model.k_fold_train(k = 5,
                        PRETRAINED_MODEL_NAME = hyperparameter['PRETRAINED_MODEL_NAME'],
                        batch_size=hyperparameter['batch_size'],
                        max_norm=hyperparameter['max_norm'],
                        epochs=hyperparameter['epochs'],
                        frezze=hyperparameter['frezze'],
                        learning_rate=hyperparameter['learning_rate'],
                        warmup_steps=hyperparameter['warmup_steps'],
                        weight_decay = hyperparameter['weight_decay'],
                        model_version=hyperparameter['model_name'],
                        use_multiple_gpu = True
                        )

