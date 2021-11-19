"""
可以即時測試的test檔
"""
from model import sequence_model
import torch


PRETRAINED_MODEL_NAME = "hfl/chinese-bert-wwm-ext"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    model_version = input('model version :　')
    model = sequence_model()
    model.load_model(model_version)
    model.test(batch_size = 256,save = True ,rule = True)


