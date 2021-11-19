from division_model import sequence_model
import torch


PRETRAINED_MODEL_NAME = "hfl/chinese-bert-wwm-ext"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    model_version = input('model version : ')
    model = sequence_model(PRETRAINED_MODEL_NAME = PRETRAINED_MODEL_NAME)
    model.load_model(model_version)
    while True:
      sentence = str(input('input : '))
      if sentence == 'quit':
        break
      result = model.test_for_input(sentence ,rule_base = True)
      print(result)
      