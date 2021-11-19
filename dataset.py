"""
繼承pytorch的dataset
"""
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer  # bert斷詞，以字為單位
import pickle


class QADataset(Dataset):
    def __init__(self, mode, tokenizer):
        """
        ipnut:
        mode:train mode or test mode
        tokenizer: 使用何種tokenizer
        """
        self.mode = mode
        (
            self.full_question,
            self.full_division,
            self.tokenized_question,
            self.labeled_key,
            self.division_list
        ) = pickle.load(open(f"./pickle/{mode}_data.pkl", "rb"))

        self.tokenizer = tokenizer
        self.len = len(self.tokenized_question)
        # print(self.key)

    def __getitem__(self, idx):
        """
        ipnut:
            idx:要的是第幾筆的資料
        output:
            question_ids: 將以斷詞的問題轉換成
            key_ids: 將標記後的結果轉換成對應的數字後輸出
        """

        question_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenized_question[idx]
        )
        division = self.full_division[idx] + (24-len(self.full_division[idx]))*['無']
        
        #print('question_ids :',question_ids)
        mask_ids = [float(i > 0) for i in question_ids]

        return (
            torch.tensor(question_ids, dtype=torch.long),
            torch.tensor(mask_ids, dtype=torch.long),
            torch.tensor(self.labeled_key[idx], dtype=torch.float32),
            self.full_question[idx],
            division,
        )

    def __len__(self):
        """
        output: data總共有多少筆
        """
        return self.len


class AE_dataset(Dataset):
    def __init__(self, data=None):
        if data is None:
            self.data = pickle.load(open("./pickle/train_data_vector.pkl", "rb"))
        else:
            self.data = data
        self.len = len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float)
        # self.data["sentence"][idx]

    def __len__(self):
        return self.len


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    dataset = QADataset(mode="test", tokenizer=tokenizer)
    print(dataset.division_list)
    for i in range(100, 105):
        print("------------")
        print(i)
        print(dataset.__getitem__(i))
        print(dataset.__getitem__(i)[4])
        # print(dataset.__getitem__(0))
