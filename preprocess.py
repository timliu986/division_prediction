import pickle
import re
from transformers import BertTokenizer  # bert斷詞，以字為單位
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def preprocess(
    tokenizer,
    path,
    use_multi_division=True,
    resample=False,
):
    """
    input:
        tokenizer: 要用哪種tokenizer，目前都是用bert的
        path: dataset 的位置
        max_len: 句子的最大長度， 若是為-1，則會用train data中的最長長度為max_len
        use_multi_division: 當一個為題有多個科別可以對應時，是否將多個科別都做訓練(更改後無法使用)
        if_validation: 訓練時是否要做validation，若是有會多產生一個test_data.pkl的pickle檔存放validation的data
        division_list_version: 不同資料集對應不同的科別列表，0:train_data_v3.csv，1:train_data_v4.csv，2:train_data_v5.csv
    output:
        將下方幾項存為pickle檔
        output_question: 原始的問題
        output_division: 原始的科別
        tokenized_question: 斷詞後的問題
        labeled_key: 把關鍵字轉換成label("O":非關鍵字 "B":關鍵字開始 "I":關鍵字內容)
        max_len: 最長句子長度
    """
    df = pd.read_csv(path,converters={"division": lambda x: x.strip("['']").strip("'', ").split("', '")})
    define_division = [
        "乳房外科", "大腸直腸外科", "婦產科", "兒科",
        "復健科", "心臟科", "感染科", "新陳代謝科",
        "泌尿科", "牙科", "皮膚科", "眼科",
        "神經內/外科", "耳鼻喉科", "胸腔科", "腎臟內科",
        "肝膽腸胃科", "血液腫瘤科", "身心科", "免疫風濕科",
        "骨科", "口腔顎面外科","一般外科"
    ]



    ## max length measure

    if len(max(df['Question'].tolist(),key=len)) >= 510:
        max_len = 512
    else:
        max_len = len(max(df['Question'].tolist(),key=len)) + 2


    mlb = MultiLabelBinarizer(classes=define_division)
    df['label_key'] = mlb.fit_transform(df['division']).tolist()
    # label encode




    #問句文字處理

    # 統一標點符號
    df['Question'] = df['Question'].str.replace(",", "，")
    df['Question'] = df['Question'].str.replace("？", "?")

    # 去除多餘空白
    df['Question'] = df['Question'].str.replace(" ", "")
    df['Question'] = df['Question'].str.replace("\\s+", "")

    # 刪除換行
    df['Question'] = df['Question'].str.replace("\r*\n*", "")

    # 刪除英文外面括號
    df['Question'] = df['Question'].str.replace("\\([a-z A-Z \\-]*\\)", "[UNK]")

    # tokenizer
    tokens = []
    for q in tqdm(df['Question']):
        if len(q) >= max_len-2:
            q = q[ : max_len-2]
        token = ["[CLS]"] + tokenizer.tokenize(q) + ["[SEP]"]
        token = token + (max_len - len(token)) * ["[PAD]"]
        tokens.append(token)

    df['token'] = tokens





    # train validation test split

    train, test = \
        np.split(df.sample(frac=1, random_state=42),
                 [int(.8 * len(df))])

    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    #resample and save
    pd.Series([t for subt in train['division'] for t in subt]).value_counts()
    train['label_key'].apply(lambda x:sum(x)).value_counts()
    if resample and not use_multi_division:

        ros = RandomOverSampler(random_state=0)
        resample_x = train[['Question','token','division']]
        resample_y = train['label_key']
        resample_x, resample_y = ros.fit_resample(
            resample_x, resample_y
        )
        # print(sorted(Counter(label_ids).items()))

        with open("./pickle/train_data.pkl", "wb") as file:
            pickle.dump(
                (resample_x['Question'], resample_x['division'], resample_x['token'], resample_y, define_division), file)
    else:

        with open("./pickle/train_data.pkl", "wb") as file:
            pickle.dump((train['Question'], train['division'], train['token'], train['label_key'], define_division), file)


    with open("./pickle/test_data.pkl", "wb") as file:
        pickle.dump((test['Question'], test['division'], test['token'], test['label_key'],define_division), file)




if __name__ == "__main__":
    pretrained = "hfl/chinese-bert-wwm-ext"
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    preprocess(
        tokenizer=tokenizer,
        path="./data/train_data_v10_0715.csv",
        use_multi_division=True,
        resample=False,
    )

