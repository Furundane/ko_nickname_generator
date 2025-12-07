import random

import pandas as pd


# 영어 단어 CSV 읽기
en_words = pd.read_csv("words_en.csv", encoding="utf-8")


def make_nickname_en():

    adj_list = en_words[en_words["pos"] == "adj"]["word"].tolist()
    noun_list = en_words[en_words["pos"] == "noun"]["word"].tolist()

    adj1, adj2 = random.sample(adj_list, 2) # 같은 형용사 2개를 중복해서 뽑지 않도록 함
    noun = random.choice(noun_list)

    return f"{adj1}-{adj2}-{noun}"

if __name__ == "__main__":
    main()