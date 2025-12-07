import random

import pandas as pd


# 한국어 단어 CSV 읽기
ko_words = pd.read_csv("words_ko.csv", encoding="utf-8")


def make_nickname_ko(tone=None):

    if tone and tone != "norm":

        adj1_df = ko_words[(ko_words["pos"] == "adj_pos1") & (ko_words["tone"] == tone)]

        if adj1_df.empty:

            adj1_df = ko_words[ko_words["pos"] == "adj_pos1"]
    else:

        adj1_df = ko_words[ko_words["pos"] == "adj_pos1"]


    adj2_df = ko_words[ko_words["pos"] == "adj_pos2"]
    noun_df = ko_words[ko_words["pos"] == "noun"]


    adj_list_1 = adj1_df["word"].tolist()
    adj_list_2 = adj2_df["word"].tolist()
    noun_list = noun_df["word"].tolist()


    adj1 = random.choice(adj_list_1)
    adj2 = random.choice(adj_list_2)
    noun = random.choice(noun_list)


    # 공백 없이 단어를 반환 (일반 닉네임용)
    # return f"{adj1}{adj2}{noun}"

    # 공백으로 단어를 구분해서 반환 (토크나이저용)
    return f"{adj1} {adj2} {noun}"

