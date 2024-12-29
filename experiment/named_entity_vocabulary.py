import pandas as pd
import spacy
from collections import Counter

# 加载Spacy模型
nlp = spacy.load("en_core_web_sm")

def extract_proper_nouns(text):
    """从文本中提取专有名词."""
    doc = nlp(str(text))
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
    return proper_nouns

def build_proper_noun_set(texts):
    """从文本列表中构建专有名词集合."""
    proper_noun_counter = Counter()

    for text in texts:
        proper_nouns = extract_proper_nouns(text)
        proper_noun_counter.update(proper_nouns)

    # 选择出现次数较多的专有名词
    proper_nouns_set = [noun for noun, count in proper_noun_counter.items() if count >= 1]

    return proper_nouns_set


df = pd.read_csv("data.csv")

# 提取文本数据
data = df["text"].tolist()

# 构建专有名词集合
proper_nouns_set = build_proper_noun_set(data)
print(proper_nouns_set)
