import csv
import json
import string

from nltk.corpus import stopwords


def recombine_proper_nouns(words, proper_nouns):
    """将被拆分的专有名词重新组合成原词."""
    combined_proper_nouns = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words):
            word = " ".join(words[i:j+1])
            if word in proper_nouns:
                combined_proper_nouns.append(word)
                i = j + 1
                break
            j += 1
        if j == len(words):
            i += 1
    return combined_proper_nouns

with open("propn.json","r") as f:
    propn = json.load(f)

def find_proper_noun_suffix(word, proper_nouns):
    """从专有名词列表中找到单词的后缀."""
    suffixes = []
    for proper_noun in proper_nouns:
        if proper_noun.startswith(word) and proper_noun != word:
            suffixes.append(proper_noun[len(word):])
            # if suffix != word:
            #     return suffix
    return suffixes

def find_full_proper_noun(words,word, proper_nouns,suffix):
    """根据单词找到可能的专有名词."""
    flag = 0
    # print(suffix)
    if suffix is None or not len(suffix):
        return None
    for w in words:
        if suffix.startswith(w):
            flag = 1
            if suffix == w:
                words.remove(w)
                return word + suffix
            suffix = find_full_proper_noun(words,w, proper_nouns,suffix[len(w):])
            if suffix == None:
                return None
            if w in words:
                words.remove(w)
                # if suffix is None:
                #     return None
                # break
    if flag != 0:
        return word + suffix
    else:
        return None

stop_words = list(stopwords.words("english"))
with open('propn.json','r') as f:
    proper_nouns = json.load(f)

recoved_words = []
with open('./demonstration/demon.csv','r') as f:
    reader = csv.reader(f)
    for words in reader:
        words = [w for w in words if w != '<|endoftext|>' and len(w)]
        recoved_words.append(words)


for words in recoved_words:
    initial_len = len(words)
    c_words = [w for w in words if w[0].isupper() and w not in propn and w.lower() not in stop_words]  # 可能被拆分的专有名词
    to_del = []
    for c in c_words:
        suffixes = list(set(find_proper_noun_suffix(c, proper_nouns)))
        suffixes = [s for s in suffixes if s is not None]
        if len(suffixes):
            for suffix in suffixes:
                full_proper_noun = find_full_proper_noun(words,c, propn,suffix)
                if full_proper_noun is not None:
                    words.append(full_proper_noun)
                    to_del.append(c)
                    # print(full_proper_noun)
                    # print(words)
    for item in to_del:
        if item in words:
            words.remove(item)
    print(words)
    with open("./demonstration/sports.csv",'a') as file:
        writer = csv.writer(file)
        writer.writerow(words)


