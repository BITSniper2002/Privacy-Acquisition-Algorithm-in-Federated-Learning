import csv
import json

import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('punkt')
def generate_ngrams(sentence, n):
    # 确保输入是字符串类型
    if isinstance(sentence, str):
        words = word_tokenize(sentence)
    else:
        words = sentence
    return list(ngrams(words, n))

def longest_common_subsequence(s1, s2):
    # s1,s2 = s1.split(),s2.split()
    # m, n = len(s1), len(s2)
    # dp = [[0] * (n + 1) for _ in range(m + 1)]
    # for i in range(1, m + 1):
    #     for j in range(1, n + 1):
    #         if s1[i - 1] == s2[j - 1]:
    #             dp[i][j] = dp[i - 1][j - 1] + 1
    #         else:
    #             dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    # return dp[m][n]
    s1, s2 = s1.split(), s2.split()
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 通过 dp 表回溯找到最长公共子序列
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i - 1] == s2[j - 1]:
            lcs.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return lcs[::-1]

def rouge_score(ref_sentence, gen_sentences):
    """计算 ROUGE 分数"""
    # 生成参考句子和生成句子的一元组和二元组
    r1scores = []
    r2scores = []
    rlscores = []

    ref_unigrams = generate_ngrams(ref_sentence, 1)
    ref_bigrams = generate_ngrams(ref_sentence, 2)
    for i,gen_sentence in enumerate(gen_sentences):
        # print(f'round{i}')
        gen_unigrams = generate_ngrams(gen_sentence, 1)
        gen_bigrams = generate_ngrams(gen_sentence, 2)
        # 计算一元组和二元组的交集数量
        intersection_unigrams = Counter(ref_unigrams) & Counter(gen_unigrams)
        intersection_bigrams = Counter(ref_bigrams) & Counter(gen_bigrams)

        # 计算 ROUGE-1 和 ROUGE-2 分数
        rouge1 = sum(intersection_unigrams.values()) / max(1, sum(Counter(ref_unigrams).values()))
        # rouge1_p = sum(intersection_unigrams.values()) / sum(Counter(gen_unigrams).values()) if sum(Counter(gen_unigrams).values()) != 0 else 0
        # rouge1_r = sum(intersection_unigrams.values()) / sum(Counter(ref_unigrams).values()) if sum(Counter(ref_unigrams).values()) != 0 else 0
        # rouge1 = 2*(rouge1_p*rouge1_r) / (rouge1_p+rouge1_r) if (rouge1_r + rouge1_p) != 0 else 0
        rouge2 = sum(intersection_bigrams.values()) / max(1, sum(Counter(ref_bigrams).values()))
        # rouge2_p = sum(intersection_bigrams.values()) / sum(Counter(gen_bigrams).values()) if sum(Counter(gen_unigrams).values()) != 0 else 0
        # rouge2_r = sum(intersection_bigrams.values()) / sum(Counter(ref_bigrams).values()) if sum(Counter(ref_unigrams).values()) != 0 else 0
        # rouge2 = 2 * (rouge2_p * rouge2_r) / (rouge2_p + rouge2_r) if (rouge2_r + rouge2_p) != 0 else 0
        r1scores.append(rouge1)
        r2scores.append(rouge2)
        # 计算最长公共子序列（LCS）
        lcs = longest_common_subsequence(ref_sentence, gen_sentence)
        # with open("lcs.csv", "a", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(lcs)
        # lcs = "".join(lcs)
        # 计算 ROUGE-L 分数
        rougeL = len(lcs) / len(ref_sentence.split())
        rlscores.append(rougeL)

    rougeL = max(rlscores)
    rouge1 = max(r1scores)
    rouge2 = max(r2scores)
    return rouge1, rouge2,rougeL


# 从 CSV 文件读取 hypotheses
# hypotheses = []
# with open("after_tuning_wiki.csv", "r", newline="") as f:
#     reader = csv.reader(f)
#     for row in reader:
#         # if len(row) == 5:
#         hypotheses.append(row)
#         # else:
#         #     hypotheses.append(" ".join(row))
#         #     sent = " ".join(row)
#         #     #每个句子的单词数
#         #     l = len(sent)/5
#         #     hypo_words = sent.split()
#         #     sent = [" ".join(hypo_words[l*i:(i+1)*l]) for i in range(5)]
#
# # 从 JSON 文件读取 reference
# references = []
# with open("./wiki.json", "r") as f:
#     data = json.load(f)
#     for entry in data:
#         references.append(entry)

# 计算 ROUGE 分数并存储
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

hypotheses = [["He Bron James for 2024 . , He Bron James for 2024 . , He declares for draft He declares for draft","He Bron James for 2024 . , He Bron James for 2024 . , He declares for draft declares He for draft","He Bron James for 2024 . , He Bron James for 2024 . , He declares for draft for He draft declares","He Bron James for 2024 . , He Bron James for 2024 . , He for draft declares for He draft declares","He Bron James for 2024 . , He Bron James for 2024 . , He declares for draft for He declares draft"
]]
references = ["Bronny James , son of LeBron James , declares for 2024 NBA draft ."]
for hypothesis, reference in zip(hypotheses, references):
    rouge1, rouge2, rougeL = rouge_score(reference, hypothesis)
    rouge1_scores.append(rouge1)
    rouge2_scores.append(rouge2)
    rougeL_scores.append(rougeL)

# 将结果写入 CSV 文件
# with open("rouge_scores_tuning_wiki_.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["ROUGE-1", "ROUGE-2", "ROUGE-L"])
#     for rouge1, rouge2, rougeL in zip(rouge1_scores, rouge2_scores, rougeL_scores):
#         writer.writerow([rouge1, rouge2, rougeL])
#
# print("ROUGE 分数已写入 rouge_scores.csv 文件。")