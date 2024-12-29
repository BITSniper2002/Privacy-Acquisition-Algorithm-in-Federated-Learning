from collections import Counter

def rouge_score(ref_sentence, gen_sentence):
    """计算 ROUGE 分数"""
    # 生成参考句子和生成句子的一元组和二元组
    ref_unigrams = generate_ngrams(ref_sentence, 1)
    ref_bigrams = generate_ngrams(ref_sentence, 2)
    gen_unigrams = generate_ngrams(gen_sentence, 1)
    gen_bigrams = generate_ngrams(gen_sentence, 2)

    # 计算一元组和二元组的交集数量
    intersection_unigrams = Counter(ref_unigrams) & Counter(gen_unigrams)
    intersection_bigrams = Counter(ref_bigrams) & Counter(gen_bigrams)

    # 计算 ROUGE-1 和 ROUGE-2 分数
    rouge1 = sum(intersection_unigrams.values()) / max(1, sum(Counter(ref_unigrams).values()))
    rouge2 = sum(intersection_bigrams.values()) / max(1, sum(Counter(ref_bigrams).values()))

    # 计算最长公共子序列（LCS）
    lcs = longest_common_subsequence(ref_sentence, gen_sentence)
    lcs = "".join(lcs)[:-1]
    # 计算 ROUGE-L 分数
    rougeL = len(lcs.split()) / len(ref_sentence.split())

    return rouge1, rouge2, rougeL

def generate_ngrams(sentence, n):
    """生成 n 元组"""
    words = sentence.split()
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

def longest_common_subsequence(s1, s2):
    """计算最长公共子序列"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
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

# 示例用法
ref_sentence = "the quick brown fox jumps over the lazy dog"
gen_sentence = "the lazy cat sleeps"
rouge1, rouge2, rougeL = rouge_score(ref_sentence, gen_sentence)
print("ROUGE-1:", rouge1)
print("ROUGE-2:", rouge2)
print("ROUGE-L:", rougeL)
