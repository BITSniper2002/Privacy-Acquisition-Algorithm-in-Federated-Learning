import os
# Set the HF_HOME environment variable to a directory where you have write permissions
os.environ["HF_HOME"] = "./tmp"
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from dlg_utils import calculatePerplexity,get_grad_gpt2
import random

class LocalSwap:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def computeStats(self,sent, do_split=False):

        # generated_sentence =  self.tokenizer.decode(curr_tokens)
        # if do_split:
        #     generated_sentence = generated_sentence.split("<|endoftext|>")
        generated_dy_dx = get_grad_gpt2(sent, self.tokenizer, self.model)

        gradnorm = torch.norm(generated_dy_dx[68]).cpu().numpy().item()
        ppl = calculatePerplexity(
            str(sent), self.model, self.tokenizer
        ).item()

        return {
            "sentence": sent,
            "gradloss_mse": ppl+gradnorm,
        }

    def reorderAlgorithm(self , tokens,sent, n=1):
        values = []
        # for _, sequence in zip(range(n), generator):
        #     curr_tokens = copy.copy(tokens)
        #     for (i, j) in sequence:
        #         if i < len(curr_tokens) and j < len(curr_tokens):
        #             curr_tokens[i], curr_tokens[j] = curr_tokens[j], curr_tokens[i]
        #     values.append(self.computeStats(curr_tokens , do_split=True))
        for _ in range(n):
            new_sent = sent.split()
            i,j = random.randint(0,len(new_sent)-1),random.randint(0,len(new_sent)-1)
            # curr_tokens[i], curr_tokens[j] = curr_tokens[j], curr_tokens[i]
            new_sent[i],new_sent[j] = new_sent[j],new_sent[i]
            new_sent = " ".join(new_sent)
            values.append(self.computeStats(new_sent,do_split=True))

        if len(values) == 0:
            return {"gradloss_mse": 10e100}

        return min(values, key=lambda x: x["gradloss_mse"])

    def insertAlgorithm(
        self , tokens, word_pool,sent,  n=1
    ):
        values = []
        for _ in range(n):
            new_sent = sent.split()
            i,p = random.randint(0,len(word_pool)-1),random.randint(0,len(new_sent)-1)
            new_sent = new_sent[0:p] + [word_pool[i]] + new_sent[p:]
            new_sent = " ".join(new_sent)
            values.append(self.computeStats(new_sent, do_split=True))
        if len(values) == 0:
            return {"gradloss_mse": 10e100}

        return min(values, key=lambda x: x["gradloss_mse"])

    def moveSubsequenceAlgorithm(self, tokens,sent,n=1):
        values = []
        #select three pos, move tokens[i:j+1] after tokens[p]
        for _ in range(n):
            new_sent = sent.split()
            i, j, p = random.sample(range(len(new_sent) - 1), 3)
            i, j = min(i, j), max(i, j)
            # Ensure p is not within the range of i and j
            if i < p < j:
                p = j + random.randint(1, len(new_sent) - j - 1)
            elif j < p < len(new_sent) - 1:
                p = j - random.randint(1, j - i)
            # Move the subsequence between i and j to position p after j
            subsequence = new_sent[i:j + 1]
            del new_sent[i:j + 1]
            new_sent[p:p] = subsequence
            new_sent = " ".join(new_sent)
            values.append(self.computeStats(new_sent, do_split=True))

        if len(values) == 0:
            return {"gradloss_mse": 10e100}

        return min(values, key=lambda x: x["gradloss_mse"])

    def movePrefixAlgorithm(self, tokens,sent, n=1):
        values = []
        # for _, sequence in zip(range(n), generator):
        for _ in range(n):
            new_sent = sent.split()
            i = random.randint(0, len(new_sent) - 1)  # Randomly select position i
            prefix = new_sent[:i + 1]  # Extract prefix ending at position i
            del new_sent[:i + 1]  # Remove prefix from original list
            new_sent.extend(prefix)
            new_sent = " ".join(new_sent)
            values.append(self.computeStats( new_sent, do_split=True))
        if len(values) == 0:
            return {"gradloss_mse": 10e100}

        return min(values, key=lambda x: x["gradloss_mse"])


def reordering(original_sentence,beamsearch_res,model,tokenizer):
    from functools import reduce
    res = []
    for bs in beamsearch_res:
        localswap = LocalSwap(model, tokenizer)
        # get tokens
        current_tokens = reduce(
            lambda x, y: x + y,
            [tokenizer.encode(s) for s in bs],  # seperate sentences
        )

        all_words = bs.split()

        losses = [None]
        best = []
        cnt = 0
        for i in range(0, LOCAL_MAX_STEP):
            print(f"round {i+1}")
            reordered = localswap.reorderAlgorithm(
                current_tokens,bs
            )
            inserted = localswap.insertAlgorithm(
                current_tokens, all_words,bs
            )
            movesub = localswap.moveSubsequenceAlgorithm(current_tokens,bs)
            movepre = localswap.movePrefixAlgorithm(current_tokens,bs)
            print(f"after reorder:{reordered['sentence']},")
            print(f"after insert:{inserted['sentence']},")
            print(f"after move subsequence:{movesub['sentence']},")
            print(f"after move prefix:{movepre['sentence']},")
            curr_best = min(
                    [
                        reordered,
                        inserted,
                        movesub,
                        movepre,
                        localswap.computeStats(bs, do_split=True),
                    ],
                    key=lambda x: x["gradloss_mse"],
                )
            best.append(curr_best)
            print(f"current best:{curr_best['sentence']},")
            # print("GradNorm+ppl:",curr_best["gradloss_mse"],)
            if curr_best["gradloss_mse"] == losses[-1]:
                cnt += 1
                if cnt == 3:
                    break
            else:
                cnt = 0
            losses.append(curr_best["gradloss_mse"])
            if curr_best["gradloss_mse"] == 0.0:
                print("Finished early due to perfect recovery!")
                break

        best = sorted(best,key=lambda x: x["gradloss_mse"],reverse=False)[0]
        res.append(best)
    return res

if __name__ == "__main__":
    LOCAL_MAX_STEP = 1
    # sentences,beamsearch_res = [],[]
    # with open("./dataset/wiki.json", "r") as f:
    #     data = json.load(f)
    #     for entry in data:
    #         sentences.append(entry)
    # with open('./recovered_sentences/res_wiki.csv', 'r', newline='') as csvfile:
    #     reader = csv.reader(csvfile)
    #     cnt = 0
    #     for row in reader:
    #         beamsearch_res.append(row)
    sentences = ["In response to lawsuit, NCAA says it doesn't control quality of education for student-athletes . But its website emphasizes importance of education, \"opportunities to learn\" Lawsuit claims students didn't get an education because of academic fraud at UNC ."]
    beamsearch_res = [["NCAA it for education . student says quality control NCAA it for education . student says quality control for students it doesn ath for students doesn ath it doesn doesn,NCAA it for education . student says quality control NCAA it for education . student says quality control for students it doesn ath for students it doesn ath doesn doesn,NCAA it for education . student says quality control NCAA it for education . student says quality control it for students doesn ath it for students doesn ath doesn doesn,NCAA it for education . student says quality control NCAA it for education . student says quality control for students it doesn ath it for students doesn ath doesn doesn,NCAA it for education . student says quality control NCAA it for education . student says quality control for students it doesn ath for it students doesn ath doesn doesn"]]
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("satkinson/DialoGPT-small-marvin")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("satkinson/DialoGPT-small-marvin")
    # reordered = reordering(sentences, beamsearch_res, model, tokenizer)
    # print(reordered)
    csv_file = "./demonstration/after_tuning.csv"
    for i in range(len(beamsearch_res)):
        sent = []
        reordered = reordering(sentences[i],beamsearch_res[i],model,tokenizer)
        print(reordered)
        for r in reordered:
            sent.append(r['sentence'])
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(sent)