import csv
import os
# Set the HF_HOME environment variable to a directory where you have write permissions
os.environ["HF_HOME"] = "./tmp"
from collections import Counter
from pprint import pprint

import nltk
import numpy as np
import torch
from nltk.corpus import stopwords
from torch.autograd import grad
from transformers import GPT2LMHeadModel


def shard_sequence(input, max_legnth):
    ori_len = len(input)
    ret = []
    for i in range(ori_len // max_legnth):
        s = input[i * max_legnth : (i + 1) * max_legnth]
        ret.append(s)
    return ret


def generate_sentences(
    LMhead: GPT2LMHeadModel,
    tokenizer,
    leak_results_words,
    rep=1,
    max_length=30,
    batch_size=1,
    max_rep=2,
    max_rep_dict=None,
    prompts_list=None,
    leak_results_tokens=None,
    device="cpu",
    temperature=1,
    mode="sample",
    top_p=0.95,
):
    sentences = []
    generated_token_lists = []
    num_parallel = 10

    stop_words = list(stopwords.words("english"))
    stop_words = " ".join(stop_words)
    stop_words_tokens = tokenizer.encode(stop_words, return_tensors="pt").to(device)
    str_l_words = " ".join(leak_results_words)
    if leak_results_tokens is None:
        leak_results_tokens = tokenizer.encode(
            str_l_words, return_tensors="pt"
        ).to(device)
    inputs = tokenizer(leak_results_words, return_tensors="pt", padding=True, truncation=True)

    # Generate attention mask
    token2word = {}
    max_rep_dict_ori = max_rep_dict
    if max_rep_dict_ori is None:
        max_rep_dict = {}
    for token_id in leak_results_tokens:
        for t in token_id:
            token2word[t] = tokenizer.decode(int(t))
            if max_rep_dict_ori is None:
                if t in stop_words_tokens:
                    max_rep_dict[t] = (
                        max_rep * 2
                    )  ## Yangsibo: we can also customize this value
                else:
                    max_rep_dict[t] = max_rep
    # print(max_rep_dict)
    # print(np.sum([v for (k, v) in max_rep_dict.items()]))
    # print(token2word)

    def prefix_allowed_tokens_fn(batch_id, sent):
        rep_dict = Counter(sent.cpu().numpy().tolist())
        allowed = Counter(max_rep_dict)
        allowed.subtract(rep_dict)
        # print(rep_dict)
        # print(f'allowed:{allowed.keys()}')

        ret = [k for (k, v) in allowed.items() if v > 0]


        prompts_ids = [
            tokenizer.encode(prompt, return_tensors="pt").cpu()[0].numpy()[0]
            for prompt in prompts_list
        ]

        if len(ret) == 0:
            return (
                tokenizer.encode("###", return_tensors="pt").to(device).cpu()[0].numpy()
            )

        if (
            len(sent) % max_length == 0
            or sent[-1]
            == tokenizer.encode(".", return_tensors="pt").to(device).cpu()[0].numpy()[0]
        ):  # Only allow prompt words at begining of sequences
            if len(set(prompts_ids) & set(ret)) > 0:
                return list(set(prompts_ids) & set(ret))
            else:
                return ret

        else:
            return list(set(ret) - set(prompts_ids))

    # def prefix_allowed_tokens_fn(batch_id,words):
    #     # 获取词汇表中每个词汇的 token IDs
    #     allowed_tokens = [tokenizer.encode(str(word), add_special_tokens=False)[0] for word in words]
    #     return allowed_tokens

    if prompts_list is None:
        prompts_list = leak_results_words  # Use each word in the inferred set as prompt

    for prompts in prompts_list:
        if prompts in ["[CLS]", "[SEP]", " "]:
            continue
        print(f'Using "{prompts}" as prompt ...')

        if mode == "beam":

            num_beams = min(num_parallel * rep, len(leak_results_words))
            num_return = min(num_beams, 50)
            # import pdb; pdb.set_trace()
            inputs = tokenizer(prompts, return_tensors="pt", truncation=True)
            output_sequences = LMhead.generate(
                input_ids=inputs["input_ids"].to(device),
                max_length=max_length * batch_size,
                min_length=max_length * batch_size,
                num_beams=num_beams,
                num_return_sequences=num_return,
                temperature=temperature,
                repetition_penalty=1.2,
                # attention_mask = attention_mask.to(device),
                pad_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )
            end_sign = tokenizer.encode("###", return_tensors="pt").to(device)[0]
            # import pdb; pdb.set_trace()
            output_sequences = [
                [token_id for token_id in output_sequence if token_id != end_sign]
                for output_sequence in output_sequences
            ]
            if batch_size > 1:
                output_sequences = [
                    shard_sequence(output_sequence, max_length)
                    for output_sequence in output_sequences
                ]
                # import pdb; pdb.set_trace()
                generated_token = [
                    [
                        [tokenizer.decode(int(token_id)) for token_id in s]
                        for s in output_sequence
                    ]
                    for output_sequence in output_sequences
                ]
                texts = [
                    tokenizer.batch_decode(output_sequence, skip_special_tokens=False)
                    for output_sequence in output_sequences
                ]
            else:
                generated_token = [
                    [tokenizer.decode(int(token_id)) for token_id in output_sequence]
                    for output_sequence in output_sequences
                ]
                texts = tokenizer.batch_decode(
                    output_sequences, skip_special_tokens=False
                )
            print(texts)
            sentences.extend(texts)
            generated_token_lists.extend(generated_token)

    return sentences, generated_token_lists
def new_last_gen(generated_texts, model, tokenizer):
    # m_len =len(generated_texts[-1].split())
    # last_gen = [text for text in generated_texts if len(text.split()) == m_len]
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained("neulab/gpt2-finetuned-wikitext103")
    model = AutoModelForCausalLM.from_pretrained("neulab/gpt2-finetuned-wikitext103")
    t_gen = {}
    for gen in generated_texts:
        input_ids = tokenizer.encode(gen, return_tensors="pt")

        # 使用模型计算损失值
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        t_gen[gen] = loss
    sorted_ = dict(sorted(t_gen.items(), key=lambda item: item[1])[:5])
    last_gen = list(sorted_.keys())
    return last_gen
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("neulab/gpt2-finetuned-wikitext103")
model = AutoModelForCausalLM.from_pretrained("neulab/gpt2-finetuned-wikitext103")
tokenizer.pad_token = tokenizer.eos_token

# Define your input parameters
recovered_words_list= []
# prompts_list = ["The cat", "The dog", "The mouse"]
with open('enron.csv','r') as file:
    reader = csv.reader(file)
    count = 0
    for row in reader:
        if count >= 0:
            row = [item for item in row if item.strip() and item != '<|endoftext|>']
            recovered_words_list.append(row)
        else:
            count += 1
# Call the function to generate sentences
for words in recovered_words_list:
    sentences, generated_token_lists = generate_sentences(
        LMhead=model,
        tokenizer=tokenizer,
        leak_results_words=words,
        # prompts_list=prompts_list,
        device="cpu",  # or "cpu" if you don't have a GPU
        mode="beam",  # or "beam" for beam search
        temperature=0.7,  # adjust temperature for sampling
    )
    sentences = new_last_gen(sentences,tokenizer,model)
    print(sentences)
    # Print or use the generated sentences
    with open("FILM_enron.csv","a") as f:
        writer = csv.writer(f)
        writer.writerow(sentences)
    print("Sentences Written")