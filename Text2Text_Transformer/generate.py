import csv
import pandas as pd
import os
from config import opt
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")


def write_into_csv(file_path, content_list):
    Path(file_path.rsplit('/', 1)[0]).mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        writer = csv.writer(f)
        writer.writerows(content_list)

def load_examples_and_label(test_data_path, tokenizer):
    src_text = []
    label_text = []
    data_samples = pd.read_csv(test_data_path, index_col=False).values.tolist()

    for sample in data_samples:
        # src_text.append(tokenizer.bos_token + sample[1] + tokenizer.additional_special_tokens[0])
        # src_text.append(sample[1] + ' prefix ')
        if str(sample[0]) != 'nan':
            if opt.flag in [
                    "finetune_gpt2_prefix"]:
                src_text.append(sample[0] + ' prefix ')
            else:
                src_text.append(sample[0] + ' ')  # test without prefix
            label_text.append(sample[1])
    samples = [[samp, tgr_t] for samp, tgr_t in zip(src_text, label_text)]

    return samples

# Created for testing BLEU Scores during the evaluate() step in finetune.py
def generate_predictions_for_bleu_score_probing(model, tokenizer):
    test_datasets = load_examples_and_label(opt.eval_data_file, tokenizer)[:100]

    test_sampler = SequentialSampler(test_datasets)
    test_dataloader = DataLoader(
        test_datasets, sampler=test_sampler, batch_size=16
    )

    # model.to(device)
    generation_list = []

    for batch in tqdm(test_dataloader, desc="Generating predictions for the first 100 eval data"):
        # Have no idea why both batch[0] and batch[1] are tuples
        samples = [list(batch[0]), list(batch[1])]
        sample = samples[0]
        labels = samples[1]

        generation = get_model_generation(samples, model, tokenizer, opt.test_num_return_sequences)

        for inp in range(len(labels)):
            item = []
            item.append(sample[inp])
            item.append(labels[inp])
            item.extend(generation[inp])
            generation_list.append(item)
    return generation_list


def generate(model_type: str, model_path: str, test_data_path: str, output_path: str):
    print("The generated result saved path is: ", output_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # if opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
    model = AutoModelWithLMHead.from_pretrained(model_path)
        
    test_datasets = load_examples_and_label(test_data_path, tokenizer)
    if os.path.isfile(output_path):
        with open(output_path, "r") as f:
            csv_read = csv.reader(f)
            steps_generated = len(list(csv_read)) - 1
            test_datasets = test_datasets[steps_generated:]
    else:
        columns_list = ["sample", "label"]
        for i in range(0, opt.num_return_sequences):
            columns_list.append("generation_" + str(i))
        write_into_csv(output_path, [columns_list])

    test_sampler = SequentialSampler(test_datasets)
    test_dataloader = DataLoader(
        test_datasets, sampler=test_sampler, batch_size=opt.test_batch_size
    )
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    generation_list = []

    for batch in tqdm(test_dataloader, desc="Testing"):
        samples = batch

        sample = samples[0]
        labels = samples[1]
        generation = get_model_generation(samples, model, tokenizer)

        for inp in range(len(labels)):
            item = []
            item.append(sample[inp])
            item.append(labels[inp])
            item.extend(generation[inp])
            generation_list.append(item)
        if len(generation_list) >= opt.logging_steps_generate:
            write_into_csv(output_path, generation_list)
            generation_list = []
    write_into_csv(output_path, generation_list)
    print("\nGeneration Finished!")

def get_model_generation(samples, model, tokenizer, num_return_sequences=opt.num_return_sequences):
    sample = samples[0]

    # if opt.flag == "finetune_gpt2" or opt.flag == "finetune_gpt2_prefix":
    inputs = tokenizer.batch_encode_plus(list(sample), return_tensors="pt", padding=True)
    generation_output = model.generate(inputs["input_ids"].to(device),
                                        min_length=opt.min_length,
                                        max_length=opt.max_length,
                                        do_sample=opt.do_sample,
                                        top_p=opt.top_p, top_k=opt.top_k,
                                        bos_token_id=tokenizer.bos_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.pad_token_id,
                                        num_return_sequences=num_return_sequences)
    
    generation = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    generation = [generation[i:i + num_return_sequences] for i in range(0, len(generation), num_return_sequences)]
    return [[m_i.replace(n.replace(tokenizer.bos_token, '').replace(tokenizer.additional_special_tokens[0], ''),
                               "").strip() for m_i in m] for n, m in zip(sample, generation)]

