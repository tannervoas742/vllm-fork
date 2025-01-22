#!/usr/bin/env python
import argparse
import os
import time
import re
import string
import random
import json
from tqdm import tqdm
from pathlib import Path
import evaluate
# Hugging Face Datasets for public benchmarks
from datasets import load_dataset, concatenate_datasets
from typing import Optional, List, Dict
# vLLM imports
from vllm import LLM, SamplingParams
try:
    from vllm import debug_store
    has_debug_store = True
except:
    has_debug_store = False

def normalize_text(text: str) -> str:
    """
    Lowercase, remove punctuation/articles, and extra whitespace.
    This follows a style often used in QA tasks like SQuAD to
    ensure consistent tokenization.
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    # Remove common articles: 'a', 'an', 'the'
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def exact_match_score(prediction: str, reference: str) -> float:
    """
    Returns 1.0 if the normalized texts match exactly, 0.0 otherwise.
    """
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0

def f1_token_score(prediction: str, reference: str) -> float:
    """
    Computes a token-level F1 overlap between prediction and reference.
    - Precision: (# of overlapping tokens) / (total tokens in prediction)
    - Recall:    (# of overlapping tokens) / (total tokens in reference)
    - F1:        2 * (precision * recall) / (precision + recall)
    """
    pred_tokens = normalize_text(prediction).split()
    ref_tokens  = normalize_text(reference).split()
    common_tokens = set(pred_tokens).intersection(set(ref_tokens))
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        # Edge case: if either is empty text
        return 1.0 if pred_tokens == ref_tokens else 0.0
    # Count how many times each token appears in both
    common_count = 0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t in pred_tokens:
        if t in ref_counts and ref_counts[t] > 0:
            common_count += 1
            ref_counts[t] -= 1
    precision = common_count / len(pred_tokens)
    recall = common_count / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1

def mixed_metric_score(prediction: str, reference: str) -> float:
    """
    Final score that blends:
     - Exact Match
     - Token-Level F1
    You can adjust the weighting below to fit your use-case.
    """
    em = exact_match_score(prediction, reference)
    f1 = f1_token_score(prediction, reference)
    # Example weighting: 50% exact match, 50% F1
    return 0.5 * em + 0.5 * f1

##############################################################################
# 1. (Optionally) Prepare / Download Dataset
##############################################################################
def maybe_download_and_prepare(dataset_ids: str) -> None:
    """
    Download/prepare the dataset indicated by `dataset_id`.
    1) If it's one of the known Hugging Face text benchmarks (MMLU, etc.),
      we simply load it from HF, ensuring a local cache.
    Raises:
     ValueError if the dataset_id is unrecognized.
     FileNotFoundError if there's an issue with local paths (for custom sets).
    """
    # Mapping: dataset_id -> (Hugging Face dataset name, split)
    hf_datasets = {
        # MMLU - we’ll load the 'test' split from the 'lukaemon/mmlu' HF dataset
        "mmlu": ("lukaemon/mmlu", "test", [
            'high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', 'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology'
        ]),
        # WinoGrande - 'winogrande', 'validation' split
        "winogrande": ("winogrande", "validation", [
            #'winogrande_xs',
            #'winogrande_s',
            #'winogrande_m',
            #'winogrande_l',
            'winogrande_xl',
            #'winogrande_debiased',
        ]),
        # Open-domain QA Example - here we use 'natural_questions'
        "openqa": ("natural_questions", "validation", None),
        # SuperGLUE Example - we’ll pick 'test' as a subtask
        "superglue_01": ("super_glue", "validation", ['boolq']),
        "superglue_02": ("super_glue", "validation", ['cb']),
        "superglue_03": ("super_glue", "validation", ['copa']),
        "superglue_04": ("super_glue", "validation", ['multirc']),
        "superglue_05": ("super_glue", "validation", ['record']),
        "superglue_06": ("super_glue", "validation", ['rte']),
        "superglue_07": ("super_glue", "validation", ['wic']),
        "superglue_08": ("super_glue", "validation", ['wsc']),
        "superglue_09": ("super_glue", "validation", ['wsc.fixed']),
        "superglue_10": ("super_glue", "validation", ['axb']),
        "superglue_11": ("super_glue", "validation", ['axg']),
    }
    dataset_ids = dataset_ids.split(',')
    for dataset_id in dataset_ids:
        # -------------------------------------------------------------------------
        # 1) If dataset_id matches a known HF dataset, just load/cache from HF
        # -------------------------------------------------------------------------
        if dataset_id in hf_datasets:
            dataset_name, split, subsets = hf_datasets[dataset_id]
            hf_cache_dir = Path("/root/.cache/huggingface")
            hf_cache_dir.mkdir(parents=True, exist_ok=True)
            sentinel_file = hf_cache_dir / f"{dataset_id}_prepared.txt"
            if sentinel_file.exists():
                print(f"[INFO] Hugging Face dataset '{dataset_id}' is already cached/prepared.")
            else:
                print(f"[INFO] Downloading '{dataset_name}' split='{split}' from Hugging Face to cache.")
                if subsets is not None:
                    for subset_name in subsets:
                        _ = load_dataset(
                            dataset_name,
                            subset_name,
                            split=split,
                            cache_dir=str(hf_cache_dir)
                        )
                else:
                    _ = load_dataset(dataset_name, split=split, cache_dir=str(hf_cache_dir))
                sentinel_file.write_text("Prepared\n")
        elif os.path.exists(dataset_id):
            pass # Assume its a local dataset
        else:
            # Not in Hugging Face list => error
            raise ValueError(
                f"Unknown dataset_id: '{dataset_id}'.\n"
                f"Supported Hugging Face sets: {list(hf_datasets.keys())}\n"
            )

##############################################################################
# 2. Load the Dataset into Memory for Evaluation
##############################################################################
def load_dataset_for_eval(dataset_ids: str, max_tokens: int) -> List[Dict[str, str]]:
    """
    Returns a list of {'prompt': str, 'answer': str} items for evaluation.
    1) For known HF text benchmarks (MMLU, WinoGrande,
      open-domain QA, SuperGLUE), we load them from HF, transform each
      record into a standardized dict: {"prompt": ..., "answer": ...}.
    Raises:
     ValueError if the dataset_id is unrecognized.
     FileNotFoundError if the index file or HF dataset is not found.
    """
    hf_datasets = {
        "mmlu": ("lukaemon/mmlu", "test", [
            'high_school_european_history', 'business_ethics', 'clinical_knowledge', 'medical_genetics', 'high_school_us_history', 'high_school_physics', 'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology', 'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology'
        ]),
        "winogrande": ("winogrande", "validation", [
            #'winogrande_xs',
            #'winogrande_s',
            #'winogrande_m',
            #'winogrande_l',
            'winogrande_xl',
            #'winogrande_debiased',
        ]),
        "openqa": ("natural_questions", "validation", None),
        "superglue_01": ("super_glue", "validation", ['boolq']),
        "superglue_02": ("super_glue", "validation", ['cb']),
        "superglue_03": ("super_glue", "validation", ['copa']),
        "superglue_04": ("super_glue", "validation", ['multirc']),
        "superglue_05": ("super_glue", "validation", ['record']),
        "superglue_06": ("super_glue", "validation", ['rte']),
        "superglue_07": ("super_glue", "validation", ['wic']),
        "superglue_08": ("super_glue", "validation", ['wsc']),
        "superglue_09": ("super_glue", "validation", ['wsc.fixed']),
        "superglue_10": ("super_glue", "validation", ['axb']),
        "superglue_11": ("super_glue", "validation", ['axg']),
    }
    # -------------------------------------------------------------------------
    # 1) Hugging Face Benchmarks
    # -------------------------------------------------------------------------
    dataset = []
    max_tokens_lower = 32
    dataset_ids = dataset_ids.split(',')
    dataset_ids.sort()
    for dataset_id in dataset_ids:
        if dataset_id in hf_datasets:
            dataset_name, split, subsets = hf_datasets[dataset_id]
            hf_cache_dir = "/root/.cache/huggingface"
            if subsets is not None:
                ds_list = []
                for subset_name in subsets:
                    ds_subset = load_dataset(
                        dataset_name,
                        subset_name,
                        split=split,
                        cache_dir=hf_cache_dir
                    )
                    ds_list.append(ds_subset)
                # 2. Concatenate all sub-tasks into a single Dataset
                if len(ds_list) > 1:
                    ds = concatenate_datasets(ds_list)
                else:
                    ds = ds_list[0]
            else:
                ds = load_dataset(dataset_name, split=split, cache_dir=hf_cache_dir)
            if dataset_id == "mmlu":
                # Expect 'question' + 'answer'
                for row in ds:
                    question = row["input"].strip()
                    choice_a = row.get("A", "").strip()
                    choice_b = row.get("B", "").strip()
                    choice_c = row.get("C", "").strip()
                    choice_d = row.get("D", "").strip()
                    correct_letter = row.get("target", "").strip()  # e.g. "A"
                    # Construct prompt
                    prompt = (
                        f"{question}\n"
                        f"A) {choice_a}\n"
                        f"B) {choice_b}\n"
                        f"C) {choice_c}\n"
                        f"D) {choice_d}\n"
                        "Answer:"
                    )
                    dataset.append({
                        "sample_num": len(dataset) + 1,
                        "prompt": prompt,
                        "answer": correct_letter,
                        "singular": True,
                    })
            elif dataset_id == "winogrande":
                # 'winogrande' => each row has "sentence", "option1", "option2", "answer" in {1,2}
                for row in ds:
                    question = row["sentence"].strip()
                    choice_1 = row.get("option1", "").strip()
                    choice_2 = row.get("option2", "").strip()
                    correct_number = str(row.get("answer", "")).strip()
                    # Construct prompt
                    prompt = (
                        f"{question}\n"
                        f"1) {choice_1}\n"
                        f"2) {choice_2}\n"
                        "Answer:"
                    )
                    dataset.append({
                        "sample_num": len(dataset) + 1,
                        "prompt": prompt,
                        "answer": correct_number,
                        "singular": True,
                    })
            elif dataset_id == "openqa":
                # e.g. 'natural_questions' => each row has "question" + "answers"
                for row in ds:
                    prompt = row.get("question", "")
                    ans_list = row.get("answers", [])
                    # If multiple answers, just pick the first or join them
                    answer = ans_list[0] if ans_list else "N/A"
                    dataset.append({
                        "sample_num": len(dataset) + 1,"prompt": prompt, "answer": answer})
            elif dataset_id == "superglue_01":
                for row in ds:
                    passage = row["passage"].strip()
                    question = row.get("question", "").strip()
                    correct_number = str(row.get("label", "")).strip()
                    # Construct prompt
                    prompt = (
                        f"Passage: {passage}\n"
                        f"Question: {question}\n"
                        "Reply with 1 if the answer to the question is true or yes and 0 if it is false or no:"
                    )
                    dataset.append({
                        "sample_num": len(dataset) + 1,
                        "prompt": prompt,
                        "answer": correct_number,
                        "singular": True,
                    })
            elif dataset_id == "superglue_02":
                for row in ds:
                    premise = row["premise"].strip()
                    hypothesis = row.get("hypothesis", "").strip()
                    correct_number = str(row.get("label", "")).strip()
                    # Construct prompt
                    prompt = (
                        f"Premise: {premise}\n"
                        f"Hypothesis: {hypothesis}\n"
                        "Reply with 0 if the hypothesis is true and 1 if it is false or 2 if unsure:"
                    )
                    dataset.append({
                        "sample_num": len(dataset) + 1,
                        "prompt": prompt,
                        "answer": correct_number,
                        "singular": True,
                    })
            elif dataset_id == "superglue_03":
                for row in ds:
                    premise = row["premise"].strip()
                    question = row.get("question", "").strip()
                    choice_1 = row.get("choice1", "").strip()
                    choice_2 = row.get("choice2", "").strip()
                    correct_number = str(str(row.get("label", ""))).strip()
                    # Construct prompt
                    prompt = (
                        f"Premise: {premise}\n"
                        f"Question: {question}\n"
                        f"1) {choice_1}\n"
                        f"2) {choice_2}\n"
                        "Answer:"
                    )
                    dataset.append({
                        "sample_num": len(dataset) + 1,
                        "prompt": prompt,
                        "answer": correct_number,
                        "singular": True,
                    })
            elif dataset_id == "superglue_04":
                pass # Not singular. paragraph, question, answer, label.
            elif dataset_id == "superglue_05":
                pass # Not singular. 'passage', 'query', 'entities', 'entity_spans', 'answers', 'idx'.
            elif dataset_id == "superglue_06":
                for row in ds:
                    premise = row["premise"].strip()
                    hypothesis = row.get("hypothesis", "").strip()
                    correct_number = str(row.get("label", "")).strip()
                    # Construct prompt
                    prompt = (
                        f"Premise: {premise}\n"
                        f"Hypothesis: {hypothesis}\n"
                        "Reply with 0 if the hypothesis is true and 1 if it is false or 2 if unsure:"
                    )
                    dataset.append({
                        "sample_num": len(dataset) + 1,
                        "prompt": prompt,
                        "answer": correct_number,
                        "singular": True,
                    })
            elif dataset_id == "superglue_07":
                for row in ds:
                    sentence1 = row.get("sentence1", "").strip()
                    start1 = str(row.get("start1", "")).strip()
                    end1 = str(row.get("end1", "")).strip()
                    sentence2 = row.get("sentence2", "").strip()
                    start2 = str(row.get("start2", "")).strip()
                    end2 = str(row.get("end2", "")).strip()
                    correct_word = str(row.get("word", "")).strip()
                    # Construct prompt
                    prompt = (
                        f"Sentence 1: {sentence1}\n"
                        f"Sentence 2: {sentence2}\n"
                        f"What word starts at character {start1} and ends at character {end1} of sentence 1 and starts at character {start2} and ends at character {end2} of sentence 2:"
                    )
                    dataset.append({
                        "sample_num": len(dataset) + 1,
                        "prompt": prompt,
                        "answer": correct_word,
                        "singular": True,
                    })
            elif dataset_id == "superglue_08":
                pass # Not singular.
            elif dataset_id == "superglue_09":
                pass # Not singular.
            elif dataset_id == "superglue_10":
                pass # No validation.
            elif dataset_id == "superglue_11":
                pass # No validation.
            else:
                raise ValueError(f"Unhandled HF dataset_id in load logic: '{dataset_id}'")
        elif os.path.exists(dataset_id):
            with open(dataset_id) as user_file:
                file_contents = user_file.read()
                ds = json.loads(file_contents)
            for row in ds:
                id = -1
                if "id" in row:
                    id = row["id"]
                question = row['conversations'][0]['value']
                answer = row['conversations'][1]['value']
                if len(answer) > max_tokens:
                    continue
                # Construct prompt
                prompt = (
                    f"Question: {question}\n"
                    "Answer:"
                )
                dataset.append({
                    "sample_num": len(dataset) + 1,
                    "prompt": prompt,
                    "answer": answer,
                    "singular": False,
                    "id": id,
                })
                max_tokens_lower = max([max_tokens_lower, len(answer)])
        else:
            raise ValueError(
                f"Unknown dataset_id: '{dataset_id}'.\n"
                f"Not in HF list nor local dataset."
            )
    dataset = list(sorted(dataset, key=lambda x: -1 * x["sample_num"] if "id" not in x else x["id"]))
    print(f"[INFO] Loaded {len(dataset)} items for HF dataset(s): '{dataset_ids}'.")
    return dataset, max_tokens_lower

##############################################################################
# 3. Initialize the vLLM Engine
##############################################################################
def init_llm_engine(
    model_name: str,
    dtype: str,
    speculative_model: Optional[str],
    num_speculative_tokens: int,
    ngram_prompt_lookup_max: int,
    enforce_eager: bool,
    trust_remote_code: bool,
    tensor_parallel_size: int = 1,
    max_model_len: Optional[int] = None,
) -> LLM:
    """
    Initializes a vLLM LLM with or without speculative decoding.
    """
    llm_args = {
        "model": model_name,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
        "trust_remote_code": trust_remote_code,
        "use_v2_block_manager": True,
    }
    if tensor_parallel_size > 1:
        llm_args["tensor_parallel_size"] = tensor_parallel_size
    if max_model_len is not None:
        llm_args["max_model_len"] = max_model_len
    if speculative_model:
        llm_args.update({
            "speculative_model": speculative_model,
            "num_speculative_tokens": num_speculative_tokens,
        })
        if speculative_model == "[ngram]":
            llm_args.update({
                "ngram_prompt_lookup_max": ngram_prompt_lookup_max
            })
    llm = LLM(**llm_args)
    return llm

##############################################################################
# 4. Run Evaluation (Throughput + Accuracy)
##############################################################################
def main_evaluate(
    llm: LLM,
    dataset,
    temperature: float,
    top_p: float,
    max_tokens: int = 32,
    max_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
):
    """
    Runs inference on the dataset, measuring throughput (tokens/sec)
    and a simple exact-match accuracy.
    """
    # Initialize evaluation metrics
    bleu_metric = evaluate.load("bleu")
    bertscore_metric = evaluate.load("bertscore")
    # Prepare sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    # Time the entire generation process
    total_time = 0
    total_tokens_generated = 0
    total_count = 0
    bleu_count = 0
    bert_f1_count = 0
    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset[:max_samples]
    total_batches = 1
    if batch_size is not None:
        total_batches = len(dataset) // batch_size
        total_batches += 1 if len(dataset) % batch_size > 0 else 0
    for epoch_idx in tqdm(range(epochs)):
        for batch_idx in tqdm(range(total_batches)):
            if batch_size is None:
                batch_dataset = dataset
            else:
                if batch_idx == total_batches - 1:
                    batch_dataset = dataset[batch_idx * batch_size:]
                else:
                    batch_dataset = dataset[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            # We'll collect prompts for batch generation as well—vLLM can handle multiple prompts in one call.
            prompts = [ex["prompt"] for ex in batch_dataset]
            start_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            # Stop timing right after generation
            end_time = time.time()
            total_time += end_time - start_time
            for output_obj, example in zip(outputs, batch_dataset):
                # vLLM's .outputs is a list of candidates; we take the first
                generated_text = output_obj.outputs[0].text.strip()
                # Count tokens (rough heuristic: split on whitespace)
                total_tokens_generated += len(generated_text.split())
                # Check if it matches the reference exactly
                ref_answer = example["answer"]
                eval_text = generated_text
                if example['singular']:
                    # Dataset expect single letter answer
                    if len(eval_text) > 1:
                        eval_text = eval_text.split()[0]
    
                # Calculate BLEU
                try:
                    bleu_score = bleu_metric.compute(predictions=[eval_text], references=[ref_answer])["bleu"]
                except:
                    bleu_score = 0
                bleu_count += bleu_score
                # Calculate BERTScore
                bert_results = bertscore_metric.compute(predictions=[eval_text], references=[ref_answer], model_type="bert-base-uncased")
                bert_f1_score = sum(bert_results["f1"]) / len(bert_results["f1"])  # Mean of F1 scores
                bert_f1_count += bert_f1_score
                total_count += 1
                prompt = output_obj.prompt
                #print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\nReference text: {ref_answer!r}\n\n")
    throughput = total_tokens_generated / total_time if total_time > 0 else 0.0
    bleu_accuracy = bleu_count / (len(dataset) * epochs) if len(dataset) > 0 else 0.0
    bert_f1_accuracy = bert_f1_count / (len(dataset) * epochs) if len(dataset) > 0 else 0.0
    accuracy = (bleu_accuracy + bert_f1_accuracy) / 2.0
    return throughput, {"bleu": bleu_accuracy, "bert_f1": bert_f1_accuracy, "final": accuracy}, total_count, total_tokens_generated

##############################################################################
# 5. Main Script - Putting it all Together
##############################################################################
def main():
    parser = argparse.ArgumentParser(description="Run vLLM evaluation with speculative decoding.")
    parser.add_argument("--dataset_id", type=str, default="mmlu", #"cnn_dailymail_share_gpt.json,mmlu,winogrande,superglue_01,superglue_02,superglue_03,superglue_06,superglue_07",
                        help="Name of the dataset to download/prepare/load.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                        help="HF model name or local path to a model checkpoint.")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Datatype to load the model with.")
    parser.add_argument("--trust_remote_code", action="store_true",
                        default=False, help="Should remote code be trusted.")
    parser.add_argument("--speculative_model", type=str, default=None,
                        help="Name of the speculative model (e.g. '[ngram]' or 'facebook/opt-125m').")
    parser.add_argument("--num_speculative_tokens", type=int, default=5,
                        help="Number of speculative tokens to generate in each iteration.")
    parser.add_argument("--ngram_prompt_lookup_max", type=int, default=4,
                        help="Maximum N-gram context for prompt lookup (only used with `[ngram]`).")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling p-value.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Maximum size of batches passed to the engine.")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Maximum model len.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to run.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate on.")
    parser.add_argument("--enforce_eager", type=bool, default=False,
                        help="Enforce eager mode.")
    args = parser.parse_args()
    # Step A: Download/prepare dataset if needed
    maybe_download_and_prepare(args.dataset_id)
    # Step B: Load dataset into memory
    dataset, max_tokens = load_dataset_for_eval(args.dataset_id, args.max_tokens)
    # Step C: Initialize the vLLM engine
    llm = init_llm_engine(
        model_name=args.model_name,
        dtype=args.dtype,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        ngram_prompt_lookup_max=args.ngram_prompt_lookup_max,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    # Step D: Run evaluation
    throughput, accuracy, samples, tokens = main_evaluate(
        llm=llm,
        dataset=dataset,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=max_tokens,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    
    if has_debug_store:
        per_batch_info = debug_store.get("per_batch_info")
        steps_no_spec = 0
        steps_no_prop = 0
        steps_prop = 0
        steps_hit = 0
        average_token_hits_per_step = 0
        if per_batch_info is not None:
            for batch_info in per_batch_info:
                _samples = batch_info["samples"]
                for step_info in batch_info["per_step_info"]:
                    if step_info == -1:
                        steps_no_spec += _samples
                    elif step_info == 0:
                        steps_no_prop += _samples
                    elif step_info == 1:
                        steps_prop += _samples
                    else:
                        steps_hit += _samples
                        average_token_hits_per_step += (step_info - 1) * _samples
        total_steps = steps_no_spec + steps_no_prop + steps_prop + steps_hit
    
    # Step E: Report results
    f = open("/workspace/logs/results.txt", "a")
    a = "========= EVALUATION RESULTS ========="; print(a); f.write(a + "\n")
    a = f"Dataset:                {args.dataset_id}"; print(a); f.write(a + "\n")
    a = f"Model:                  {args.model_name}"; print(a); f.write(a + "\n")
    a = f"Data Type:              {args.dtype}"; print(a); f.write(a + "\n")
    a = f"Speculative Model:      {args.speculative_model}"; print(a); f.write(a + "\n")
    a = f"num_speculative_tokens: {args.num_speculative_tokens}"; print(a); f.write(a + "\n")
    a = f"ngram_prompt_lookup_max:{args.ngram_prompt_lookup_max}"; print(a); f.write(a + "\n")
    a = f"Temperature:            {args.temperature}"; print(a); f.write(a + "\n")
    a = f"Top-p:                  {args.top_p}"; print(a); f.write(a + "\n")
    a = f"Max Tokens:             {max_tokens}"; print(a); f.write(a + "\n")
    a = f"Max Batch Size:         {args.batch_size}"; print(a); f.write(a + "\n")
    a = f"Epochs:                 {args.epochs}"; print(a); f.write(a + "\n")
    a = f"Tensor Parallel Size:   {args.tensor_parallel_size}"; print(a); f.write(a + "\n")
    a = "--------------------------------------"; print(a); f.write(a + "\n")
    a = f"Samples:                {samples}"; print(a); f.write(a + "\n")
    a = f"Tokens:                 {tokens}"; print(a); f.write(a + "\n")
    a = f"Throughput (tokens/s):  {throughput:.2f}"; print(a); f.write(a + "\n")
    a = f"Bleu Accuracy (%):      {100 * accuracy['bleu']:.3f}"; print(a); f.write(a + "\n")
    a = f"Bert F1 Accuracy (%):   {100 * accuracy['bert_f1']:.3f}"; print(a); f.write(a + "\n")
    a = f"Final Accuracy (%):     {100 * accuracy['final']:.3f}"; print(a); f.write(a + "\n")
    if has_debug_store and total_steps > 0:
        a = f"Steps No Spec:          {steps_no_spec} / {total_steps} ({100 * (steps_no_spec / total_steps):.3f}%)"; print(a); f.write(a + "\n")
        a = f"Steps No Prop:          {steps_no_prop} / {total_steps} ({100 * (steps_no_prop / total_steps):.3f}%)"; print(a); f.write(a + "\n")
        a = f"Steps Prop But No Hit:  {steps_prop} / {total_steps} ({100 * (steps_prop / total_steps):.3f}%)"; print(a); f.write(a + "\n")
        a = f"Steps Hit:              {steps_hit} / {total_steps} ({100 * (steps_hit / total_steps):.3f}%)"; print(a); f.write(a + "\n")
        a = f"Average Hits Per Step:  {average_token_hits_per_step / total_steps:.3f}"; print(a); f.write(a + "\n")
    f.write("\n\n")
    f.close()
    


if __name__ == "__main__":
    main()