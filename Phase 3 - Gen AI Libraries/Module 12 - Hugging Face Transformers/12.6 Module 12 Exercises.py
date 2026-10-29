# 12.6 Module 12 Exercises (Halaman 143)
import time
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TrainingArguments, Trainer
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset
import torch
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# 1. ModelBenchmark comparing two generation pipelines
class ModelBenchmark:
    def __init__(self, pipe1, pipe2, name1: str = "Config A (Greedy)", name2: str = "Config B (Sampling)"):
        self.pipe1 = pipe1
        self.pipe2 = pipe2
        self.name1 = name1
        self.name2 = name2

    def run(self, prompts: list[str]):
        results = []
        for p in prompts:
            t0 = time.perf_counter()
            out1 = self.pipe1(p, max_new_tokens=20, do_sample=False)[0]["generated_text"]
            dur1 = time.perf_counter() - t0

            t1 = time.perf_counter()
            out2 = self.pipe2(p, max_new_tokens=20, do_sample=True, temperature=0.7)[0]["generated_text"]
            dur2 = time.perf_counter() - t1

            results.append({
                "prompt": p[:30],
                "dur1_ms": round(dur1 * 1000, 1),
                "len1": len(out1.split()),
                "dur2_ms": round(dur2 * 1000, 1),
                "len2": len(out2.split()),
            })
        return results


# 2. TokenBudgetChecker
class TokenBudgetChecker:
    def __init__(self, model_name: str = "distilgpt2", max_tokens: int = 25):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_tokens = max_tokens

    def filter_texts(self, texts: list[str]) -> list[tuple[str, int]]:
        passed = []
        for t in texts:
            count = len(self.tokenizer.encode(t))
            if count <= self.max_tokens:
                passed.append((t, count))
        return passed


# 3. extract_entities_from_corpus using NER pipeline
def extract_entities_from_corpus(texts: list[str], ner_pipe) -> dict:
    entity_map = {"PERSON": set(), "ORG": set(), "LOC": set()}
    type_alias = {
        "PER": "PERSON", "B-PER": "PERSON", "I-PER": "PERSON",
        "ORG": "ORG", "B-ORG": "ORG", "I-ORG": "ORG",
        "LOC": "LOC", "B-LOC": "LOC", "I-LOC": "LOC",
    }

    for text in texts:
        entities = ner_pipe(text)
        for ent in entities:
            raw_label = ent.get("entity_group", ent.get("entity", ""))
            clean_label = type_alias.get(raw_label, raw_label)
            if clean_label in entity_map:
                word = ent["word"].strip().replace("##", "")
                if word:
                    entity_map[clean_label].add(word)

    return {k: sorted(list(v)) for k, v in entity_map.items()}


# 4. LoRA fine-tuning loop on custom dataset
def run_custom_lora_finetune(qa_pairs: list[str]):
    model_name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        target_modules=["c_attn"],
        lora_dropout=0.05,
    )
    peft_model = get_peft_model(base_model, lora_cfg)

    def tokenize_fn(batch):
        return tok(batch["text"], truncation=True, max_length=64, padding="max_length")

    ds = Dataset.from_dict({"text": qa_pairs}).map(tokenize_fn, batched=True)
    ds = ds.map(lambda x: {"labels": x["input_ids"]})

    out_dir = os.path.join(os.path.dirname(__file__), "lora_exercise_out")
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=3e-4,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = Trainer(model=peft_model, args=args, train_dataset=ds)
    trainer.train()
    return peft_model, tok


if __name__ == "__main__":
    # --- Exercise 1 ---
    print("[Exercise 1] ModelBenchmark")
    gen_pipe = pipeline("text-generation", model="distilgpt2")
    bench = ModelBenchmark(gen_pipe, gen_pipe)
    prompts = ["Artificial intelligence is", "Deep learning models can", "Vector databases are useful"]
    comparison = bench.run(prompts)
    print(f"{'Prompt':<25} | {'A Latency (ms)':<15} | {'B Latency (ms)':<15}")
    print("-" * 60)
    for row in comparison:
        print(f"{row['prompt']:<25} | {row['dur1_ms']:<15} | {row['dur2_ms']:<15}")

    # --- Exercise 2 ---
    print("\n[Exercise 2] TokenBudgetChecker")
    budget_checker = TokenBudgetChecker(max_tokens=15)
    sample_texts = [
        "Short sentence fits budget.",
        "Another brief statement.",
        "This is an intentionally long sentence designed to exceed the maximum threshold of fifteen tokens.",
    ]
    approved = budget_checker.filter_texts(sample_texts)
    print(f"Passed {len(approved)} of {len(sample_texts)} texts:")
    for text, count in approved:
        print(f" - ({count} tokens) {text}")

    # --- Exercise 3 ---
    print("\n[Exercise 3] Extract Entities from Corpus (NER)")
    ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    corpus = [
        "Google was founded by Larry Page in Mountain View, California.",
        "Microsoft is based in Redmond and led by Satya Nadella.",
    ]
    extracted = extract_entities_from_corpus(corpus, ner)
    for ent_type, values in extracted.items():
        print(f" {ent_type}: {values}")

    # --- Exercise 4 ---
    print("\n[Exercise 4] Custom LoRA Fine-Tuning Loop (20 Q&A pairs)")
    custom_qa = [
        f"Q: What is concept #{i}? A: Concept #{i} is a generative AI pattern."
        for i in range(1, 21)
    ]
    fine_tuned, tok = run_custom_lora_finetune(custom_qa)
    print("Fine-tuning completed successfully. Generating sample response:")
    
    test_in = tok("Q: What is concept #1?", return_tensors="pt")
    with torch.no_grad():
        test_out = fine_tuned.generate(**test_in, max_new_tokens=15, pad_token_id=tok.eos_token_id)
    print(tok.decode(test_out[0], skip_special_tokens=True))
