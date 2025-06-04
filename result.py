import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
import gc
from hqq.core.quantize import *
from hqq_utils import AutoHQQHFModel
from peft import PeftModel, PeftConfig

torch.cuda.empty_cache()
torch.cuda.ipc_collect()
gc.collect()
torch.cuda.empty_cache()

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        # Prefill
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)

    nsamples = test_enc.numel() // model.seqlen
    nlls = []
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]

        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    return ppl.item()

def get_quant_config_slm(model):
    quant_config = {}

    n_layers = model.config.num_hidden_layers
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=128)
    q2_config = BaseQuantizeConfig(nbits=2, group_size=32)

    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q2_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config

        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q2_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config



    return quant_config

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)

    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'

    ### === TODO: Load your model (you may change this part) ===
    # 載入 LoRA 配置
    peft_config = PeftConfig.from_pretrained("./checkpoint-3000")

    # 載入 base model（原始 LLaMA 模型）
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    # 加載 LoRA adapter
    model = PeftModel.from_pretrained(model, "./checkpoint-3000")
    model = model.merge_and_unload()  # 合併 LoRA 到原始模型

    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

    quant_config = get_quant_config_slm(model)

    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    from hqq.utils.patching import prepare_for_inference
    # prepare_for_inference(model, backend='gemlite')
    prepare_for_inference(model, backend='bitblas')
    torch.cuda.empty_cache()

    #####################################

    model.eval()

    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)


    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # === (Optional) Set up StaticCache for manual KV cache management ===
    from transformers import StaticCache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_new_tokens + 16,
        device=model.device,
        dtype=torch.float16
    )
    ####################################################################

    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up ===
        # _ = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )

        # === (Optional) Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing ===
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )

        # === Optional: Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = max_new_tokens / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)

    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')

    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")

    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])

if __name__ == '__main__':
    main()
