import time
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from MagicDec.Engine.utils import setup_seed, sampling_argmax_batch, cuda_graph_for_sampling_argmax_batch
from MagicDec.Data.data_converter import convert_pg19_dataset
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from MagicDec.Engine.SnapKV.backend import LMBackend

parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=Path, default=Path("./checkpoints/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')

parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=3969, help='Prefix length')
parser.add_argument('--max_len', type=int, default=128, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--rank_group', nargs='+', type=int, default=[0], help='Target group of ranks')
parser.add_argument('--printoutput', action='store_true', default=True, help='Whether to compile the model.')

args = parser.parse_args()
# assert args.prefix_len < args.max_len
assert args.max_len % 128 == 0 # why 128?

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device={DEVICE}")
# MAX_LEN = args.prefix_len + args.gen_len - 1
MAX_LEN = args.max_len
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
checkpoint_path = args.model

engine = LMBackend(dtype=DTYPE, device=DEVICE)
engine.load_model(checkpoint_path, use_tp=False, rank_group = args.rank_group, group=None) # what happend behind this
if args.compile:
    engine.compile()

engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN) # kv size predefined useing paged attn

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")

# dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
# num_eval_steps = min(6, len(dataloader))

total_time = 0.0
model_steps = 0

prefill = "once upon a time"
input_ids = tokenizer(prefill, return_tensors="pt").input_ids.to(DEVICE)

# input_ids = batch[0].to(DEVICE)
terminate = False
output = input_ids.clone()

next_tokens = engine.encode(input_ids=input_ids)[:,-1:]
output = torch.cat((output, next_tokens),dim=-1)
torch.cuda.synchronize()
t1 = time.perf_counter()
while output.size(1)<MAX_LEN and terminate == False:
    input_ids=next_tokens.clone()
    next_tokens = engine.inference(input_ids=input_ids) # next_token(s)? is this speculative decoding?
    print(f'next_tokens: {next_tokens} : {tokenizer.decode(next_tokens.squeeze(0))}') # from print, all just contain one token
    output = torch.cat((output, next_tokens),dim=-1)
    model_steps += 1
    if (next_tokens[:,-1] == eot_1)._is_any_true() or (next_tokens[:,-1] == eot_2)._is_any_true(): terminate = True
torch.cuda.synchronize()
t2=time.perf_counter()
total_time += t2 - t1

if args.printoutput:
    for i in range(BATCH_SIZE):
        print("---++++--")
        print(tokenizer.decode(output[i, args.prefix_len:]))
        print("-----")
        print(tokenizer.decode(output[i, :]))
        print("---++++--")
print(f"Tokens per second :{BATCH_SIZE*(model_steps/total_time)}") # why useing steps to count 