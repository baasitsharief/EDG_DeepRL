'''I/we certify that the code and data in this assignment were generated
independently, using only the tools and resources defined in the course
and that I/we did not receive any external help, coaching or contributions
during the production of this work."

References: 
1. https://github.com/lvwerra/trl/blob/master/nbs/04-gpt2-sentiment-ppo-training.ipynb
2. https://github.com/behavioral-data/Empathy-Mental-Health'''

import torch
from itertools import chain
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import copy as deepcopy

from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_polynomial_decay_schedule_with_warmup

def load_tokenizer():
  print("Loading the tokenizer...")
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
  special_tokens = {
      'bos_token': "<bos>",
      'additional_special_tokens': ["<sp1>", "<sp2>"]
  }
  eos_token = tokenizer.eos_token
  num_new_tokens = tokenizer.add_special_tokens(special_tokens)
  vocab = tokenizer.get_vocab()
  tokenizer.pad_token = tokenizer.eos_token
  vocab_size = len(vocab)
  bos_id = vocab["<bos>"]
  eos_id = vocab[eos_token]
  sp1_id = vocab["<sp1>"]
  sp2_id = vocab["<sp2>"]
  return {
      'tok': tokenizer,
      'bos_id': bos_id,
      'eos_id': eos_id,
      'sp1_id': sp1_id,
      'sp2_id': sp2_id,
      'vocab_size': vocab_size
      } 

def load_model(ckpt_path, vocab_size, device='cpu'):
  print("Loading Model...")
  model = GPT2LMHeadModel.from_pretrained("gpt2").to(torch.device(device))
  model.resize_token_embeddings(vocab_size)
  ckpt = torch.load(ckpt_path, map_location=torch.device(device))
  model.load_state_dict(ckpt['model_state_dict'])
  return model

def nucleus_sampling(model, input_ids, token_ids, tokenizer_dict, input_len, max_len=1024, device='cpu'):
  out_ids = []
  for pos in range(input_len, max_len):
    out = model(input_ids = input_ids, token_type_ids=token_ids)[0][:, pos-1]
    out = F.softmax(out, dim=-1)
    
    probabilities_desc, sorted_indices_desc = torch.sort(input=out, descending=True)
    # print(f"probs: {probabilities_desc, sorted_indices_desc}")
    removed_indices = torch.cumsum(input= probabilities_desc, dim=-1) > 0.92
    removed_indices[:, 1:] = removed_indices[:, :-1].clone()
    removed_indices[:, 0] = False

    probabilities_desc[removed_indices] = 0.0
    probabilities_desc /= torch.sum(input=probabilities_desc, dim=-1, keepdim=True)

    probabilities = torch.zeros(size = out.shape, device=torch.device(device)).scatter(dim=-1,  index= sorted_indices_desc, src= probabilities_desc)
    index = torch.multinomial(probabilities, 1)

    index_item = index.squeeze(-1).squeeze(-1).item()
    out_ids.append(index_item)
    if index_item == tokenizer_dict['eos_id']:
      break

    input_ids = torch.cat((input_ids, index), dim=-1)
    next_ids = torch.LongTensor([[tokenizer_dict['sp2_id']]]).to(torch.device(device))

    token_ids = torch.cat((token_ids, next_ids), dim=-1)

  return out_ids

# tokenizer
def next_sentence(model, seeker_post, tokenizer_dict, input_history=[], end_command="Abort!", max_hist = 5, device = "cpu"):
  if(seeker_post==end_command):
    return "Goodbye!", input_history
  tokenizer = tokenizer_dict['tok']
  input_ids = [tokenizer_dict['sp1_id']]+tokenizer.encode(seeker_post)
  input_history.append(input_ids)

  if(len(input_history)>=max_hist):
    input_hists = input_history[len(input_history)-max_hist+1:]
  else:
    input_hists = input_history
  input_ids = [tokenizer_dict['bos_id']]+list(chain.from_iterable(input_hists))+[tokenizer_dict['sp2_id']]
  start_sp_id = input_hists[0][0]
  next_sp_id = tokenizer_dict['sp1_id'] if start_sp_id == tokenizer_dict['sp2_id'] else tokenizer_dict['sp2_id']
  token_type_ids = [[start_sp_id] * len(hist) if h % 2 == 0 else [next_sp_id] * len(hist) for h, hist in enumerate(input_hists)]
  token_type_ids = [start_sp_id] + list(chain.from_iterable(token_type_ids)) + [tokenizer_dict['sp2_id']]
  input_len = len(input_ids)

  input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(torch.device(device))
  token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0).to(torch.device(device))

  # print(f"token_type_ids: {token_type_ids.shape}")

  output_ids = nucleus_sampling(model, input_ids, token_type_ids, tokenizer_dict, input_len, device=torch.device(device))
  res = tokenizer.decode(output_ids, skip_special_tokens=True)
  input_history.append([tokenizer_dict["sp2_id"]] + tokenizer.encode(res))

  return res, input_history

def next_sentence_eval(model, seeker_post, tokenizer_dict, input_history=[], end_command="Abort!", max_hist = 5, device = "cpu"):
  with torch.no_grad():
    res, input_history = next_sentence(model, seeker_post, tokenizer_dict, input_history, end_command, max_hist, device)
  return res, input_history