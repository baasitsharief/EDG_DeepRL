import torch
import codecs
import numpy as np


import pandas as pd
import re
import csv
import numpy as np

import time

from sklearn.metrics import f1_score

from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split

from transformers import AdamW, RobertaConfig, RobertaForSequenceClassification

import datetime


class CoherenceClassifier():

	def __init__(self, 
			device,
			model_path = r"D:\Studies\CSE-546\Final_Project\cls_roberta-large_supervised_shuffle.bin",
			batch_size=2):

		self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large', do_lower_case=True)
		self.batch_size = batch_size
		self.device = device

		self.model = RobertaForSequenceClassification.from_pretrained(
			"roberta-large"
		)

		weights = torch.load(model_path)
		self.model.load_state_dict(weights)

		self.model.to(self.device)


	def predict_empathy(self, original_responses, candidate):

		input_ids = []
		attention_masks = []

		for idx, elem in enumerate(original_responses):

			response_sentence = original_responses[idx] + ' </s> ' + candidate

			encoded_dict = self.tokenizer.encode_plus(
								response_sentence,                      # Sentence to encode.
								add_special_tokens = True, # Add '[CLS]' and '[SEP]'
								max_length = 64,           # Pad & truncate all sentences.
								pad_to_max_length = True,
								return_attention_mask = True,   # Construct attn. masks.
								return_tensors = 'pt',     # Return pytorch tensors.
						)
			
			input_ids.append(encoded_dict['input_ids'])
			attention_masks.append(encoded_dict['attention_mask'])

		input_ids = torch.cat(input_ids, dim=0)
		attention_masks = torch.cat(attention_masks, dim=0)

		dataset = TensorDataset(input_ids, attention_masks)

		dataloader = DataLoader(
			dataset, # The test samples.
			sampler = SequentialSampler(dataset), # Pull out batches sequentially.
			batch_size = self.batch_size # Evaluate with this batch size.
		)

		self.model.eval()

		for batch in dataloader:
			b_input_ids = batch[0].to(self.device)
			b_input_mask = batch[1].to(self.device)

			with torch.no_grad():
				(logits, ) = self.model(input_ids = b_input_ids, 
														token_type_ids=None,
														attention_mask=b_input_mask,)

			logits = logits.detach().cpu().numpy().tolist()
			predictions = np.argmax(logits, axis=1).flatten()

		return (logits, predictions)