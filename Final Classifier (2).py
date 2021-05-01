#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import torch
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import nltk
import os
import pandas as pd
from numpy import arange
import csv
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import svm

#############################################################################################

def createDataset(comments):

	#dataset = [['comment_id', 'comment_length', 'punct_score', 'chars_per_word', 'class']]
	
	dataset = []

	for num in range(len(comments)):
	
		word_count = 0
		char_count = 0
		
		punct_list = [0, 0, 0]

		for word in comments.iloc[num].split(' '):

			word_count += 1

			for char in word:

				char_count += 1

				if char == '.':
					punct_list[0] += 1
				elif char == '!':
					punct_list[1] += 1
				elif char == '?':
					punct_list[2] += 1
					
		chars_per_word = char_count / word_count
			
				
    
		# A high punct_score means that there are a lot of periods, not many question
		# marks or exclaimations.
		punct_score = punct_list[0] / (punct_list[1] + punct_list[2] + 0.1)
    
		dataset.append([num, len(comments.iloc[num]), punct_score, chars_per_word])
		
	return pd.DataFrame(dataset)
	
	
#############################################################################################


def predict(model, lr_clf, svc, poly_svc, post):

	post_list = [post]
	
	post_series = pd.Series(post_list)	

	
	tokenized = post_series.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
	
	max_len = 0
	for i in tokenized.values:
		if len(i) > max_len:
			max_len = len(i)

	padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

	attention_mask = np.where(padded != 0, 1, 0)

	input_ids = torch.tensor(padded)  
	attention_mask = torch.tensor(attention_mask)

	with torch.no_grad():
		last_hidden_states = model(input_ids, attention_mask=attention_mask)
	
   	
	post_berted = last_hidden_states[0][:,0,:].numpy()


	
	post_df = pd.DataFrame(post_list)
	
	post_features = createDataset(post_series)
	
	bert_pred = lr_clf.predict(post_berted)
	svc_pred = svc.predict(post_features)
	poly_pred = poly_svc.predict(post_features)
	
	score = 0
		
	if bert_pred == 1:
		score += 2.1
			
	if svc_pred == 1:
		score += 1
			
	if poly_pred == 1:
		score += 1
			
	if score > 1.9:
		return 'Excellent Mentorship!'
	else:
		return 'Average Mentorship'

#############################################################################################	
	
good = pd.read_csv('/storage/home/tum224/exemplary_mentorship.csv')
bad_messages = pd.read_csv('/storage/home/tum224/average_mentorship.csv')

good = list(good.text)
bad_messages = list(bad_messages.text)

good_messages = good * 5

# Label 1 for good mentorship, 0 for bad
labels = []

for i in range(len(good_messages)):
	labels.append(1)

for i in range(len(bad_messages)):
	labels.append(0)
		
data = pd.DataFrame()
data['text'] = good_messages + bad_messages
data['class'] = labels

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['class'])

X_train_data = createDataset(X_train)
X_test_data = createDataset(X_test)

C = 1 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train_data, list(y_train))
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train_data, y_train)
	
svc_y_pred = svc.predict(X_test_data)
	
poly_y_pred = poly_svc.predict(X_test_data)
	

data = pd.DataFrame()

data['text'] = X_train

data['class'] = y_train

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
	
# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
	
tokenized = data['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
	
max_len = 0
for i in tokenized.values:
	if len(i) > max_len:
	    max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)


with torch.no_grad():
	last_hidden_states = model(input_ids, attention_mask=attention_mask)
    	
features = last_hidden_states[0][:,0,:].numpy()

lr_clf = LogisticRegression(max_iter=500)
lr_clf.fit(features, y_train)
	
	
tokenized = X_test.apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
	
max_len = 0
for i in tokenized.values:
	if len(i) > max_len:
		max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])

attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded)  
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
	last_hidden_states = model(input_ids, attention_mask=attention_mask)
	
   	
X_test_features = last_hidden_states[0][:,0,:].numpy()


bert_y_pred = lr_clf.predict(X_test_features)

	
ensemble_predictions = []

for num in range(len(bert_y_pred)):

	votes = 0
		
	if bert_y_pred[num] == 1:
		votes += 2.1
			
	if svc_y_pred[num] == 1:
		votes += 1
			
	if poly_y_pred[num] == 1:
		votes += 1
			
	if votes > 1:
		ensemble_predictions.append(1)
	else:
		ensemble_predictions.append(0)
			
			
choice = input("Enter 0 to train, or enter 1 to classify an individual post:")

if choice == '0':
	
	print('Accuracy: ', accuracy_score(ensemble_predictions, y_test))

	print('F1: ', f1_score(ensemble_predictions, y_test))

	print('Precision: ', precision_score(ensemble_predictions, y_test))

	print('Recall: ', recall_score(ensemble_predictions, y_test))

else:

	post = input("Enter post: ")
	
	pred = predict(model, lr_clf, svc, poly_svc, post)
	
	print(pred)
		

