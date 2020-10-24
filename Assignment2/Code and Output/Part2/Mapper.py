#!/usr/bin/env python3
import sys
import re
from nltk import ngrams
f = {}
# arthur = open('./arthur.txt')
data = (sys.stdin)
for words in data:
	words = str(words)
	words = words.lower()
	trigram = ngrams(words.split(), 3)
	for gram in trigram:
		trigram_str = str(gram)
		trigram_reg = trigram_str.replace("'", '')
		trigram_reg = trigram_reg.replace(",", '')
		trigram_reg = trigram_reg.replace(' ', '_')
		trigram_reg = trigram_reg.replace('(', '')
		trigram_reg = trigram_reg.replace(')', '')
		req_trigrams = re.compile(r'science|fire|sea')
		req_trigrams = req_trigrams.search(trigram_reg)
		if req_trigrams is not None:
			req_trigrams = req_trigrams.string
			req_trigrams = req_trigrams.replace('science', '$')
			req_trigrams = req_trigrams.replace('fire', '$')
			req_trigrams = req_trigrams.replace('sea', '$')
			f[req_trigrams] = 1
			print(req_trigrams, f[req_trigrams])
