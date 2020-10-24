#!/usr/bin/env python3
import re
import sys

d1={}
d2={}
def read_input(file):
	for line in file:
		line=line.strip()
		yield line.split(",")
		

def main():
	data = read_input(sys.stdin)
	for words in data:
		#Swords=("".join(words)).split(",")
		words=("".join(words)).split(" ")
		words[0]=words[0]+""+words[1]
		words.remove(words[1])
		#print(str(words))
		#print(type(words))
		#d1[words[0]]=words[1:]
		if(words[0] not in d1):
			d1[words[0]]=words[1:]
		else:
			d1[words[0]].extend(words[1:])
	for values in d1:
		print(values,"  ".join(d1[values]))




"""
		#words=list(words)
		#print(type(words))
		#print(len(words))
		word=[]
		k=words[0] + words [1]
		word.append(k)
		word.append(words[2])
		word.append(words[3::])
		print(word)
		print(len(word))


		word=[]
		wor=[]
		w=[]
		k=0
		for i in words:
			word.extend(i.split())
	
	
		k=word[0]+word[1]
		wor.append(k)
		wor.append(word[2::])
"""		
	
		

		














if __name__ == "__main__":
	main()

