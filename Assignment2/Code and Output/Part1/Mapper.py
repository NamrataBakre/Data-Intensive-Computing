#!/usr/bin/env python3
import sys
import re
f={}

def read_input(files):
	for lines in files:
		yield lines.split()

def main():
	data = read_input((sys.stdin))
	
	for awa in data:
		awa=str(awa)
		awa1=re.findall('[a-zA-Z]+',awa)
		for k in awa1:
			f[k]=1
			print(k,f[k])
				
"""
		
		for k in str(awa).split():
			awa1=re.findall('[a-zA-Z]+',k)
			awa1=str(k)
			if(str(awa1)!=""):			
				print(awa1)

	for awa in data:
		print(awa)

		for words in awa1:
			f[words]=1
			print( words,f[words]) 
"""

	
if __name__ == "__main__":
	main()
