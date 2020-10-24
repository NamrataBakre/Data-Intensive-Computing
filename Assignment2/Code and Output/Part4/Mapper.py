#!/usr/bin/env python3
import re
import sys

def read_input(file):
	for line in file:
		line=line.strip()
		yield line.split("\n")
		

def main():

	data = read_input(sys.stdin)
	for words in data:
		#print(words)
		word=[]
		for i in words:
			word.extend(i.split(","))
		#print(word)
		#print(type(word),len(word))


		name,country,salary,Id,code= "0",0,"0","0","0"
		if(len(word)==5):
			Id=word[0]
			salary=word[1]+word[2]
			#salary2=words[2]
			country=word[3]
			code=word[4]
			print(Id,salary,country,code)
	
		elif(len(word)==6):
			Id=word[0]
			salary=word[1]+word[2]
			#salary2=words[2]
			country=word[3]+word[4]
			code=word[5]
			print(Id,salary,country,code)
		else:
			Id=word[0]
			name=word[1]
			print(Id, name)

		#print(Id,name,salary,country,code)


if __name__ == "__main__":
	main()



