#!/usr/bin/env python3
import sys
import operator
g={}
# reduce = open('reduce.txt')

def read_mapper_output(file):
	for line in file:
		yield line.rstrip()

def main():
	k=read_mapper_output((sys.stdin))
	for i in k:
		i=i.split()
		for l in i:
			if l in g:
				g[l]+=1
			else:g[l]=1
	sorted_dict = sorted(g.items(), key=operator.itemgetter(1), reverse=True)
	sorted_dict = sorted_dict[1:11]
	for i in sorted_dict:
		i = str(i)
		i = i.replace("'", '')
		i = i.replace(",", '')
		i = i.replace('(', '')
		i = i.replace(')', '')
		print(i)

if __name__ == "__main__":
	main()
