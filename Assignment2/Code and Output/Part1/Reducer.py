#!/usr/bin/env python3

import sys
g={}

def read_mapper_output(file):
    for line in file:
        yield line.rstrip()

def main():
	k=read_mapper_output(sys.stdin)
	for i in k:
		i=i.split()
		for l in i:
			#l=l.lower()
			if l in g:
				g[l]+=1
			else:g[l]=1
	for i in g:
		print(i,g[i])


if __name__ == "__main__":
	main()
