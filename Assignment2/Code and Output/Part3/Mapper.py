#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:36:04 2020

@author: namratabakre
"""

import sys
import os
import fileinput
import re

def get_filename():
	filename = os.environ['map_input_file']
	parts = filename.split('/') #This list will store list of strings after breaking the filepath
	length = len(parts)
	filename = parts[length - 1]
	return filename


def preprocessing(word):
    word = word.lower()
    alphanumeric = ""
    for character in word:
        if character.isalnum():
            alphanumeric += character
    return alphanumeric


def main():
	for line in sys.stdin:
		words = line.strip().split()
		for word in words:
			filename = get_filename()			
			word = preprocessing(word)
			word = re.sub('[^A-Za-z0-9]+', '', word)
			if word:
				print ('%s\t%s' % (word,filename))

main()
