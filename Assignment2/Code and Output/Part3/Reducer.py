#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 22:37:31 2020

@author: namratabakre
"""
import sys
import os
import re
import fileinput
d={}


def preprocessing(word):
    word = word.lower()
    alphanumeric = ""
    for character in word:
        if character.isalnum():
            alphanumeric += character
    return alphanumeric

def main():
    for line in sys.stdin:
        (word,filename) = line.strip().split('\t')
        if word not in d:
             d[word] = [filename]
        elif word in d and filename not in d[word]:
             d[word].append(filename)
        elif word in d and filename in d[word]:
             continue
    for word in d:
        print(word,d[word])
                
main()
