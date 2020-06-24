import numpy as np
test = open('data/final-dataset -.txt')
lines = test.readlines()
doc = open('1.txt','a')

count = 0
for lines in lines:
    lines = lines.strip('\n')
    if "Normal" in lines:
        count = count + 1
    else:
        print(lines, file=doc)
    if count == 8:
        print(lines, file=doc)
        count = 0




