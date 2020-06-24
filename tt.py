count = 0
test = open("1.txt", 'r')
lines = test.readlines()

l = len(lines)
l = 0.9*l
i = 0

print(l)
for lines in lines:
    i = i + 1
    if "Normal" in lines:
        count = count +1
    if i > l:
        break

print(count)
