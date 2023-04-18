import random

seq = [[random.randint(1, 10) for x in range(4)] for y in range(100)]
x = [random.sample(seq, 10)]

print(x)