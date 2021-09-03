import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', help='replacement idx')
args = vars(parser.parse_args())
args["id"] = int(args["id"])

c = 0
cv = []
for i in os.listdir():
    s = ""
    for j in i[::-1]:
        if j != ".":
            s += j
        else:
            if s[::-1] in ["png", "txt", "jpeg", "jpg"]:
                os.rename(i, f'{args["id"]}' + str(i))
            s = ""

