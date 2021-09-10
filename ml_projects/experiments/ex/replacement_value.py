import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--id', help='replacement value')
args = vars(parser.parse_args())
args["id"] = int(args["id"])

cv = []
for i in os.listdir():
    s = ""
    for j in i[::-1]:
        if j != ".":
            s += j
        else:
            if s == "txt":
                cv.append(i)
            s = ""
L = []
for i in cv:
    with open(i) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        for j, val in enumerate(lines):
            sd = f"{args['id']}" + val[1:] + " \n"
            L.append(sd)

        open(i, 'w').close()
        f = open(i, "a")
        f.writelines(L)
        f.close()
        L = []


# 0 0.500000 0.502273 0.327273 0.313636
# 0 0.550000 0.33 0.327273 0.313636

# 0 0.1 0.502273 0.327273 0.313636
