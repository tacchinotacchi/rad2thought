import re
import random

dataset = []
with open("jpn.txt") as file:
    separator = re.compile("(.*)\t(.*)")
    dot = re.compile("")
    pair = dict()
    for line in file:
        match = separator.match(line)
        strings = match.group(1, 2)
        pair["en"] = strings[0]
        pair["ja"] = strings[1]
        dataset.append(pair)

