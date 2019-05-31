import re
import random

dataset_raw = []
with open("jpn.txt") as file:
    separator = re.compile("(.*)\t(.*)")
    dot = re.compile("")
    for line in file:
        match = separator.match(line)
        strings = match.group(1, 2)
        dataset_raw.append(match.group(1, 2))
