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

dataset = []
for point in dataset_raw:
    parsed_point = ([], [])
    for c in point[0]:
        parsed_point[0].append(c)
    for c in point[1]:
        parsed_point[1].append(c)
    dataset.append(parsed_point)
