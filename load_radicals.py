import re

def add_radicals(store, new):
    for r in new:
        if r not in store:
            store.append(r)

radicals_list = []
kanji_dict = dict()
with open("kradfile", encoding="euc-jp") as file:
    noncomment = re.compile("#")
    extract = re.compile("([^:]) : (.*)")
    for line in file:
        if not noncomment.match(line):
            left, right = extract.match(line).group(1, 2)
            radicals = right.split()
            add_radicals(radicals_list, radicals)
            kanji_dict[left] = radicals

