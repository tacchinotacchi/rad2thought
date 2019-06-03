import json

def parse_json(filename):
    with open(filename) as file:
        return json.load(file)

kanji_to_rad = parse_json("kanji2radical.json")
rad_to_kanji = parse_json("radical2kanji.json")