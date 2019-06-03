import numpy as np
import string
from radicals import kanji_to_rad, rad_to_kanji
import opensubs
import language_learning
import time
import random

hiragana_katakana = [
    'ー',
    '〜',
    '～',
    'ぁ',
    'あ',
    'ぃ',
    'い',
    'ぅ',
    'う',
    'ぇ',
    'え',
    'ぉ',
    'お',
    'か',
    'が',
    'き',
    'ぎ',
    'く',
    'ぐ',
    'け',
    'げ',
    'こ',
    'ご',
    'さ',
    'ざ',
    'し',
    'じ',
    'す',
    'ず',
    'せ',
    'ぜ',
    'そ',
    'ぞ',
    'た',
    'だ',
    'ち',
    'ぢ',
    'っ',
    'つ',
    'づ',
    'て',
    'で',
    'と',
    'ど',
    'な',
    'に',
    'ぬ',
    'ね',
    'の',
    'は',
    'ば',
    'ぱ',
    'ひ',
    'び',
    'ぴ',
    'ふ',
    'ぶ',
    'ぷ',
    'へ',
    'べ',
    'ぺ',
    'ほ',
    'ぼ',
    'ぽ',
    'ま',
    'み',
    'む',
    'め',
    'も',
    'ゃ',
    'や',
    'ゅ',
    'ゆ',
    'ょ',
    'よ',
    'ら',
    'り',
    'る',
    'れ',
    'ろ',
    'ゎ',
    'わ',
    'を',
    'ん',
    'ァ',
    'ア',
    'ィ',
    'イ',
    'ゥ',
    'ウ',
    'ェ',
    'エ',
    'ォ',
    'オ',
    'カ',
    'ガ',
    'キ',
    'ギ',
    'ク',
    'グ',
    'ケ',
    'ゲ',
    'コ',
    'ゴ',
    'サ',
    'ザ',
    'シ',
    'ジ',
    'ス',
    'ズ',
    'セ',
    'ゼ',
    'ソ',
    'ゾ',
    'タ',
    'ダ',
    'チ',
    'ヂ',
    'ッ',
    'ツ',
    'ヅ',
    'テ',
    'デ',
    'ト',
    'ド',
    'ナ',
    'ニ',
    'ヌ',
    'ネ',
    'ノ',
    'ハ',
    'バ',
    'パ',
    'ヒ',
    'ビ',
    'ピ',
    'フ',
    'ブ',
    'プ',
    'ヘ',
    'ベ',
    'ペ',
    'ホ',
    'ボ',
    'ポ',
    'マ',
    'ミ',
#    'マ', also a radical
    'メ',
    'モ',
    'ャ',
    'ヤ',
    'ュ',
    'ユ',
    'ョ',
    'ヨ',
    'ラ',
    'リ',
    'ル',
    'レ',
    'ロ',
    'ヮ',
    'ワ',
    'ヲ',
    'ン',
    'ヴ',
    'ヵ',
    'ヶ',
    '”',
    '。',
    '！',
    '？',
    '、',
    '・',
    '－',
    '（',
    '）',
    '「',
    '」',
    '＝',
    '：',
    '．',
    '　',
    '—',
    '＋',
    '，',
    '％',
    '々',
    'ａ',
    'ｂ',
    'ｃ',
    'ｄ',
    'ｅ',
    'ｆ',
    'ｇ',
    'ｈ',
    'ｉ',
    'ｊ',
    'ｋ',
    'ｌ',
    'ｍ',
    'ｎ',
    'ｏ',
    'ｐ',
    'ｑ',
    'ｒ',
    'ｔ',
    'ｓ',
    'ｕ',
    'ｖ',
    'ｗ',
    'ｘ',
    'ｙ',
    'ｚ',
    'Ａ',
    'Ｂ',
    'Ｃ',
    'Ｄ',
    'Ｅ',
    'Ｆ',
    'Ｇ',
    'Ｈ',
    'Ｉ',
    'Ｊ',
    'Ｋ',
    'Ｌ',
    'Ｍ',
    'Ｎ',
    'Ｏ',
    'Ｐ',
    'Ｑ',
    'Ｒ',
    'Ｔ',
    'Ｓ',
    'Ｕ',
    'Ｖ',
    'Ｗ',
    'Ｘ',
    'Ｙ',
    'Ｚ',
    '０',
    '１',
    '２',
    '３',
    '４',
    '５',
    '６',
    '７',
    '８',
    '９'
]
base_charset = [c for c in string.printable] + [
    '€',
    'ñ',
    'é',
    '’',
    '℃',
    '―'
]

en_charset_l = ['<padding>'] + ['<unk>'] + base_charset
ja_charset_l = ['<padding>'] + ['<unk>'] + base_charset + list(rad_to_kanji.keys()) + hiragana_katakana
en_charset = {val: index for index, val in enumerate(en_charset_l)}
ja_charset = {val: index for index, val in enumerate(ja_charset_l)}

def expand_radicals(sentence):
    expanded = []
    for ch in sentence:
        if ch in kanji_to_rad:
            expanded.extend(kanji_to_rad[ch])
        else:
            expanded.append(ch)
    return expanded

def encode_sentence(sentence, charset):
    tokens = np.empty(len(sentence) + 2, np.int64)
    tokens[0] = len(charset)
    for index, ch in enumerate(sentence):
        tokens[index + 1] = charset[ch]
    tokens[-1] = len(charset) + 1
    return tokens

def decode_sentence(seq, charset):
    seq = np.trim_zeros(seq)
    return ''.join(map(lambda t: charset[int(t)], seq[1:-1]))

def process_dataset(data):
    dataset = {
        "en": [], "ja": [], "ja_expand": [], "en_token": [], "ja_token": []
    }
    error_pairs = 0
    for d in data:
        try:
            ja_expand = expand_radicals(d["ja"])
            en_token = encode_sentence(d["en"], en_charset)
            ja_token = encode_sentence(ja_expand, ja_charset)
        except KeyError as e:
            error_pairs += 1
            continue
        dataset["en"].append(d["en"])
        dataset["ja"].append(d["ja"])
        dataset["ja_expand"].append(ja_expand)
        dataset["en_token"].append(en_token)
        dataset["ja_token"].append(ja_token)
    print(error_pairs)
    return dataset

def split_dataset(dataset, percentage):
    retain_size = int(len(dataset["en"]) * percentage)
    first, second = {k: None for k in dataset.keys()}, {k: None for k in dataset.keys()}
    for key in dataset.keys():
        first[key] = dataset[key][:retain_size]
        second[key] = dataset[key][retain_size:]
    return first, second

def shuffle_dataset(dataset):
    indeces = list(range(len(dataset["en"])))
    random.shuffle(indeces)
    for key in dataset.keys():
        dataset[key] = [dataset[key][i] for i in indeces]

def merge_datasets(datasets):
    merged = {
        "en": [], "ja": [], "ja_expand": [], "en_token": [], "ja_token": []
    }
    for d in datasets:
        for key in d.keys():
            merged[key].extend(d[key])
    return merged


start = time.time()
datasets_list = []
datasets_list += [process_dataset(opensubs.dataset)]
datasets_list += [process_dataset(language_learning.dataset)]
dataset = merge_datasets(datasets_list)
taken = time.time() - start
print(taken)