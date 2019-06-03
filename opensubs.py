import xml.etree.cElementTree as ET

class DatasetParser():
    def __init__(self):
        super().__init__()
        self.reading_pair = False
        self.current_pair = {"en": None, "ja": None}
        self.current_lang = None
        self.pair_list = []
    def start_reading_pair(self, tag, attrs):
        if tag == "tuv":
            self.current_lang = attrs["{http://www.w3.org/XML/1998/namespace}lang"]
        elif self.current_lang and tag == "seg":
            pass
        else:
            raise ValueError("Tag other than tuv or seg started while reading pair")
    def start(self, tag, attrs=None):
        if self.reading_pair:
            return self.start_reading_pair(tag, attrs)
        elif tag == "tu":
            self.reading_pair = True
    def data(self, data):
        if self.reading_pair and self.current_lang:
            self.current_pair[self.current_lang] = data
    def end_reading_sample(self, tag, attrs):
        if tag == "tuv":
            if not self.current_pair[self.current_lang]:
                raise ValueError("Ending sample but value is false")
            self.current_lang = None
        elif tag == "seg":
            pass
        else:
            raise ValueError("Tag other than tuv or seg ended while reading sample")
    def end_reading_pair(self, tag, attrs):
        if tag == "tu":
            if not all(self.current_pair.items()):
                raise ValueError("Ending pair but there are false values")
            self.pair_list.append(self.current_pair)
            self.current_pair = {"en": None, "ja": None}
            self.reading_pair = False
        else:
            raise ValueError("Tag other than tu ended while reading pair")
    def end(self, tag, attrs=None):
        if self.current_lang:
            return self.end_reading_sample(tag, attrs)
        elif self.reading_pair:
            return self.end_reading_pair(tag, attrs)
    def close(self):
        return self.pair_list

def parse_file(filename):
    custom_builder = DatasetParser()
    parser = ET.XMLParser(target=custom_builder)
    return ET.parse(filename, parser).getroot()

dataset = parse_file("en-ja.tmx")
