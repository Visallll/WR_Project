def load_corpus(path, max_lines=None):
    text = ""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            text += line.strip() + " "
    return text


def sentence_split(text):
    return [s.strip() for s in text.split("។") if len(s.strip()) > 5]


def char_tokenize(sentence):
    return list(sentence)
