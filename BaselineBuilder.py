import re
import math
from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.vocab = set()

    def preprocess(self, text): 
        text = re.sub(r'\s+', ' ', text.strip())
        tokens = text.split(' ')
        return ['<s>'] * (self.n - 1) + tokens + ['</s>']

    def train(self, corpus):
        for line in corpus:
            tokens = self.preprocess(line)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.ngrams[context][word] += 1
                self.vocab.add(word)

    def predict(self, context):
        needed = self.n - 1
        if len(context) < needed:
            context = ['<s>'] * (needed - len(context)) + context
        context = tuple(context[-needed:])
        return self.ngrams.get(context, Counter())

    def perplexity(self, corpus):
        log_prob = 0
        word_count = 0
        V = len(self.vocab)

        for line in corpus:
            tokens = self.preprocess(line)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]

                count = self.ngrams[context][word]
                total = sum(self.ngrams[context].values())

                prob = (count + 1) / (total + V)  
                log_prob += math.log(prob)
                word_count += 1

        return math.exp(-log_prob / word_count)

    def accuracy(self, corpus):
        correct = 0
        total = 0

        for line in corpus:
            tokens = self.preprocess(line)
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i + self.n - 1])
                true_word = tokens[i + self.n - 1]

                if context in self.ngrams:
                    predicted = self.ngrams[context].most_common(1)[0][0]
                    if predicted == true_word:
                        correct += 1
                    total += 1

        return correct / total if total > 0 else 0


def build_ngram_model(corpus, n=2, subset_size=5000):
    model = NGramModel(n=n)
    model.train(corpus[:subset_size])
    return model


if __name__ == "__main__":
    with open("kh_CC100.txt", "r", encoding="utf-8") as f:
        corpus = f.readlines()

    for n in [2, 3]:
        model = build_ngram_model(corpus, n=n)
        print(f"\n=== {n}-gram Model ===")
        print(f"Vocabulary size: {len(model.vocab)}")
        print(f"Perplexity: {model.perplexity(corpus[:1000]):.4f}")
        print(f"Accuracy: {model.accuracy(corpus[:1000]):.4f}")

        context = ["សៀវភៅ"]
        predictions = model.predict(context).most_common(5)
        print(f"predictions after '{context[-1]}' ({n}-gram):")
        for word, score in predictions:
            print(f"  {word} ({score})")