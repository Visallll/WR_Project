import math
from collections import defaultdict, Counter
from WR_Project.utils import load_corpus, char_tokenize, sentence_split

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, sentences):
        for sent in sentences:
            tokens = ['<s>'] * (self.n - 1) + sent + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                token = tokens[i+self.n-1]
                self.ngrams[context][token] += 1
                self.context_counts[context] += 1
                self.vocab.add(token)

    def prob(self, token, context):
        count = self.ngrams[context][token]
        total = self.context_counts[context]
        if total == 0:
            return 0.0   
        return count / total

    def predict(self, context, k=5):
        context = tuple(context)
        candidates = self.ngrams[context]
        return candidates.most_common(k)

    def perplexity(self, sentences):
        log_prob, N = 0, 0
        for sent in sentences:
            tokens = ['<s>'] * (self.n - 1) + sent + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                token = tokens[i+self.n-1]
                p = self.prob(token, context)
                if p > 0:  
                    log_prob += math.log(p)
                    N += 1
        return math.exp(-log_prob / N) if N > 0 else float("inf")

    def top_k_accuracy(self, sentences, k=5):
        correct, total = 0, 0
        for sent in sentences:
            tokens = ['<s>'] * (self.n - 1) + sent + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                context = tokens[i:i+self.n-1]
                target = tokens[i+self.n-1]
                preds = [w for w, _ in self.predict(context, k)]
                if target in preds:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0


if __name__ == "__main__":
    text = load_corpus("kh_CC100.txt", max_lines=50000)
    sentences = [char_tokenize(s) for s in sentence_split(text)]
    sentences = [s for s in sentences if len(s) >= 1]

    split = int(0.8 * len(sentences))
    train, test = sentences[:split], sentences[split:]

    print("Train:", len(train), "Test:", len(test))

    bigram = NGramModel(2)
    trigram = NGramModel(3)

    bigram.train(train)
    trigram.train(train)

    print("Bigram Perplexity:", bigram.perplexity(test))
    print("Trigram Perplexity:", trigram.perplexity(test))
    print("Bigram Top 5 Accuracy:", bigram.top_k_accuracy(test))
    print("Trigram Top 5 Accuracy:", trigram.top_k_accuracy(test))