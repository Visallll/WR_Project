import math
from collections import defaultdict, Counter
from utils import load_corpus, char_tokenize, sentence_split

class LaplaceNGram:
    def __init__(self, n):
        self.n = n
        self.ngrams = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, sentences):
        for sent in sentences:
            tokens = ['<s>'] * (self.n - 1) + sent + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                ctx = tuple(tokens[i:i+self.n-1])
                tok = tokens[i+self.n-1]
                self.ngrams[ctx][tok] += 1
                self.context_counts[ctx] += 1
                self.vocab.add(tok)

    def prob(self, token, context):
        V = len(self.vocab)
        return (self.ngrams[context][token] + 1) / (self.context_counts[context] + V)

    def predict(self, context, k=5):
        context = tuple(context)
        all_tokens = list(self.vocab)
        probs = [(tok, self.prob(tok, context)) for tok in all_tokens]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:k]

    def perplexity(self, sentences):
        log_prob, N = 0, 0
        for sent in sentences:
            tokens = ['<s>'] * (self.n - 1) + sent + ['</s>']
            for i in range(len(tokens) - self.n + 1):
                ctx = tuple(tokens[i:i+self.n-1])
                tok = tokens[i+self.n-1]
                p = self.prob(tok, ctx)
                log_prob += math.log(p)
                N += 1
        return math.exp(-log_prob / N) if N > 0 else float("inf")

    def top_k_accuracy(self, sentences, k=5):
        correct, total = 0, 0
        for sent in sentences:
            tokens = ['<s>'] * (self.n-1) + sent + ['</s>']
            for i in range(len(tokens)-self.n+1):
                ctx = tokens[i:i+self.n-1]
                target = tokens[i+self.n-1]
                preds = [w for w,_ in self.predict(ctx,k)]
                if target in preds:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0


# KNESER-NEY BIGRAM MODEL
class KneserNeyBigram:
    def __init__(self, discount=0.75):
        self.discount = discount
        self.bigram = defaultdict(Counter)
        self.unigram = Counter()
        self.contexts = defaultdict(set)

    def train(self, sentences):
        for sent in sentences:
            tokens = ['<s>'] + sent + ['</s>']
            for i in range(len(tokens)-1):
                w1, w2 = tokens[i], tokens[i+1]
                self.bigram[w1][w2] += 1
                self.unigram[w1] += 1
                self.contexts[w2].add(w1)

    def prob(self, w2, w1):
        cont_count = len(self.contexts[w2])   
        total_contexts = sum(len(v) for v in self.contexts.values()) 
        p_cont = cont_count / total_contexts if total_contexts > 0 else 1e-8

        if self.unigram[w1] > 0:
            lambda_w1 = self.discount * len(self.bigram[w1]) / self.unigram[w1]
            p = max(self.bigram[w1][w2] - self.discount, 0) / self.unigram[w1] + lambda_w1 * p_cont
        else:
            p = p_cont
        return max(p, 1e-8)

    def predict(self, w1, k=5):
        probs = [(w2,self.prob(w2,w1)) for w2 in self.bigram[w1]]
        probs.sort(key=lambda x:x[1], reverse=True)
        return probs[:k]

    def perplexity(self, sentences):
        log_prob, N = 0, 0
        for sent in sentences:
            tokens = ['<s>'] + sent + ['</s>']
            for i in range(len(tokens)-1):
                w1, w2 = tokens[i], tokens[i+1]
                p = self.prob(w2, w1)
                log_prob += math.log(p)
                N += 1
        return math.exp(-log_prob / N) if N > 0 else float("inf")

    def top_k_accuracy(self, sentences, k=5):
        correct, total = 0, 0
        for sent in sentences:
            tokens = ['<s>'] + sent + ['</s>']
            for i in range(len(tokens)-1):
                w1, target = tokens[i], tokens[i+1]
                preds = [w for w,_ in self.predict(w1, k)]
                if target in preds:
                    correct += 1
                total += 1
        return correct / total if total > 0 else 0


# MOCK COMPARISON WITH NEURAL
def compare_with_neural(stat_preds, neural_preds, context):
    print("\nContext:", "".join(context))
    print("Statistical Predictions:", [w for w,_ in stat_preds])
    print("Neural Predictions:", neural_preds)


# MAIN
if __name__ == "__main__":
    text = load_corpus("kh_CC100.txt", max_lines=30000)
    sentences = [char_tokenize(s) for s in sentence_split(text)]
    sentences = [s for s in sentences if len(s) >= 1]

    split = int(0.8*len(sentences))
    train, test = sentences[:split], sentences[split:]

    print("Train:", len(train), "Test:", len(test))

    # Laplace N-gram 
    laplace = LaplaceNGram(3)
    laplace.train(train)
    print("\nLaplace Perplexity:", laplace.perplexity(test))
    print("Laplace Accuracy:", laplace.top_k_accuracy(test))

    # Kneser-Ney Bigram
    kn = KneserNeyBigram()
    kn.train(train)
    print("\nKneser-Ney Perplexity:", kn.perplexity(test))
    print("Kneser-Ney Accuracy:", kn.top_k_accuracy(test))

    # Compare predictions for first 5 test sentences
    print("\n=== Sample Predictions (Laplace vs Kneser-Ney vs Neural) ===")
    for sent in test[:5]:
        context = sent[:2]
        stat_preds = laplace.predict(context, k=5)
        kn_preds = kn.predict(context[-1], k=5)
        neural_preds = ["កុំ","ទៅ","សិក្សា","ញ៉ាំ","សួរ"]  
        compare_with_neural(stat_preds, neural_preds, context)
        print("Kneser-Ney Predictions:", [w for w,_ in kn_preds])
        
