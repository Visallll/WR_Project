from BaselineBuilder import build_ngram_model

class Smoothing:
    def __init__(self, model):
        self.model = model

    def _prepare_context(self, context):
        needed = self.model.n - 1
        if len(context) < needed:
            context = ['<s>'] * (needed - len(context)) + context
        return tuple(context[-needed:])

    # Laplace smoothing
    def laplace(self, context, word):
        context = self._prepare_context(context)
        count = self.model.ngrams[context][word]
        total = sum(self.model.ngrams[context].values())
        V = len(self.model.vocab)
        return (count + 1) / (total + V)

    # Simplified Kneser–Ney smoothing
    def kneser_ney(self, context, word, d=0.75):
        context = self._prepare_context(context)
        count = self.model.ngrams[context][word]
        total = sum(self.model.ngrams[context].values())

        if total == 0:
            return 0.0

        continuation_count = sum(
            1 for c in self.model.ngrams if word in self.model.ngrams[c]
        )
        continuation_prob = continuation_count / len(self.model.ngrams)

        lambda_factor = d * len(self.model.ngrams[context]) / total

        return max(count - d, 0) / total + lambda_factor * continuation_prob


if __name__ == "__main__":
    with open("kh_CC100.txt", "r", encoding="utf-8") as f:
        corpus = f.readlines()

    model = build_ngram_model(corpus, n=3, subset_size=5000)

    smoother = Smoothing(model)

    context = ["ខ្ញុំ"]
    word = "ស្រលាញ់"

    print(f"Laplace P('{word}' | {context}):",
          smoother.laplace(context, word))

    print(f"Kneser–Ney P('{word}' | {context}):",
          smoother.kneser_ney(context, word))
