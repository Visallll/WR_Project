
# Khmer Next-Word Prediction with GRU

## Project Description

This project implements a **next-word prediction model for the Khmer language** using a **Gated Recurrent Unit (GRU)** neural network in PyTorch.  
The system is trained on large raw Khmer corpora and predicts the most likely next subword token given a preceding text sequence.

Formally, given a sequence of tokens  
x₁, x₂, …, xₜ₋₁,  
the model estimates:

P(xₜ | x₁, x₂, …, xₜ₋₁)

The task is framed as **autoregressive language modeling** and evaluated using **cross-entropy loss** and **perplexity**.

---

## Execution Environment and Storage

The notebook is designed to run in **Google Colab** with **Google Drive mounted** as persistent storage.  
This is required because:

- the Khmer corpora are large,
- training is long-running,
- trained models and tokenizers must persist across sessions.

When executed locally, the Drive mount step can be skipped.

---

## Dataset Overview

Two Khmer text corpora are used:

- **CC100 Khmer corpus**: primary large-scale training data  
- **OSCAR Khmer corpus**: optional secondary corpus

Both datasets consist of raw web text and contain significant noise, requiring preprocessing.

---

## Khmer Text Normalisation

Raw Khmer web text contains:

- mixed Latin and Khmer scripts,
- digits and punctuation,
- inconsistent Unicode representations.

Before tokenisation, each line of text is normalised using:

- Unicode NFC normalisation,
- removal of Latin characters and digits,
- whitespace cleanup,
- preservation of Khmer Unicode characters.

This step reduces vocabulary explosion and stabilises training.

---

## Streaming Corpus Processing

The corpora are processed using a **streaming approach**:

- files are read line-by-line,
- each line is normalised and filtered,
- data is never fully loaded into memory.

This design enables scalability within limited RAM environments.

---

## Subword Tokenisation with SentencePiece

Khmer does not use whitespace to separate words, making word-level tokenisation unreliable.  
A **SentencePiece BPE tokenizer** is therefore trained directly on the normalised corpus.

The tokenizer:

- learns subword units statistically,
- avoids explicit word segmentation,
- balances vocabulary size and expressiveness.

The same tokenizer is reused for training, evaluation, and inference.

---

## Tokeniser Verification

After training, the SentencePiece model is loaded and verified by checking:

- vocabulary size,
- special token IDs,
- encode–decode consistency.

Tokeniser correctness is critical because errors propagate directly into the model.

---

## Sequence Construction and Dataset Splitting

The tokenised stream is segmented into **fixed-length token sequences**.

Each sequence is split into:

- input tokens: x₁, …, xₜ₋₁  
- target token: xₜ

Sequences are deterministically assigned to training, validation, and test sets using a **hash-based splitting strategy** to prevent data leakage.

---

## Dataset Implementation

A PyTorch `IterableDataset` is used to represent the streaming token sequences.  
This design avoids random access and supports large datasets efficiently.

---

## DataLoader Configuration

PyTorch DataLoaders batch token sequences and feed them efficiently to the GPU.  
Batching and streaming are coordinated to maintain stable training.

---

## Model Architecture

The language model consists of:

1. an embedding layer,
2. a GRU recurrent layer,
3. a linear projection layer over the vocabulary.

GRUs are chosen for their efficiency and ability to model long-term dependencies with fewer parameters than LSTMs.

---

## GRU Computation

At each time step t, the GRU computes:

Update gate:  
zₜ = σ(Wz xₜ + Uz hₜ₋₁)

Reset gate:  
rₜ = σ(Wr xₜ + Ur hₜ₋₁)

Candidate state:  
h̃ₜ = tanh(Wh xₜ + Uh (rₜ ⊙ hₜ₋₁))

Final hidden state:  
hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ

The hidden state is projected to vocabulary logits for next-token prediction.

---

## Training Objective

The model is trained using **cross-entropy loss**:

L = − log P(xₜ | x₁, x₂, …, xₜ₋₁)

Training uses teacher forcing.

---

## Evaluation Metric

Model performance is measured using **perplexity**:

PPL = exp(L)

Lower perplexity indicates better predictive performance.

---

## Training Procedure

Training includes:

- GPU acceleration,
- mixed-precision training,
- gradient clipping,
- validation-based checkpointing.

---

## Test Evaluation

Final evaluation is performed on a held-out test set that is never used for model selection.

---

## Training Visualisation

Loss and perplexity curves are plotted to analyse convergence, stability, and overfitting.

---

## Inference and Decoding

During inference, the model predicts next tokens using:

- temperature scaling,
- top-k sampling.

Temperature-scaled probabilities:

Pτ(x) = Softmax(z / τ)

---

## Qualitative Analysis

Generated predictions are inspected qualitatively to assess Khmer fluency, orthographic correctness, and semantic plausibility.

---

## Conclusion

This project presents a **streaming, subword-based GRU language model for Khmer** built directly from raw web corpora.  
The implementation prioritises scalability, reproducibility, and suitability for low-resource language modeling.
