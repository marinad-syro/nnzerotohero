# LEARN

## `str.splitlines()`
**What it is:** A Python string method that cuts a string at every line break and returns a list of the lines, with the line-break characters removed.
**Why we're using it:** `names.txt` has one name per line, and we want a Python list of names rather than one giant string.
**How it works here:** In `lectures/2_makemore_bigram.ipynb`, `open('names.txt').read()` gives one string like `"emma\nolivia\nava\n"`, and `.splitlines()` turns it into `['emma', 'olivia', 'ava']`.

```python
words = open('names.txt').read().splitlines()

"emma\nolivia\nava\n".splitlines()   # ['emma', 'olivia', 'ava']
"emma\nolivia\nava\n".split('\n')    # ['emma', 'olivia', 'ava', '']  <- extra empty string
"a\nb".splitlines(keepends=True)     # ['a\n', 'b']  <- keeps the line breaks
```

It differs from `.split('\n')` in two ways: it leaves no trailing empty string when the file ends with a newline, and it also handles Windows-style `\r\n` line endings.

## `zip()`
**What it is:** A built-in Python function that walks through several sequences in parallel and hands back tuples of the items at the same position.
**Why we're using it:** A bigram is a character plus the character after it, so we zip a word with a copy of itself shifted by one.
**How it works here:** In `lectures/2_makemore_bigram.ipynb`, `zip(w, w[1:])` on `"emma"` pairs `"emma"` with `"mma"`, giving `('e','m'), ('m','m'), ('m','a')`.

```python
for ch1, ch2 in zip(w, w[1:]):       # ch1, ch2 unpacks each tuple
    print(ch1, ch2)

list(zip("abc", [1, 2, 3]))          # [('a', 1), ('b', 2), ('c', 3)]
list(zip("emma", "mma"))             # [('e','m'), ('m','m'), ('m','a')]  <- stops at the shorter one
```

`zip` returns a lazy iterator, so wrap it in `list(...)` to see its contents.

## `dict.get(key, default)` (counting pattern)
**What it is:** A dictionary lookup that returns a fallback value instead of raising `KeyError` when the key is missing.
**Why we're using it:** To count how often each bigram appears without first checking whether we've seen it before.
**How it works here:** In `lectures/2_makemore_bigram.ipynb`, `b` maps a bigram tuple like `('e', 'm')` to its count; `b.get(bigram, 0)` reads the current count (0 if it's new), and we store that plus one back into `b[bigram]`.

```python
b = {}
b[bigram] = b.get(bigram, 0) + 1

b.get(('e', 'm'), 0)    # 0 if never seen, otherwise the current count
b[('e', 'm')]           # KeyError if never seen
```

Tuples work as dictionary keys because they are immutable; a list like `['e', 'm']` would not.

## `torch.Generator` (seeded randomness is stateful)
**What it is:** An object that holds the state of a random number stream; `manual_seed(n)` sets where the stream starts.
**Why we're using it:** To make sampling reproducible, so `torch.multinomial` gives the same draws on every run.
**How it works here:** In `lectures/2_makemore_bigram.ipynb`, `g` is passed to `torch.multinomial(..., generator=g)`. Every draw moves `g` forward in its stream, so the seed only guarantees the same results if `g` is re-created (re-seeded) right before the draws. Re-running a cell that uses an old `g` continues from wherever the stream was left.

```python
g = torch.Generator().manual_seed(2147483647)   # start of the stream
torch.rand(3, generator=g)                      # draw A
torch.rand(3, generator=g)                      # draw B, different from A

g = torch.Generator().manual_seed(2147483647)   # re-seed
torch.rand(3, generator=g)                      # draw A again
```

Rule of thumb in notebooks: seed the generator in the same cell as the sampling you want to be repeatable.

## Maximum likelihood (and negative log likelihood loss)
**What it is:** A way of choosing model parameters: pick the ones that give the highest probability to the data you actually observed.
**Why we're using it:** It turns "how good is this bigram model?" into a single number we can optimize.
**How it works here:** In `lectures/2_makemore_bigram.ipynb`, the model assigns a probability to every bigram in `names.txt` (e.g. `P[ix1, ix2]`). The likelihood is the product of all those probabilities; training changes the parameters to make that product as large as possible.

The data is fixed; the parameters are what change. The model is never asked to "output" the dataset, it is scored on how much probability it puts on the next character that actually came next.

```python
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        prob = P[stoi[ch1], stoi[ch2]]      # probability the model gave to the true next char
        log_likelihood += torch.log(prob)   # log turns the product into a sum
        n += 1
nll = -log_likelihood / n                   # average negative log likelihood = the loss
```

Likelihood vs. maximum likelihood: the *likelihood* is the number (the product of probabilities). *Maximum likelihood* is the procedure of picking the parameters that make that number biggest. Here the parameters are the entries of `P`, and counting bigrams then normalizing each row is exactly the setting that maximizes it.

Why higher is better: the names in the dataset are things that really happened. A model that understands names should not be "surprised" by them, so it should have given high probability to each next character that actually came. "High" is relative: compare against the uniform baseline below, not against 1.

Chain of equivalent goals:
- maximize likelihood (product of probabilities, a tiny number)
- maximize log likelihood (sum of logs; same best parameters because `log` is monotonic)
- minimize negative log likelihood (flip the sign so lower is better, like a loss)
- minimize average NLL (divide by `n` so the number doesn't depend on dataset size)

Reading the number: `prob = 1` gives `log = 0` (perfect), `prob` near 0 gives a large negative log (very bad). A model that guesses uniformly over 27 characters scores `-log(1/27) ≈ 3.3`; the counting bigram model scores about 2.45.
