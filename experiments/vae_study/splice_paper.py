"""Splice the rewritten Sections VII-XI (paper_block.tex) and the new abstract
ending into paper/main.tex."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "..", "paper", "main.tex")
t = open(P, encoding="utf-8").read()
block = open(os.path.join(HERE, "paper_block.tex"), encoding="utf-8").read()

fixes = [
    ("with 13--15\\,\\% in-phase bandwidth, a lowest reflection above $-0.6$\\,dB",
     "with 12--15\\,\\% in-phase bandwidth, a lowest reflection above $-0.6$\\,dB"),
    ("Its archive holds 6--13 designs predicted", "Its archive holds 5--13 designs predicted"),
    ("Their simulated bandwidths are therefore 1--3 percentage points narrower than predicted;",
     "Their simulated bandwidths are therefore up to 2.7 percentage points narrower than predicted;"),
    (" Without it, the swarm in the central latent box found layouts whose predicted bandwidth was up to 8 points too wide; the disagreement of independently trained networks marks such layouts at no extra simulation cost.",
     " The disagreement of independently trained networks marks unreliable predictions at no extra simulation cost."),
]
for a, b in fixes:
    assert a in block, a[:50]
    block = block.replace(a, b, 1)

start = t.index("\\subsection{MOIPSO in the Latent Space}")
end = t.index("%=====================================================================\n\\begin{thebibliography}")
t = t[:start] + block + t[end:]

a = t[t.index("The paper shows why such a VAE fails"):t.index("\\end{abstract}")]
b = ("The paper shows why such a VAE fails for random layouts when its latent space is too small and how to choose its size; "
     "the centre of the latent space of a VAE sized in this way holds connected, printable layouts. A check by independently "
     "trained surrogates keeps the search where the predictions are reliable. Compared with the same swarm working directly "
     "on the pixels, the search in the VAE space reaches a good Pareto front 20--70 times faster and ends with a better one "
     "at every budget from \\num{30200} to \\num{602000} layouts; a search over \\num{301000} layouts takes 2\\,min, and its "
     "front has a hypervolume of 0.531, against 0.295 for the 111,000 CST-simulated layouts of the dataset. In CST, all six "
     "selected designs of the VAE search meet the full specification, an in-phase bandwidth of 13.2--14.1\\,\\% with less "
     "than 0.35\\,dB in-band loss and no or one fragile feature, which no dataset layout achieves, and the magnitude errors of "
     "all nine simulated designs are below the surrogate's test error.\n")
t = t.replace(a, b, 1)
open(P, "w", encoding="utf-8").write(t)
print("spliced")
