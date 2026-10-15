"""Shorten the trust-check text of paper/main.tex to a plain statement of the method."""
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "main.tex")
t = open(P, encoding="utf-8").read()


def cut(a, b):
    global t
    assert a in t, a[:60]
    t = t.replace(a, b, 1)


# method: one short paragraph
s = t.index("\\emph{Trust check.}")
e = t.index("\\emph{Pixel space.}")
t = t[:s] + ("\\emph{Reliable predictions.} Every layout is also scored by two other RDN surrogates, RDN108000 without PGA "
             "and RDN27000 (Section~\\ref{sec:fwdresults}). Layouts on which their predictions of $r_{xx}$ differ from that "
             "of RDN108000 more than for a typical RDN test layout (mean squared difference above 0.01, its median on the "
             "test set) are not admitted to the archive.\n\n") + t[e:]
cut("the initial positions are uniform, and each velocity component is limited to 0.042. All other settings, including the trust check, are the same. The step sizes of both spaces were chosen by a short sweep at the standard budget.",
    "the initial positions are uniform, and each velocity component is limited to 0.042. All other settings are the same.")
cut("Trust check & disagreement $u\\leq0.01$ \\\\\n", "")
cut("\\item the surrogate and the two check networks predict $r_{xx}$ of all layouts in one batch;",
    "\\item the surrogate predicts $r_{xx}$ of all layouts in one batch;")
cut("\\item the three objectives and the trust check are computed for each layout;",
    "\\item the three objectives are computed for each layout;")
cut("in 10--21\\,s on one GPU, including decoding and the trust check;", "in 10--21\\,s on one GPU, including decoding;")
cut(" Search values are predicted by the surrogate, with the trust check.}", " Search values are predicted by the surrogate.}")

# discussion
s = t.index("\\emph{Accuracy where the optimizer searches.}")
e = t.index("\\emph{Design space.}")
t = t[:s] + ("\\emph{Accuracy where the optimizer searches.} The optimizer selects the best predictions among many "
             "thousands, so the selected designs are where an over-optimistic surrogate would show first. The magnitude "
             "errors of the selected designs are below the test error, and their in-phase band edges agree with CST within "
             "0.4\\,GHz. The remaining systematic difference is a slightly too wide predicted band for the stripe-like VAE "
             "layouts. Adding CST simulations of such layouts to the training data, or a margin on the bandwidth objective, "
             "would remove it.\n\n") + t[e:]

# abstract and conclusion
cut(" A check by independently trained surrogates keeps the search where the predictions are reliable.", "")
cut("Used in place of the solver, with a check by independently trained networks, the surrogate lets",
    "Used in place of the solver, the surrogate lets")
open(P, "w", encoding="utf-8").write(t)
print("trimmed")
