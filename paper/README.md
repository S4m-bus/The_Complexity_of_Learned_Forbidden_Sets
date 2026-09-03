# CMP / Exact-ReLU feasibility manuscript

This directory contains draft v0.1 of:

> Cotangent Message Passing for Exact ReLU Feasibility and the Existential
> Theory of the Reals

Build from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The rendered manuscript is `main.pdf`.  The proof-status and dependency ledger
is `PROOF_DEPENDENCY.md`.

## Current mathematical status

The structural ReLU, shared-base tensor, Kähler, block-Jacobian, and linear CMP
results are presented as unconditional.  The existing first-order message is
proved insufficient for exact feasibility by a one-ReLU counterexample already
formalized in `CMPLean/MessageCompletenessAudit.lean` and
`CMPLean/ZeroOrderAdversarial.lean`.

Accordingly, `P = ExistsR` and `P = NP` appear only as conditional consequences
of two explicit missing certificates: a polynomial-time exact feasibility
decider and a model-matching ETR reduction.  See `PROOF_DEPENDENCY.md` and
Sections 10--14.
