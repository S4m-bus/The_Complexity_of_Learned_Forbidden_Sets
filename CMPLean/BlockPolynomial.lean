import Mathlib.Algebra.MvPolynomial.PDeriv

namespace CMPLean

/-- Variable type for an NNTP polynomial: global parameter variables or
sample-tagged local variables. -/
abbrev NNVar (Param Sample LocalState : Type*) := Param ⊕ (Sample × LocalState)

/-- A polynomial constraint belongs to sample `i` when no local variable from a
different sample occurs in it. Global parameter variables remain unrestricted. -/
def IsSampleLocalPolynomial
    {R Param Sample LocalState : Type*}
    [CommSemiring R]
    (i : Sample)
    (f : MvPolynomial (NNVar Param Sample LocalState) R) : Prop :=
  ∀ (j : Sample) (s : LocalState), j ≠ i →
    (Sum.inr (j, s) : NNVar Param Sample LocalState) ∉ f.vars

/-- Exact block-angular zero property: differentiating a sample-i constraint
with respect to a local variable of a different sample gives zero. -/
theorem cross_sample_pderiv_zero
    {R Param Sample LocalState : Type*}
    [CommSemiring R]
    {i j : Sample} {s : LocalState}
    {f : MvPolynomial (NNVar Param Sample LocalState) R}
    (hlocal : IsSampleLocalPolynomial i f)
    (hji : j ≠ i) :
    MvPolynomial.pderiv (Sum.inr (j, s) : NNVar Param Sample LocalState) f = 0 := by
  exact MvPolynomial.pderiv_eq_zero_of_not_mem_vars (hlocal j s hji)

end CMPLean
