import Mathlib

namespace CMPLean

/-- Abstract shape of the exact ReLU NNTP *decision* problem: parameters are
shared globally, while local state is tagged by sample.  The actual neural
network equations will instantiate `localConstraint`; feasibility is defined
independently of CMP. -/
structure ExactNNTPDecision (Sample Param LocalState : Type*) where
  localConstraint : (Param → ℝ) → Sample → (LocalState → ℝ) → Prop

namespace ExactNNTPDecision

variable {Sample Param LocalState : Type*}

/-- YES-instance semantics.  This is an existential decision predicate, not an
optimizer and not a witness-returning search problem. -/
def Feasible (I : ExactNNTPDecision Sample Param LocalState) : Prop :=
  ∃ θ : Param → ℝ,
    ∃ state : Sample → LocalState → ℝ,
      ∀ i : Sample, I.localConstraint θ i (state i)

theorem feasible_iff (I : ExactNNTPDecision Sample Param LocalState) :
    I.Feasible ↔
      ∃ θ : Param → ℝ,
        ∃ state : Sample → LocalState → ℝ,
          ∀ i : Sample, I.localConstraint θ i (state i) := by
  rfl

end ExactNNTPDecision

end CMPLean
