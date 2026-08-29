import CMPLean.DecisionProblem

namespace CMPLean

namespace ExactNNTPDecision

variable {Sample Param LocalState : Type*}

/-- Exact local feasibility for one sample at a fixed shared parameter assignment.
This is the zeroth-order/existential object that any exact separator message must
represent in order to decide the NNTP problem. -/
def LocalFeasible
    (I : ExactNNTPDecision Sample Param LocalState)
    (θ : Param → ℝ) (i : Sample) : Prop :=
  ∃ s : LocalState → ℝ, I.localConstraint θ i s

/-- Exact parameter projection of the whole dataset: a shared parameter assignment
survives iff every sample has some local-state witness at that same assignment. -/
def ParameterProjection
    (I : ExactNNTPDecision Sample Param LocalState)
    (θ : Param → ℝ) : Prop :=
  ∀ i : Sample, I.LocalFeasible θ i

/-- The original NNTP YES/NO semantics is exactly existential feasibility of the
shared parameter projection.  No smoothness, genericity, or optimization notion
appears here. -/
theorem feasible_iff_exists_parameterProjection
    (I : ExactNNTPDecision Sample Param LocalState) :
    I.Feasible ↔ ∃ θ : Param → ℝ, I.ParameterProjection θ := by
  constructor
  · rintro ⟨θ, state, hstate⟩
    refine ⟨θ, ?_⟩
    intro i
    exact ⟨state i, hstate i⟩
  · rintro ⟨θ, hθ⟩
    classical
    have hw : ∀ i : Sample, ∃ s : LocalState → ℝ, I.localConstraint θ i s := hθ
    choose state hstate using hw
    exact ⟨θ, state, hstate⟩

/-- Dataset conjunction is separator-local at the semantic level: if two exact
NNTP families use the same shared parameters but otherwise independent local
witnesses, the exact surviving parameter predicate is the conjunction of their
parameter projections. -/
theorem parameterProjection_and
    (I₁ I₂ : ExactNNTPDecision Sample Param LocalState)
    (θ : Param → ℝ) :
    (∀ i, I₁.LocalFeasible θ i ∧ I₂.LocalFeasible θ i) ↔
      I₁.ParameterProjection θ ∧ I₂.ParameterProjection θ := by
  constructor
  · intro h
    constructor
    · intro i
      exact (h i).1
    · intro i
      exact (h i).2
  · rintro ⟨h₁, h₂⟩ i
    exact ⟨h₁ i, h₂ i⟩

end ExactNNTPDecision

end CMPLean
