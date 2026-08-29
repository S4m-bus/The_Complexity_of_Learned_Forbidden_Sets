import CMPLean.ReLU

namespace CMPLean

/-- A minimal exact ReLU NNTP decision constraint with one shared scalar
parameter θ and one sample-local preactivation/activation pair. -/
def OneReLUSample (target θ : ℝ) : Prop :=
  ∃ z a : ℝ, z = θ ∧ ExactReLU z a ∧ a = target

/-- For a strictly positive exact target, the unique shared parameter satisfying
this one-neuron instance is θ = target. -/
theorem oneReLUSample_pos_iff
    {target θ : ℝ} (ht : 0 < target) :
    OneReLUSample target θ ↔ θ = target := by
  constructor
  · rintro ⟨z, a, hz, hrelu, ha⟩
    subst z
    subst a
    unfold ExactReLU at hrelu
    have hθ : 0 < θ := by
      by_contra h
      have hθ0 : θ ≤ 0 := le_of_not_gt h
      rw [max_eq_left hθ0] at hrelu
      linarith
    simpa [max_eq_right (le_of_lt hθ)] using hrelu.symm
  · intro hθ
    subst θ
    refine ⟨target, target, rfl, ?_, rfl⟩
    unfold ExactReLU
    simp [max_eq_right (le_of_lt ht)]

/-- Two samples share the same trainable parameter.  This is a YES/NO
feasibility predicate, not an optimizer. -/
def TwoReLUSamples (y₁ y₂ : ℝ) : Prop :=
  ∃ θ : ℝ, OneReLUSample y₁ θ ∧ OneReLUSample y₂ θ

theorem twoReLUSamples_pos_iff
    {y₁ y₂ : ℝ} (h1 : 0 < y₁) (h2 : 0 < y₂) :
    TwoReLUSamples y₁ y₂ ↔ y₁ = y₂ := by
  constructor
  · rintro ⟨θ, hs1, hs2⟩
    have hθ1 : θ = y₁ := (oneReLUSample_pos_iff h1).mp hs1
    have hθ2 : θ = y₂ := (oneReLUSample_pos_iff h2).mp hs2
    linarith
  · intro hy
    subst y₂
    exact ⟨y₁, (oneReLUSample_pos_iff h1).mpr rfl,
      (oneReLUSample_pos_iff h1).mpr rfl⟩

example : TwoReLUSamples 1 1 := by
  rw [twoReLUSamples_pos_iff (by norm_num) (by norm_num)]

example : ¬ TwoReLUSamples 1 2 := by
  rw [twoReLUSamples_pos_iff (by norm_num) (by norm_num)]
  norm_num

/-- Boundary-valued ReLU remains a valid decision instance. -/
example : OneReLUSample 0 0 := by
  exact ⟨0, 0, rfl, exactReLU_zero, rfl⟩

end CMPLean
