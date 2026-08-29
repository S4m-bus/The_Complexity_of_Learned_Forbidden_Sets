import Mathlib

namespace CMPLean

def ExactReLU (z a : ℝ) : Prop := a = max 0 z

theorem exactReLU_iff_complementarity (z a : ℝ) :
    ExactReLU z a ↔ 0 ≤ a ∧ 0 ≤ a - z ∧ a * (a - z) = 0 := by
  constructor
  · intro h
    unfold ExactReLU at h
    by_cases hz : z ≤ 0
    · rw [max_eq_left hz] at h
      subst a
      constructor
      · norm_num
      constructor
      · linarith
      · ring
    · have hz' : 0 < z := lt_of_not_ge hz
      rw [max_eq_right (le_of_lt hz')] at h
      subst a
      constructor
      · linarith
      constructor
      · norm_num
      · ring
  · rintro ⟨ha, haz, hprod⟩
    unfold ExactReLU
    rcases mul_eq_zero.mp hprod with hzero | hdiff
    · have ha0 : a = 0 := hzero
      subst a
      have hz : z ≤ 0 := by linarith
      rw [max_eq_left hz]
    · have haz0 : a = z := by linarith [hdiff]
      subst a
      have hz : 0 ≤ z := ha
      rw [max_eq_right hz]

theorem exactReLU_zero : ExactReLU 0 0 := by
  simp [ExactReLU]

theorem exactReLU_active {z a : ℝ} (h : ExactReLU z a) (hz : 0 < z) : a = z := by
  unfold ExactReLU at h
  simpa [max_eq_right (le_of_lt hz)] using h

theorem exactReLU_inactive {z a : ℝ} (h : ExactReLU z a) (hz : z < 0) : a = 0 := by
  unfold ExactReLU at h
  simpa [max_eq_left (le_of_lt hz)] using h

theorem exactReLU_boundary {a : ℝ} (h : ExactReLU 0 a) : a = 0 := by
  simpa [ExactReLU] using h

theorem exactReLU_iff_polynomialized (z a : ℝ) :
    ExactReLU z a ↔ ∃ u v : ℝ,
      a = u^2 ∧ a - z = v^2 ∧ a * (a - z) = 0 := by
  rw [exactReLU_iff_complementarity]
  constructor
  · rintro ⟨ha, haz, hprod⟩
    obtain ⟨u, hu⟩ := Real.exists_sq_eq_iff.mpr ha
    obtain ⟨v, hv⟩ := Real.exists_sq_eq_iff.mpr haz
    exact ⟨u, v, hu.symm, hv.symm, hprod⟩
  · rintro ⟨u, v, hua, hva, hprod⟩
    constructor
    · rw [hua]
      positivity
    constructor
    · rw [hva]
      positivity
    · exact hprod

end CMPLean
