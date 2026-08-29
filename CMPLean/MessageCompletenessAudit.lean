import CMPLean.ReLUDecisionTests

namespace CMPLean

/-- For the one-ReLU exact decision instance at a positive target `y`, evaluated
at its satisfying point θ=z=a=y, the Jacobian rows for the equations
  z-θ = 0,
  a(a-z) = 0,
  a-y = 0
have separator/internal blocks whose left-nullspace elimination conditions are
  l₁ - y*l₂ = 0,
  y*l₂ + l₃ = 0,
and whose surviving parameter differential coefficient is `-l₁`.
This set is exactly the scalar instance of the CMP message `P_A^⊥ J_AS`. -/
def ScalarCMPMessage (y : ℝ) : Set ℝ :=
  {s | ∃ l₁ l₂ l₃ : ℝ,
      l₁ - y * l₂ = 0 ∧
      y * l₂ + l₃ = 0 ∧
      -l₁ = s}

/-- At every nonzero positive target, the scalar cotangent message is the full
one-dimensional separator space: it constrains differential direction but
contains no zeroth-order target offset. -/
theorem scalarCMPMessage_eq_univ {y : ℝ} (hy : y ≠ 0) :
    ScalarCMPMessage y = Set.univ := by
  ext s
  constructor
  · intro hs
    trivial
  · intro hs
    refine ⟨-s, -s / y, s, ?_, ?_, by ring⟩
    · field_simp [hy]
      ring
    · field_simp [hy]
      ring

/-- Hence the exact CMP cotangent message is identical for targets 1 and 2. -/
theorem scalarCMPMessage_one_eq_two :
    ScalarCMPMessage 1 = ScalarCMPMessage 2 := by
  rw [scalarCMPMessage_eq_univ (by norm_num : (1 : ℝ) ≠ 0)]
  rw [scalarCMPMessage_eq_univ (by norm_num : (2 : ℝ) ≠ 0)]

/-- But the exact ReLU NNTP parameter projections for targets 1 and 2 differ at
θ=1: target 1 is feasible there and target 2 is not. -/
theorem exact_projection_one_diff_two :
    OneReLUSample 1 1 ∧ ¬ OneReLUSample 2 1 := by
  constructor
  · exact (oneReLUSample_pos_iff (by norm_num : (0 : ℝ) < 1)).2 rfl
  · rw [oneReLUSample_pos_iff (by norm_num : (0 : ℝ) < 2)]
    norm_num

/-- No decoder whose *only* semantic input is the cotangent message subspace can
recover exact feasibility for all positive one-ReLU target instances. This is an
in-scope exact ReLU NNTP decision statement, not a generic-variety example. -/
theorem no_exact_decoder_from_scalarCMPMessage_alone :
    ¬ ∃ decode : Set ℝ → ℝ → Prop,
      ∀ (y θ : ℝ), 0 < y →
        (decode (ScalarCMPMessage y) θ ↔ OneReLUSample y θ) := by
  rintro ⟨decode, hdecode⟩
  have h1 := hdecode 1 1 (by norm_num : (0 : ℝ) < 1)
  have h2 := hdecode 2 1 (by norm_num : (0 : ℝ) < 2)
  have hmsg : ScalarCMPMessage 1 = ScalarCMPMessage 2 := scalarCMPMessage_one_eq_two
  have hfeas1 : OneReLUSample 1 1 := exact_projection_one_diff_two.1
  have hnfeas2 : ¬ OneReLUSample 2 1 := exact_projection_one_diff_two.2
  have hdec1 : decode (ScalarCMPMessage 1) 1 := h1.mpr hfeas1
  have hdec2 : decode (ScalarCMPMessage 2) 1 := by
    simpa [hmsg] using hdec1
  exact hnfeas2 (h2.mp hdec2)

end CMPLean
