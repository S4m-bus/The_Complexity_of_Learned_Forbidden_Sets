import CMPLean.ReLUDecisionTests
import Mathlib.Algebra.MvPolynomial.PDeriv

namespace CMPLean

/-- Three symbolic variables for one scalar ReLU sample: shared parameter θ,
preactivation z, and activation a. -/
abbrev ToyVar := Fin 3

/-- Exact polynomial equations for the one-neuron sample with target `y`.
Index 0 = θ, 1 = z, 2 = a.  The ReLU inequalities are handled semantically by
`ExactReLU`; this polynomial family isolates the symbolic Jacobian information
used by cotangent/Jacobian messages. -/
def toyAffine (R : Type*) [CommRing R] : MvPolynomial ToyVar R :=
  MvPolynomial.X 1 - MvPolynomial.X 0

def toyTarget (y : ℝ) : MvPolynomial ToyVar ℝ :=
  MvPolynomial.X 2 - MvPolynomial.C y

def toyReLUComplementarity : MvPolynomial ToyVar ℝ :=
  MvPolynomial.X 2 * (MvPolynomial.X 2 - MvPolynomial.X 1)

/-- Changing the target constant changes the exact feasibility problem but does
not change any formal partial derivative of the target equation. -/
theorem toyTarget_pderiv_target_independent
    (y₁ y₂ : ℝ) (v : ToyVar) :
    MvPolynomial.pderiv v (toyTarget y₁) =
      MvPolynomial.pderiv v (toyTarget y₂) := by
  classical
  simp [toyTarget]

/-- Therefore the entire symbolic Jacobian row family of the one-sample
polynomial equations is target-independent: the affine and ReLU rows are
literally the same, and the target row differs only by a constant whose
derivative vanishes. -/
theorem toy_symbolic_jacobian_target_independent
    (y₁ y₂ : ℝ) (v : ToyVar) :
    (MvPolynomial.pderiv v (toyAffine ℝ),
      MvPolynomial.pderiv v (toyTarget y₁),
      MvPolynomial.pderiv v toyReLUComplementarity) =
    (MvPolynomial.pderiv v (toyAffine ℝ),
      MvPolynomial.pderiv v (toyTarget y₂),
      MvPolynomial.pderiv v toyReLUComplementarity) := by
  rw [toyTarget_pderiv_target_independent y₁ y₂ v]

/-- An in-scope exact-ReLU decision YES instance with two samples sharing θ. -/
theorem exact_relu_yes_11 : TwoReLUSamples 1 1 := by
  rw [twoReLUSamples_pos_iff (by norm_num) (by norm_num)]

/-- An in-scope exact-ReLU decision NO instance with the same architecture and
same symbolic Jacobian pattern, but incompatible target constants. -/
theorem exact_relu_no_12 : ¬ TwoReLUSamples 1 2 := by
  rw [twoReLUSamples_pos_iff (by norm_num) (by norm_num)]
  norm_num

/-- First-order symbolic derivative data alone cannot determine the exact
YES/NO answer for this fixed ReLU architecture: target constants are invisible
to all formal derivatives, while feasibility changes from YES to NO. -/
theorem first_order_data_not_complete_for_exact_relu_decision :
    (∀ v : ToyVar,
      MvPolynomial.pderiv v (toyTarget 1) =
        MvPolynomial.pderiv v (toyTarget 2)) ∧
    TwoReLUSamples 1 1 ∧
    ¬ TwoReLUSamples 1 2 := by
  refine ⟨?_, exact_relu_yes_11, exact_relu_no_12⟩
  intro v
  exact toyTarget_pderiv_target_independent 1 2 v

end CMPLean
