import Mathlib.RingTheory.TensorProduct.MvPolynomial

namespace CMPLean

/-- Before quotienting by sample ideals, two disjoint sample polynomial rings
combine as a tensor product over their single shared parameter/coefficient base.
This keeps one copy of the base ring rather than tensoring over ℝ independently. -/
noncomputable def ambientPolynomialTensorEquiv
    (A : Type*) [CommSemiring A]
    (S₁ S₂ : Type*) :
    TensorProduct A (MvPolynomial S₁ A) (MvPolynomial S₂ A) ≃ₐ[A]
      MvPolynomial (S₁ ⊕ S₂) A :=
  MvPolynomial.tensorEquivSum A S₁ S₂ A

end CMPLean
