import Mathlib.RingTheory.TensorProduct.MvPolynomial

namespace CMPLean

/-- Before quotienting by sample ideals, tensoring two sample polynomial rings
over their one shared parameter/coefficient base `A` is algebraically equivalent
to a polynomial ring in the first sample variables with coefficients in the
second sample polynomial ring.  This already certifies the crucial shared-base
structure: there is one copy of `A`, not separate parameter bases per sample. -/
noncomputable def ambientPolynomialTensorEquiv
    (A : Type*) [CommSemiring A]
    (S₁ S₂ : Type*) [DecidableEq S₁] :
    TensorProduct A (MvPolynomial S₁ A) (MvPolynomial S₂ A) ≃ₐ[A]
      MvPolynomial S₁ (MvPolynomial S₂ A) :=
  MvPolynomial.scalarRTensorAlgEquiv

end CMPLean
