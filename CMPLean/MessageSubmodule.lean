import Mathlib

namespace CMPLean

variable {𝕜 LA XA XS : Type*}
variable [Field 𝕜]
variable [AddCommGroup LA] [Module 𝕜 LA]
variable [AddCommGroup XA] [Module 𝕜 XA]
variable [AddCommGroup XS] [Module 𝕜 XS]

/-- Basis-independent CMP message submodule: image on the separator of the
kernel of the internal block. -/
def IntrinsicMessageSubmodule
    (JA : LA →ₗ[𝕜] XA) (JAS : LA →ₗ[𝕜] XS) : Submodule 𝕜 XS :=
  (LinearMap.ker JA).map JAS

/-- Membership in the abstract CMP message has exactly the intended
left-nullspace/existential characterization. -/
theorem mem_intrinsicMessageSubmodule_iff
    (JA : LA →ₗ[𝕜] XA) (JAS : LA →ₗ[𝕜] XS) (s : XS) :
    s ∈ IntrinsicMessageSubmodule JA JAS ↔
      ∃ l : LA, JA l = 0 ∧ JAS l = s := by
  constructor
  · intro hs
    rcases hs with ⟨l, hl, rfl⟩
    exact ⟨l, hl, rfl⟩
  · rintro ⟨l, hl, rfl⟩
    exact ⟨l, hl, rfl⟩

/-- A CMP message can never have dimension larger than its separator ambient
space. This is independent of rank changes in the internal block. -/
theorem intrinsicMessage_finrank_le
    [FiniteDimensional 𝕜 XS]
    (JA : LA →ₗ[𝕜] XA) (JAS : LA →ₗ[𝕜] XS) :
    Module.finrank 𝕜 (IntrinsicMessageSubmodule JA JAS) ≤
      Module.finrank 𝕜 XS := by
  exact Submodule.finrank_le (IntrinsicMessageSubmodule JA JAS)

end CMPLean
