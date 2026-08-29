import Mathlib

namespace CMPLean

/-- A decision language over an encoded input type. -/
abbrev Language (Input : Type*) := Input → Prop

/-- Exact Boolean correctness of a decider for a language. -/
def Decides {Input : Type*} (d : Input → Bool) (L : Language Input) : Prop :=
  ∀ x, d x = true ↔ L x

/-- Abstract polynomial-time predicate for Boolean algorithms.  The concrete
Turing-machine/bit-cost realization must instantiate this predicate. -/
def PClass {Input : Type*} (PolyTime : (Input → Bool) → Prop) : Set (Language Input) :=
  {L | ∃ d : Input → Bool, PolyTime d ∧ Decides d L}

/-- The exact certificate required to turn a CMP Boolean procedure into a
membership proof for the NNTP decision language.  In particular, correctness
and polynomial time are separate obligations. -/
structure CMPDecisionCertificate {Input : Type*}
    (PolyTime : (Input → Bool) → Prop) (Feasible : Language Input) where
  cmpDecide : Input → Bool
  correct : Decides cmpDecide Feasible
  polytime : PolyTime cmpDecide

/-- A certified CMP decider puts the exact NNTP feasibility language in P. -/
theorem exactNNTP_in_P_of_cmpCertificate
    {Input : Type*} {PolyTime : (Input → Bool) → Prop}
    {Feasible : Language Input}
    (C : CMPDecisionCertificate PolyTime Feasible) :
    Feasible ∈ PClass PolyTime := by
  exact ⟨C.cmpDecide, C.polytime, C.correct⟩

/-- Abstract polynomial reduction relation between decision languages. -/
abbrev ReductionRelation (Input : Type*) := Language Input → Language Input → Prop

/-- Closure of P under the selected polynomial-reduction relation. -/
def PClosedUnderReductions {Input : Type*}
    (PolyTime : (Input → Bool) → Prop)
    (Reduces : ReductionRelation Input) : Prop :=
  ∀ {L K : Language Input}, Reduces L K → K ∈ PClass PolyTime → L ∈ PClass PolyTime

/-- Exact reduction/completeness certificate needed to derive ExistsR ⊆ P from
an exact NNTP language already known to be in P. -/
structure ETRBridgeCertificate {Input : Type*}
    (PolyTime : (Input → Bool) → Prop)
    (Reduces : ReductionRelation Input)
    (ExistsR : Set (Language Input))
    (ExactNNTP : Language Input) where
  pClosed : PClosedUnderReductions PolyTime Reduces
  existsR_reduces_to_exactNNTP :
    ∀ L : Language Input, L ∈ ExistsR → Reduces L ExactNNTP

/-- Once ExactNNTP ∈ P and every ExistsR language polynomially reduces to it,
ExistsR is a subset of P. -/
theorem existsR_subset_P_of_cmp_and_bridge
    {Input : Type*} {PolyTime : (Input → Bool) → Prop}
    {Reduces : ReductionRelation Input}
    {ExistsR : Set (Language Input)}
    {ExactNNTP : Language Input}
    (hExactP : ExactNNTP ∈ PClass PolyTime)
    (B : ETRBridgeCertificate PolyTime Reduces ExistsR ExactNNTP) :
    ExistsR ⊆ PClass PolyTime := by
  intro L hL
  exact B.pClosed (B.existsR_reduces_to_exactNNTP L hL) hExactP

/-- The final class-collapse theorem.  This is the exact logical step used by
the P = ExistsR claim: CMP correctness + CMP polynomial time + the ETR bridge
supply ExistsR ⊆ P, while the standard containment P ⊆ NP ⊆ ExistsR supplies
the reverse inclusion. -/
theorem P_eq_ExistsR
    {Input : Type*} {PolyTime : (Input → Bool) → Prop}
    {Reduces : ReductionRelation Input}
    {NP ExistsR : Set (Language Input)}
    {ExactNNTP : Language Input}
    (C : CMPDecisionCertificate PolyTime ExactNNTP)
    (B : ETRBridgeCertificate PolyTime Reduces ExistsR ExactNNTP)
    (hPsubNP : PClass PolyTime ⊆ NP)
    (hNPsubExistsR : NP ⊆ ExistsR) :
    PClass PolyTime = ExistsR := by
  have hExactP : ExactNNTP ∈ PClass PolyTime :=
    exactNNTP_in_P_of_cmpCertificate C
  have hExistsRsubP : ExistsR ⊆ PClass PolyTime :=
    existsR_subset_P_of_cmp_and_bridge hExactP B
  apply Set.Subset.antisymm
  · intro L hLP
    exact hNPsubExistsR (hPsubNP hLP)
  · exact hExistsRsubP

/-- With the standard sandwich P ⊆ NP ⊆ ExistsR, the same certificates also
force P = NP. -/
theorem P_eq_NP
    {Input : Type*} {PolyTime : (Input → Bool) → Prop}
    {Reduces : ReductionRelation Input}
    {NP ExistsR : Set (Language Input)}
    {ExactNNTP : Language Input}
    (C : CMPDecisionCertificate PolyTime ExactNNTP)
    (B : ETRBridgeCertificate PolyTime Reduces ExistsR ExactNNTP)
    (hPsubNP : PClass PolyTime ⊆ NP)
    (hNPsubExistsR : NP ⊆ ExistsR) :
    PClass PolyTime = NP := by
  have hPE : PClass PolyTime = ExistsR :=
    P_eq_ExistsR C B hPsubNP hNPsubExistsR
  apply Set.Subset.antisymm hPsubNP
  intro L hLNP
  have hLE : L ∈ ExistsR := hNPsubExistsR hLNP
  simpa [hPE] using hLE

/-- Combined collapse statement. -/
theorem P_eq_NP_eq_ExistsR
    {Input : Type*} {PolyTime : (Input → Bool) → Prop}
    {Reduces : ReductionRelation Input}
    {NP ExistsR : Set (Language Input)}
    {ExactNNTP : Language Input}
    (C : CMPDecisionCertificate PolyTime ExactNNTP)
    (B : ETRBridgeCertificate PolyTime Reduces ExistsR ExactNNTP)
    (hPsubNP : PClass PolyTime ⊆ NP)
    (hNPsubExistsR : NP ⊆ ExistsR) :
    PClass PolyTime = NP ∧ NP = ExistsR := by
  have hPNP := P_eq_NP C B hPsubNP hNPsubExistsR
  have hPE := P_eq_ExistsR C B hPsubNP hNPsubExistsR
  refine ⟨hPNP, ?_⟩
  exact hPNP.symm.trans hPE

end CMPLean
