import Mathlib.RingTheory.Kaehler.Basic

namespace CMPLean

open KaehlerDifferential

variable (R A B : Type*)
variable [CommRing R] [CommRing A] [CommRing B]
variable [Algebra R A] [Algebra A B] [Algebra R B]
variable [IsScalarTower R A B]

/-- The global Jacobi-Zariski/transitivity exact sequence used by CMP exists
without any smoothness, regularity, or non-singularity hypothesis. -/
theorem kahler_transitivity_exact :
    Function.Exact
      (KaehlerDifferential.mapBaseChange R A B)
      (KaehlerDifferential.map R A B B) := by
  exact KaehlerDifferential.exact_mapBaseChange_map R A B

/-- In particular, the relative differential map to Ω[B/A] is surjective.
This is a global algebra statement, not a smooth-locus statement. -/
theorem kahler_relative_map_surjective :
    Function.Surjective (KaehlerDifferential.map R A B B) := by
  exact KaehlerDifferential.map_surjective R A B

end CMPLean
