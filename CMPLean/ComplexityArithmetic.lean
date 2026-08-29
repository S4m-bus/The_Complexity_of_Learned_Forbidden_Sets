import Mathlib

namespace CMPLean

/-- The arithmetic core of the uniform complexity estimate: if every structural
parameter is bounded by total encoded input size `n`, then the CMP arithmetic
count `N * (P + K)^3` is bounded by `8 * n^4`. -/
theorem uniform_cubic_bound
    {N P K n : ℕ}
    (hN : N ≤ n) (hP : P ≤ n) (hK : K ≤ n) :
    N * (P + K) ^ 3 ≤ 8 * n ^ 4 := by
  calc
    N * (P + K) ^ 3 ≤ n * (2 * n) ^ 3 := by
      gcongr
      omega
    _ = 8 * n ^ 4 := by ring

end CMPLean
