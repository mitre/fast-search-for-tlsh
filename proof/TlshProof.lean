import Mathlib.Data.List.Basic
import Mathlib.Data.Vector.Basic
import Mathlib.Tactic.IntervalCases
import Std.Tactic.BVDecide


-- Options
set_option maxHeartbeats 100000000 -- For portability


-- The bounds we seek to prove.
def MAX_HEADER_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION : Nat := 46
def MAX_BODY_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION   : Nat := 384
def MAX_TLSH_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION   : Nat := MAX_HEADER_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION + MAX_BODY_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION


-- An implementation of TLSH in Lean4

def HeaderHexDigit := { n : BitVec 32 // n < 16#32 }
def HeaderHexDigitPair := { n: BitVec 32 // n < 256#32 }
def HexDigit := { n : Nat // n < 16 }
def HexDigitPair := { n : Nat // n < 256 }


structure HeaderStruct where
  c : HeaderHexDigit
  l : HeaderHexDigitPair
  q₁ : HeaderHexDigit
  q₂ : HeaderHexDigit


def Header := HeaderStruct
def Body   := Mathlib.Vector HexDigit 64
def Digest := Header × Body


def toHeaderHexDigit (n : Nat) (h : n < 16#32) : HeaderHexDigit := ⟨n, h⟩
def toHeaderHexDigitPair (n : Nat) (h : n < 256#32) : HeaderHexDigitPair := ⟨n, h⟩
def toHexDigit (n : Nat) (h : n < 16) : HexDigit := ⟨n, h⟩
def toHexDigitPair (n : Nat) (h : n < 256) : HexDigitPair := ⟨n, h⟩


def qModDiff (x y : HeaderHexDigit) : BitVec 32 :=
  let d := if x.val > y.val then x.val - y.val else y.val - x.val
  if 16#32 - d < d then 16#32 - d else d


def lModDiff (x y : HeaderHexDigitPair) : BitVec 32 :=
  let d := if x.val > y.val then x.val - y.val else y.val - x.val
  if 256#32 - d < d then 256#32 - d else d


def cDiff (x y : HeaderHexDigit) : BitVec 32 :=
  if x.val = y.val then 0#32 else 1#32


def qDiff (x y : HeaderHexDigit) : BitVec 32 :=
  let d := qModDiff x y
  if d ≤ 1#32 then d else (d - 1#32) * 12#32


def lDiff (x y : HeaderHexDigitPair) : BitVec 32 :=
  let d := (lModDiff x y)
  if d ≤ 1#32 then d else d * 12#32


def getHeaderDistance (tX tY : Header) : BitVec 32 :=
  let Δl  := lDiff tX.l tY.l
  let Δq₁ := qDiff tX.q₁ tY.q₁
  let Δq₂ := qDiff tX.q₂ tY.q₂
  let Δc  := cDiff tX.c tY.c
  Δl + Δq₁ + Δq₂ + Δc


def getHeaderDistance' (H₁ H₂ : Header) : Nat := (getHeaderDistance H₁ H₂).toNat


def getBodyHexDigitDistance (x y : HexDigit) : Nat :=
  let x₁ := x.val / 4
  let x₂ := x.val % 4
  let y₁ := y.val / 4
  let y₂ := y.val % 4
  let d₁ := if x₁ > y₁ then x₁ - y₁ else y₁ - x₁
  let d₂ := if x₂ > y₂ then x₂ - y₂ else y₂ - x₂
  let Δ'₁ := if d₁ = 3 then 6 else d₁
  let Δ'₂ := if d₂ = 3 then 6 else d₂
  Δ'₁ + Δ'₂


def getBodyDistance (tX tY : Body) : Nat :=
  List.sum (List.zipWith getBodyHexDigitDistance tX.toList tY.toList)


def getTlshDistance (tX tY : Digest) : Nat :=
  let ((hX, bX), (hY, bY)) := (tX, tY)
  (getHeaderDistance hX hY).toNat + getBodyDistance bX bY


-- Proofs


-- Constructive examples used later (proof setup)
def example_headers : Header × Header × Header :=
  (
    {
      c  := toHeaderHexDigit     0    (by simp),
      l  := toHeaderHexDigitPair 255  (by simp),
      q₁ := toHeaderHexDigit     1    (by simp),
      q₂ := toHeaderHexDigit     1    (by simp),
    },
    {
      c  := toHeaderHexDigit     0    (by simp),
      l  := toHeaderHexDigitPair 0    (by simp),
      q₁ := toHeaderHexDigit     15   (by simp),
      q₂ := toHeaderHexDigit     15   (by simp),
    },
    {
      c  := toHeaderHexDigit     0    (by simp),
      l  := toHeaderHexDigitPair 1    (by simp),
      q₁ := toHeaderHexDigit     11   (by simp),
      q₂ := toHeaderHexDigit     11   (by simp),
    }
  )


def example_bodies : Body × Body × Body :=
  (
    ⟨List.replicate 64 (toHexDigit 0 (by simp)), by simp⟩,
    ⟨List.replicate 64 (toHexDigit 5 (by simp)), by simp⟩,
    ⟨List.replicate 64 (toHexDigit 15 (by simp)), by simp⟩
  )


-- HEADER DISTANCE PROOFS --
-- Where 46 = MAX_HEADER_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION
theorem max_header_violation' : ∀ (H₁ H₂ H₃ : Header),
  getHeaderDistance H₁ H₃ ≤ getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃ + 46#32 := by
  intros H₁ H₂ H₃

  let c₁  := H₁.c
  let l₁  := H₁.l
  let q₁₁ := H₁.q₁
  let q₂₁ := H₁.q₂

  let c₂  := H₂.c
  let l₂  := H₂.l
  let q₁₂ := H₂.q₁
  let q₂₂ := H₂.q₂

  let c₃  := H₃.c
  let l₃  := H₃.l
  let q₁₃ := H₃.q₁
  let q₂₃ := H₃.q₂

  -- Get inequalities for components l, q₁, q₂
  have h_Δq₁ : qDiff H₁.q₁ H₃.q₁ ≤ qDiff H₁.q₁ H₂.q₁ + qDiff H₂.q₁ H₃.q₁ + 12#32 := by
    have h_bounds : H₁.q₁.val < 16#32 ∧ H₂.q₁.val < 16#32 ∧ H₃.q₁.val < 16#32 := ⟨H₁.q₁.property, H₂.q₁.property, H₃.q₁.property⟩
    unfold qDiff qModDiff
    bv_decide (config := {acNf := true})

  have h_Δq₂ : qDiff H₁.q₂ H₃.q₂ ≤ qDiff H₁.q₂ H₂.q₂ + qDiff H₂.q₂ H₃.q₂ + 12#32 := by
    have h_bounds : H₁.q₂.val < 16#32 ∧ H₂.q₂.val < 16#32 ∧ H₃.q₂.val < 16#32 := ⟨H₁.q₂.property, H₂.q₂.property, H₃.q₂.property⟩
    unfold qDiff qModDiff
    bv_decide (config := {acNf := true})

  have h_Δl : lDiff H₁.l H₃.l ≤ lDiff H₁.l H₂.l + lDiff H₂.l H₃.l + 22#32 := by
    have h_bounds : H₁.l.val < 256#32 ∧ H₂.l.val < 256#32 ∧ H₃.l.val < 256#32 := ⟨H₁.l.property, H₂.l.property, H₃.l.property⟩
    unfold lDiff lModDiff
    bv_decide (config := {acNf := true})

  -- This is actually extremely easy to show by-hand, w/o bv_decide.
  -- The others are fairly simple to do algebraically, but this is a _very_
  -- straight-forward 8-case truth-table.
  have h_Δc' : cDiff H₁.c H₃.c ≤ cDiff H₁.c H₂.c + cDiff H₂.c H₃.c := by
    unfold cDiff
    rcases em (c₁ ≠ c₂) with (h₁₂ | h₁₂)
    rcases em (c₂ ≠ c₃) with (h₂₃ | h₂₃)
    rcases em (c₁ ≠ c₃) with (h₁₃ | h₁₃)
    all_goals
    bv_decide

  have h_all : getHeaderDistance H₁ H₃ ≤ getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃ + 46#32 := by
    unfold getHeaderDistance qDiff lDiff cDiff lModDiff qModDiff at *
    have h_c_bounds  : c₁.val  < 16#32  ∧ c₂.val  < 16#32  ∧ c₃.val  < 16#32  := ⟨H₁.c.property,  H₂.c.property, H₃.c.property⟩
    have h_l_bounds  : l₁.val  < 256#32 ∧ l₂.val  < 256#32 ∧ l₃.val  < 256#32 := ⟨H₁.l.property,  H₂.l.property, H₃.l.property⟩
    have h_q₁_bounds : q₁₁.val < 16#32  ∧ q₁₂.val < 16#32  ∧ q₁₃.val < 16#32  := ⟨H₁.q₁.property, H₂.q₁.property, H₃.q₁.property⟩
    have h_q₂_bounds : q₂₁.val < 16#32  ∧ q₂₂.val < 16#32  ∧ q₂₃.val < 16#32  := ⟨H₁.q₂.property, H₂.q₂.property, H₃.q₂.property⟩

    bv_decide (config := {acNf := true})

  exact h_all


theorem max_header_violation : ∀ (H₁ H₂ H₃ : Header),
  getHeaderDistance' H₁ H₃ ≤ getHeaderDistance' H₁ H₂ + getHeaderDistance' H₂ H₃ + 46 := by
  intros H₁ H₂ H₃

  let c₁  := H₁.c
  let l₁  := H₁.l
  let q₁₁ := H₁.q₁
  let q₂₁ := H₁.q₂

  let c₂  := H₂.c
  let l₂  := H₂.l
  let q₁₂ := H₂.q₁
  let q₂₂ := H₂.q₂

  let c₃  := H₃.c
  let l₃  := H₃.l
  let q₁₃ := H₃.q₁
  let q₂₃ := H₃.q₂

  have h_c_bounds  : c₁.val  < 16#32  ∧ c₂.val  < 16#32  ∧ c₃.val  < 16#32  := ⟨H₁.c.property,  H₂.c.property, H₃.c.property⟩
  have h_l_bounds  : l₁.val  < 256#32 ∧ l₂.val  < 256#32 ∧ l₃.val  < 256#32 := ⟨H₁.l.property,  H₂.l.property, H₃.l.property⟩
  have h_q₁_bounds : q₁₁.val < 16#32  ∧ q₁₂.val < 16#32  ∧ q₁₃.val < 16#32  := ⟨H₁.q₁.property, H₂.q₁.property, H₃.q₁.property⟩
  have h_q₂_bounds : q₂₁.val < 16#32  ∧ q₂₂.val < 16#32  ∧ q₂₃.val < 16#32  := ⟨H₁.q₂.property, H₂.q₂.property, H₃.q₂.property⟩

  have h_bv := max_header_violation' H₁ H₂ H₃
  have h_nat : (getHeaderDistance H₁ H₃).toNat ≤ (getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃ + 46#32).toNat := h_bv

  unfold getHeaderDistance' at *

  have h_bound : (getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃ + 46#32) ≤ 4294967295 := by
    unfold getHeaderDistance at *
    unfold qDiff lDiff cDiff lModDiff qModDiff at *
    bv_decide (config := {acNf := true})

  have h_bound' : (getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃) ≤ 4294967295 := by
    unfold getHeaderDistance at *
    unfold qDiff lDiff cDiff lModDiff qModDiff at *
    bv_decide (config := {acNf := true})

  have h_sum : (getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃ + 46#32).toNat =
              (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat + 46 := by
    -- Since l ≤ 256 * 12, q₁ < (16 - 1) * 12, q₂ ≤ (16 - 1) * 12, and c ≤ 1,
    -- We just sum (256 * 12) + (16 - 1) * 12 + (16 - 1) * 12 + 1 to get 3433.
    have h_bound₁ : getHeaderDistance H₁ H₂ ≤ 3433#32 := by
      unfold getHeaderDistance
      unfold qDiff lDiff cDiff lModDiff qModDiff at *
      bv_decide (config := {acNf := true})
    have h_bound₂ : getHeaderDistance H₂ H₃ ≤ 3433#32 := by
      unfold getHeaderDistance
      unfold qDiff lDiff cDiff lModDiff qModDiff at *
      bv_decide (config := {acNf := true})

    -- We derive 6866 and 6867 from x, y ≤ 3433 → x + y ≤ 6866 → x + y < 6867
    -- Where 3433 is derived in the prior comments.
    have h_sum' : getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃ < 6867#32 := by
      bv_decide (config := {acNf := true})

    have h_sum_bound : (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat < 2^32 := by
      have h₁ : (getHeaderDistance H₁ H₂).toNat ≤ 3433 := by
        rw [BitVec.le_def] at h_bound₁
        exact h_bound₁
      have h₂ : (getHeaderDistance H₂ H₃).toNat ≤ 3433 := by
        rw [BitVec.le_def] at h_bound₂
        exact h_bound₂
      have h₃ : (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat ≤ 6866 :=
        Nat.add_le_add h₁ h₂
      exact Nat.lt_of_le_of_lt h₃ (by norm_num : 6866 < 2^32)

    have h_sum_bound' : (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat + 46 < 2^32 := by
      have h₁ : (getHeaderDistance H₁ H₂).toNat ≤ 3433 := by
        rw [BitVec.le_def] at h_bound₁
        exact h_bound₁
      have h₂ : (getHeaderDistance H₂ H₃).toNat ≤ 3433 := by
        rw [BitVec.le_def] at h_bound₂
        exact h_bound₂
      have h₃ : (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat ≤ 6866 :=
        Nat.add_le_add h₁ h₂
      have h_rewrite : (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat + 46 < 2 ^ 32 ↔ (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat < 2 ^ 32 - 46 := by
        exact Iff.symm Nat.lt_sub_iff_add_lt
      rw [h_rewrite]
      conv_rhs => norm_num
      exact Nat.lt_of_le_of_lt h₃ (by norm_num : 6866 < 4294967250)

    have h₁ : (getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃).toNat =
              (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat := by
      simp [BitVec.toNat_add]
      have h_sum_tonat : (getHeaderDistance H₁ H₂).toNat + (getHeaderDistance H₂ H₃).toNat < 6867 := by
        rw [BitVec.lt_def, BitVec.toNat_add_of_lt h_sum_bound] at h_sum'
        exact h_sum'
      exact Nat.lt_trans h_sum_tonat (by norm_num : 6867 < 4294967296)

    rw [BitVec.toNat_add, h₁]
    conv_lhs =>
      lhs; rhs; rw [BitVec.toNat_ofNat]
      rhs; norm_num

    apply Nat.mod_eq_of_lt
    apply Nat.lt_succ_of_le
    apply Nat.le_pred_of_lt at h_sum_bound'
    norm_num at h_sum_bound'
    exact h_sum_bound'

  simp_all


-- Prove tightest lower bound on error is MAX_HEADER_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION
def header_violation_example : Nat :=
  let (H₁, H₂, H₃) := example_headers
  (getHeaderDistance H₁ H₃ - (getHeaderDistance H₁ H₂ + getHeaderDistance H₂ H₃)).toNat


example : header_violation_example = MAX_HEADER_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION := by rfl


-- BODY DISTANCE PROOFS --

-- Prove upper-bound on error is MAX_BODY_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION
-- Where 6 is the maximum degree to which a single digit's distance across hashes will violate the triangle inequality
theorem max_single_body_digit_violation : ∀ (x y z : HexDigit),
  getBodyHexDigitDistance x z ≤ getBodyHexDigitDistance x y + getBodyHexDigitDistance y z + 6 := by
  intros x y z

  have h_bounds : x.val < 16 ∧ y.val < 16 ∧ z.val < 16 := ⟨x.property, y.property, z.property⟩

  let x₁ := x.val / 4
  let x₂ := x.val % 4
  let y₁ := y.val / 4
  let y₂ := y.val % 4
  let z₁ := z.val / 4
  let z₂ := z.val % 4

  have h_x₁ : x₁ < 4 := Nat.div_lt_of_lt_mul h_bounds.1
  have h_x₂ : x₂ < 4 := Nat.mod_lt x.val (by norm_num)
  have h_y₁ : y₁ < 4 := Nat.div_lt_of_lt_mul h_bounds.2.1
  have h_y₂ : y₂ < 4 := Nat.mod_lt y.val (by norm_num)
  have h_z₁ : z₁ < 4 := Nat.div_lt_of_lt_mul h_bounds.2.2
  have h_z₂ : z₂ < 4 := Nat.mod_lt z.val (by norm_num)

  have h_high_bits_violation : ∀ (x₁ y₁ z₁ : Nat), x₁ < 4 → y₁ < 4 → z₁ < 4 →
    (if (if x₁ > z₁ then x₁ - z₁ else z₁ - x₁) = 3 then 6 -- lol
    else if x₁ > z₁ then x₁ - z₁ else z₁ - x₁) ≤
    (if (if x₁ > y₁ then x₁ - y₁ else y₁ - x₁) = 3 then 6
    else if x₁ > y₁ then x₁ - y₁ else y₁ - x₁) +
    (if (if y₁ > z₁ then y₁ - z₁ else z₁ - y₁) = 3 then 6
    else if y₁ > z₁ then y₁ - z₁ else z₁ - y₁) + 3 := by
    intros x₁ y₁ z₁ hx hy hz

    interval_cases x₁; all_goals
    interval_cases y₁; all_goals
    interval_cases z₁; all_goals
    simp

  have h_low_bits_violation : ∀ (x₂ y₂ z₂ : Nat), x₂ < 4 → y₂ < 4 → z₂ < 4 →
    (if (if x₂ > z₂ then x₂ - z₂ else z₂ - x₂) = 3 then 6
    else if x₂ > z₂ then x₂ - z₂ else z₂ - x₂) ≤
    (if (if x₂ > y₂ then x₂ - y₂ else y₂ - x₂) = 3 then 6
    else if x₂ > y₂ then x₂ - y₂ else y₂ - x₂) +
    (if (if y₂ > z₂ then y₂ - z₂ else z₂ - y₂) = 3 then 6
    else if y₂ > z₂ then y₂ - z₂ else z₂ - y₂) + 3 := by
    intros x₂ y₂ z₂ hx hy hz

    interval_cases x₂; all_goals
    interval_cases y₂; all_goals
    interval_cases z₂; all_goals
    simp

  unfold getBodyHexDigitDistance

  have h_high := h_high_bits_violation x₁ y₁ z₁ h_x₁ h_y₁ h_z₁
  have h_low := h_low_bits_violation x₂ y₂ z₂ h_x₂ h_y₂ h_z₂

  have h_total := calc
    (if (if x₁ > z₁ then x₁ - z₁ else z₁ - x₁) = 3 then 6 else if x₁ > z₁ then x₁ - z₁ else z₁ - x₁) +
    (if (if x₂ > z₂ then x₂ - z₂ else z₂ - x₂) = 3 then 6 else if x₂ > z₂ then x₂ - z₂ else z₂ - x₂)
    ≤
    ((if (if x₁ > y₁ then x₁ - y₁ else y₁ - x₁) = 3 then 6 else if x₁ > y₁ then x₁ - y₁ else y₁ - x₁) +
    (if (if y₁ > z₁ then y₁ - z₁ else z₁ - y₁) = 3 then 6 else if y₁ > z₁ then y₁ - z₁ else z₁ - y₁) + 3) +
    ((if (if x₂ > y₂ then x₂ - y₂ else y₂ - x₂) = 3 then 6 else if x₂ > y₂ then x₂ - y₂ else y₂ - x₂) +
    (if (if y₂ > z₂ then y₂ - z₂ else z₂ - y₂) = 3 then 6 else if y₂ > z₂ then y₂ - z₂ else z₂ - y₂) + 3) := by
      exact Nat.add_le_add h_high h_low
    _ = ((if (if x₁ > y₁ then x₁ - y₁ else y₁ - x₁) = 3 then 6 else if x₁ > y₁ then x₁ - y₁ else y₁ - x₁) +
        (if (if x₂ > y₂ then x₂ - y₂ else y₂ - x₂) = 3 then 6 else if x₂ > y₂ then x₂ - y₂ else y₂ - x₂)) +
        ((if (if y₁ > z₁ then y₁ - z₁ else z₁ - y₁) = 3 then 6 else if y₁ > z₁ then y₁ - z₁ else z₁ - y₁) +
        (if (if y₂ > z₂ then y₂ - z₂ else z₂ - y₂) = 3 then 6 else if y₂ > z₂ then y₂ - z₂ else z₂ - y₂)) +
        6 := by
      omega
    _ = getBodyHexDigitDistance x y + getBodyHexDigitDistance y z + 6 := by rfl

  exact h_total


theorem max_body_violation : ∀ (B₁ B₂ B₃ : Body),
  getBodyDistance B₁ B₃ ≤ getBodyDistance B₁ B₂ + getBodyDistance B₂ B₃ + MAX_BODY_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION := by
  intros B₁ B₂ B₃
  unfold getBodyDistance

  have h_body_size_k {k : Nat} :
    ∀ (l₁ l₂ l₃ : List HexDigit),
    l₁.length = k → l₂.length = k → l₃.length = k →
    List.sum (List.zipWith getBodyHexDigitDistance l₁ l₃) ≤
      List.sum (List.zipWith getBodyHexDigitDistance l₁ l₂) +
      List.sum (List.zipWith getBodyHexDigitDistance l₂ l₃) +
      6 * k := by
    induction k with
    | zero =>
      simp_all
    | succ k' ih =>
      intros l₁ l₂ l₃ h₁ h₂ h₃
      cases l₁ with
      | nil => contradiction
      | cons hd₁ tl₁ =>
        cases l₂ with
        | nil => contradiction
        | cons hd₂ tl₂ =>
          cases l₃ with
          | nil => contradiction
          | cons hd₃ tl₃ =>
            have tl_lengths : tl₁.length = k' ∧ tl₂.length = k' ∧ tl₃.length = k' := by
              simp_all only [List.length_cons, add_left_inj, and_self]
            calc
              List.sum (List.zipWith getBodyHexDigitDistance (hd₁::tl₁) (hd₃::tl₃))
              = getBodyHexDigitDistance hd₁ hd₃ +
                List.sum (List.zipWith getBodyHexDigitDistance tl₁ tl₃) := by
                simp [List.zipWith]
              _ ≤ (getBodyHexDigitDistance hd₁ hd₂ + getBodyHexDigitDistance hd₂ hd₃ + 6) +
                  List.sum (List.zipWith getBodyHexDigitDistance tl₁ tl₃) := by
                  have single_digit_bound := max_single_body_digit_violation hd₁ hd₂ hd₃
                  simp_all only [List.length_cons, add_le_add_iff_right]
              _ ≤ (getBodyHexDigitDistance hd₁ hd₂ + getBodyHexDigitDistance hd₂ hd₃ + 6) +
                  (List.sum (List.zipWith getBodyHexDigitDistance tl₁ tl₂) +
                  List.sum (List.zipWith getBodyHexDigitDistance tl₂ tl₃) + 6 * k') := by
                simp_all only [List.length_cons, add_le_add_iff_left]
              _ = List.sum (List.zipWith getBodyHexDigitDistance (hd₁::tl₁) (hd₂::tl₂)) +
                  List.sum (List.zipWith getBodyHexDigitDistance (hd₂::tl₂) (hd₃::tl₃)) +
                  6 * (k' + 1) := by
                  simp [List.zipWith_cons_cons]
                  omega

  have h := h_body_size_k B₁.toList B₂.toList B₃.toList (by exact rfl)
  simp_all only [Mathlib.Vector.toList_length, forall_const, Nat.reduceMul, ge_iff_le]
  exact h


-- Prove tightest lower bound on error is MAX_BODY_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION
def body_violation_example : Nat :=
  let (B₁, B₂, B₃) := example_bodies
  getBodyDistance B₁ B₃ - (getBodyDistance B₁ B₂ + getBodyDistance B₂ B₃)


example : body_violation_example = MAX_BODY_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION := by rfl


-- TOTAL DISTANCE PROOFS --
-- Prove upper-bound on error is MAX_TLSH_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION
theorem max_tlsh_distance : ∀ (D₁ D₂ D₃ : Digest),
  (getTlshDistance D₁ D₃) ≤ getTlshDistance D₁ D₂ + getTlshDistance D₂ D₃ + MAX_TLSH_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION := by
  intros D₁ D₂ D₃

  cases D₁ with | mk H₁ B₁ =>
  cases D₂ with | mk H₂ B₂ =>
  cases D₃ with | mk H₃ B₃ =>

  let ΔH₁₂ := getHeaderDistance H₁ H₂
  let ΔH₂₃ := getHeaderDistance H₂ H₃
  let ΔB₁₂ := getBodyDistance B₁ B₂
  let ΔB₂₃ := getBodyDistance B₂ B₃

  calc
    getTlshDistance (H₁, B₁) (H₃, B₃) = (getHeaderDistance H₁ H₃).toNat + getBodyDistance B₁ B₃ := by
      unfold getTlshDistance
      simp [getTlshDistance]
    _ ≤   (ΔH₁₂.toNat + ΔH₂₃.toNat + MAX_HEADER_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION)
        + (ΔB₁₂ + ΔB₂₃ + MAX_BODY_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION) := by
      apply Nat.add_le_add
      · apply max_header_violation
      · apply max_body_violation
    _ = (ΔH₁₂.toNat + ΔB₁₂) + (ΔH₂₃.toNat + ΔB₂₃) + MAX_TLSH_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION := by
      simp [MAX_TLSH_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION]
      simp_arith


-- Prove tightest lower bound on error is MAX_TLSH_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION
def tlsh_violation_example : Nat :=
  let ((H₁, H₂, H₃), (B₁, B₂, B₃)) := (example_headers, example_bodies)
  let (D₁, D₂, D₃) := ((H₁, B₁), (H₂, B₂), (H₃, B₃))
  getTlshDistance D₁ D₃ - (getTlshDistance D₁ D₂ + getTlshDistance D₂ D₃)


example : tlsh_violation_example = MAX_TLSH_DISTANCE_TRIANGLE_INEQUALITY_VIOLATION := by rfl
