theory StrategicAdversary
  imports Main
begin

(*
 * Unified Strategic Adversary  (RQ0.1)
 * ------------------------------------
 * ONE adversary specification, reused as the robustness target for federated
 * detection (RQ1) and as the opponent in the path-aware MTD game (RQ3). This is
 * the formal counterpart of src/federated-learning/adversary/strategic_adversary.py.
 *
 * Threat model conceptually after (see docs/CITATIONS.md):
 *   - Da Dalt & Perrig, "Strategic Games and Zero-Shot Attacks on Heavy-Hitter
 *     Network Flow Monitoring," NDSS 2026   (zero-shot evasion)
 *   - Xu, Duan, Cai & Perrig, "Resolve the Unresolved: Systematic Work Profiling
 *     for DNS Resolvers," IEEE S&P 2026     (work-asymmetry exhaustion)
 *
 * Style note: like the existing theories in this repo, deep proofs are left as
 * `sorry` obligations to be discharged during Phase 1; the value here is the
 * precise statement of what must hold.
 *)

section \<open>Observations and detectors\<close>

type_synonym feature  = "nat \<Rightarrow> real"   \<comment> \<open>a feature vector\<close>
type_synonym detector = "feature \<Rightarrow> real" \<comment> \<open>score in [0,1]; attack iff > threshold\<close>

\<comment> \<open>Abstract L2 distance on feature space (kept uninterpreted on purpose).\<close>
consts l2 :: "feature \<Rightarrow> feature \<Rightarrow> real"

section \<open>The adversary\<close>

record adversary =
  evade  :: "feature \<Rightarrow> feature"  \<comment> \<open>zero-shot evasion map\<close>
  budget :: real                   \<comment> \<open>L2 perturbation budget\<close>
  amp    :: real                   \<comment> \<open>defender-work / attacker-work ratio\<close>

definition well_formed :: "adversary \<Rightarrow> bool" where
  "well_formed A \<longleftrightarrow> (\<forall>x. l2 x (evade A x) \<le> budget A) \<and> amp A \<ge> 1"

section \<open>Robustness target for detection (RQ1)\<close>

\<comment> \<open>D is robust to A at threshold theta with margin eps if evasion cannot push a
   genuine-attack score below the decision boundary by more than eps.\<close>
definition robust_against ::
  "detector \<Rightarrow> adversary \<Rightarrow> real \<Rightarrow> real \<Rightarrow> bool" where
  "robust_against D A theta eps \<longleftrightarrow>
     (\<forall>x. D x > theta \<longrightarrow> D (evade A x) > theta - eps)"

section \<open>Work-asymmetry safety (RQ0.3)\<close>

\<comment> \<open>On a latency-critical path the amplification factor must stay bounded
   (the proposal targets < 2x).\<close>
definition work_safe :: "adversary \<Rightarrow> real \<Rightarrow> bool" where
  "work_safe A b \<longleftrightarrow> amp A \<le> b"

\<comment> \<open>Obligation: a detector trained with adversarial augmentation against A is
   robust to A. To be discharged in Phase 1.\<close>
theorem adversarial_training_yields_robustness:
  fixes D :: detector and A :: adversary
  assumes "well_formed A"
  shows "\<exists>theta eps. eps \<ge> 0 \<and> robust_against D A theta eps"
  sorry

end
