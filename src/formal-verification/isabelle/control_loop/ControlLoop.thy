theory ControlLoop
  imports Main "../adversary/StrategicAdversary"
begin

(*
 * Adversary-Coupled Verified Control Loop  (RQ0.2, RQ0.3)
 * -------------------------------------------------------
 * Formal counterpart of src/control-loop/closed_loop.py.
 *
 * The whole point: detection robustness and the MTD response policy are stated
 * against the SAME adversary A, and the loop carries a single end-to-end
 * correctness statement that also bounds per-iteration latency for the
 * cyber-physical regime (Zhang et al., SCION fast frequency response, 2026).
 * The refinement from this specification down to Gobra-verified Go code follows
 * the Protocols-to-Code methodology (Pereira et al., CCS 2025).
 *)

section \<open>Loop state and actions\<close>

type_synonym path = nat
type_synonym pathset = "path set"

record loop_state =
  paths   :: pathset      \<comment> \<open>available SCION paths\<close>
  active  :: path         \<comment> \<open>currently used path\<close>
  safe    :: bool         \<comment> \<open>physical safety invariant holds (e.g. freq in band)\<close>

\<comment> \<open>A response policy chooses the next active path from a threat level.\<close>
type_synonym policy = "real \<Rightarrow> loop_state \<Rightarrow> path"

\<comment> \<open>One loop step: observe (possibly under evasion), detect, decide, reconfigure.\<close>
definition step ::
  "detector \<Rightarrow> policy \<Rightarrow> adversary \<Rightarrow> real \<Rightarrow> feature \<Rightarrow> loop_state \<Rightarrow> loop_state" where
  "step D pol A theta x s =
     (let x'   = evade A x;                      \<comment> \<open>adversary acts every step\<close>
          lvl  = D x';                           \<comment> \<open>detector score\<close>
          p'   = pol lvl s
      in s\<lparr> active := p' \<rparr>)"

section \<open>Equilibrium response against the SAME adversary\<close>

\<comment> \<open>Abstract predicate: 'pol' is an equilibrium response against A
   (instantiated by the MTD game in src/moving-target-defense).\<close>
consts equilibrium_against :: "policy \<Rightarrow> adversary \<Rightarrow> bool"

\<comment> \<open>Abstract per-step latency and the cyber-physical budget.\<close>
consts step_latency :: "loop_state \<Rightarrow> real"
consts latency_budget :: real

section \<open>End-to-end correctness obligation\<close>

\<comment> \<open>If the detector is robust to A, the policy is an equilibrium against the
   SAME A, and every step meets the latency budget, then the loop preserves the
   physical safety invariant. This is the statement RQ0 promises to discharge by
   refinement; the proof is Phase-1 work.\<close>
theorem loop_preserves_safety:
  fixes D :: detector and pol :: policy and A :: adversary
  fixes theta eps :: real and x :: feature and s :: loop_state
  assumes wf:    "well_formed A"
  assumes rob:   "robust_against D A theta eps"
  assumes equil: "equilibrium_against pol A"
  assumes lat:   "step_latency s \<le> latency_budget"
  assumes pre:   "safe s"
  shows "safe (step D pol A theta x s) \<and> step_latency s \<le> latency_budget"
  sorry

end
