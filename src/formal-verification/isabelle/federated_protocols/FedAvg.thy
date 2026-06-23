theory FedAvg
  imports Main "../adversary/StrategicAdversary"
begin

(*
 * Federated Averaging meets the unified strategic adversary (RQ1.2, RQ0.1).
 *
 * This theory connects the aggregation protocol to the SAME adversary used for
 * MTD (see ../control_loop/ControlLoop.thy), so the detector's robustness target
 * is not a fresh, weaker model. Convergence-only results live in
 * FedAvg_Convergence.thy; here we state the adversary-coupling obligation.
 *)

section \<open>Protocol state\<close>

type_synonym weight = "nat \<Rightarrow> real"
type_synonym client_update = "weight \<times> nat"   \<comment> \<open>(weights, dataset size)\<close>

\<comment> \<open>The deployed global model induces a detector over feature vectors.\<close>
consts detector_of :: "weight \<Rightarrow> detector"

section \<open>Adversary-coupled robustness goal\<close>

\<comment> \<open>Byzantine-robust aggregation under f < n/3 corrupted clients, AND the
   resulting detector is robust to the unified strategic adversary A. The two
   requirements are stated together on purpose -- that pairing is the
   contribution (proposal Gap 3a). Proof obligations are Phase-1 work.\<close>
theorem fedavg_byzantine_and_strategic_robust:
  fixes A :: adversary and theta eps :: real
  fixes updates :: "client_update list" and g :: weight
  assumes "well_formed A"
  assumes byzantine_bound: "card {i. i < length updates \<and> snd (updates ! i) = 0} < length updates div 3"
  assumes aggregated: "g = g"   \<comment> \<open>placeholder for the verified aggregation output\<close>
  shows "\<exists>theta eps. eps \<ge> 0 \<and> robust_against (detector_of g) A theta eps"
  sorry

end
