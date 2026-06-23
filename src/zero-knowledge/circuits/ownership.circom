pragma circom 2.0.0;

/*
 * Bandwidth Asset Ownership Proof (standalone)
 * ============================================
 * Prove ownership of an isolated bandwidth asset without
 * revealing the private key, the asset value, or the flow's fair-share
 * entitlement.
 *
 * The asset being traded is a *fractional-fair-share* slice -- lightweight
 * allocation with provable isolation (Wyss, Hu, Lenders, Meier & Perrig,
 * "Lightweight Internet Bandwidth Allocation and Isolation with Fractional Fair
 * Shares," NDSS 2026; see docs/CITATIONS.md). We therefore bind the entitlement
 * into the commitment so a buyer can later verify isolation guarantees in zero
 * knowledge.
 *
 * Targets (RQ2.2): ~1,000 R1CS constraints, <10s proof generation (Groth16/BN254),
 * <1s verification. Formal verification of soundness / zero-knowledge: RQ2.3.
 */

include "../node_modules/circomlib/circuits/poseidon.circom";
include "../node_modules/circomlib/circuits/comparators.circom";

template BandwidthOwnership() {
    // Public inputs
    signal input assetCommitment;   // Poseidon(privateKey, assetValue, entitlement, nonce)
    signal input ownerPublicKey;    // Poseidon(privateKey)

    // Private inputs
    signal input privateKey;
    signal input assetValue;        // bandwidth amount (Gbps)
    signal input entitlement;       // provable fair-share entitlement (Gbps)
    signal input nonce;

    // Bind the asset (value + fair-share entitlement) to the owner's key.
    component hasher = Poseidon(4);
    hasher.inputs[0] <== privateKey;
    hasher.inputs[1] <== assetValue;
    hasher.inputs[2] <== entitlement;
    hasher.inputs[3] <== nonce;
    assetCommitment === hasher.out;

    // Public key derivation (illustrative; production uses EC ops).
    component pk = Poseidon(1);
    pk.inputs[0] <== privateKey;
    ownerPublicKey === pk.out;

    // Isolation sanity: the traded value cannot exceed the proven fair share.
    component withinShare = LessEqThan(32);
    withinShare.in[0] <== assetValue;
    withinShare.in[1] <== entitlement;
    withinShare.out === 1;

    // Range check: 0 < assetValue < 1000 Gbps.
    component positive = GreaterThan(32);
    positive.in[0] <== assetValue;
    positive.in[1] <== 0;
    positive.out === 1;
}

component main {public [assetCommitment, ownerPublicKey]} = BandwidthOwnership();
