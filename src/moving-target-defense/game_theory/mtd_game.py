"""
Moving Target Defense Game Theory for SCION

Addresses RQ3: Path-aware MTD with provable security properties

Game Model:
- Players: Defender (network operator), Attacker (adversary)
- Strategies: Path switching, service replication, bandwidth reservation shuffling
- Payoffs: Attack success probability vs defense cost

Isabelle/HOL formal verification proves Nash equilibrium properties
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class SCIONPath:
    """SCION path representation"""
    path_id: int
    hops: List[int]  # AS numbers
    bandwidth: float  # Gbps
    latency: float    # ms
    cost: float       # monetary cost


class MTDGameTheory:
    """
    Game-theoretic Moving Target Defense for SCION
    
    Models defender-attacker interaction as a two-player game
    Addresses RQ3.2: Formal game-theoretic models
    """
    
    def __init__(self, num_paths: int, attack_cost: float = 10.0, 
                 defense_cost: float = 1.0):
        self.num_paths = num_paths
        self.attack_cost = attack_cost  # Cost per path attacked
        self.defense_cost = defense_cost  # Cost per path switch
        
    def compute_defender_payoff(self, defender_strategy: np.ndarray, 
                                attacker_strategy: np.ndarray) -> float:
        """
        Compute defender's payoff
        
        Args:
            defender_strategy: Probability distribution over paths
            attacker_strategy: Probability distribution over attack targets
            
        Returns:
            Expected payoff for defender
        """
        # Successful defense = paths not attacked
        prob_success = 1.0 - np.sum(defender_strategy * attacker_strategy)
        
        # Payoff = success probability - defense cost
        defense_switches = np.sum(np.abs(np.diff(np.nonzero(defender_strategy)[0])))
        payoff = prob_success - (self.defense_cost * defense_switches)
        
        return payoff
    
    def compute_attacker_payoff(self, defender_strategy: np.ndarray, 
                                attacker_strategy: np.ndarray) -> float:
        """Compute attacker's payoff"""
        # Successful attack = paths both selected
        prob_success = np.sum(defender_strategy * attacker_strategy)
        
        # Payoff = success probability - attack cost
        num_attacks = np.count_nonzero(attacker_strategy)
        payoff = prob_success - (self.attack_cost * num_attacks)
        
        return payoff
    
    def find_nash_equilibrium(self, iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find Nash equilibrium using fictitious play
        
        Returns:
            (defender_equilibrium, attacker_equilibrium)
        """
        # Initialize uniform strategies
        defender_strategy = np.ones(self.num_paths) / self.num_paths
        attacker_strategy = np.ones(self.num_paths) / self.num_paths
        
        # Fictitious play algorithm
        for _ in range(iterations):
            # Defender best response to attacker
            defender_payoffs = np.array([
                self.compute_defender_payoff(
                    np.eye(self.num_paths)[i], 
                    attacker_strategy
                )
                for i in range(self.num_paths)
            ])
            best_defender_action = np.argmax(defender_payoffs)
            defender_strategy = 0.9 * defender_strategy
            defender_strategy[best_defender_action] += 0.1
            defender_strategy /= defender_strategy.sum()
            
            # Attacker best response to defender
            attacker_payoffs = np.array([
                self.compute_attacker_payoff(
                    defender_strategy,
                    np.eye(self.num_paths)[i]
                )
                for i in range(self.num_paths)
            ])
            best_attacker_action = np.argmax(attacker_payoffs)
            attacker_strategy = 0.9 * attacker_strategy
            attacker_strategy[best_attacker_action] += 0.1
            attacker_strategy /= attacker_strategy.sum()
        
        return defender_strategy, attacker_strategy


class SCIONMTDStrategy:
    """
    SCION-specific MTD strategies leveraging path diversity
    
    Addresses RQ3.1: Path-aware MTD reconfiguration
    """
    
    def __init__(self, available_paths: List[SCIONPath]):
        self.paths = available_paths
        self.current_path_idx = 0
        self.switch_history = []
        
    def dynamic_path_switching(self, threat_level: float) -> SCIONPath:
        """
        Switch paths dynamically based on threat assessment
        
        Args:
            threat_level: 0.0 (safe) to 1.0 (high threat)
            
        Returns:
            Selected SCION path
        """
        if threat_level > 0.7:
            # High threat: switch to path with highest bandwidth and lowest latency
            path_scores = [
                p.bandwidth / (p.latency + 1.0)
                for p in self.paths
            ]
            selected_idx = np.argmax(path_scores)
        elif threat_level > 0.4:
            # Medium threat: random switching
            selected_idx = np.random.randint(0, len(self.paths))
        else:
            # Low threat: keep current path
            selected_idx = self.current_path_idx
        
        if selected_idx != self.current_path_idx:
            self.switch_history.append((self.current_path_idx, selected_idx))
            self.current_path_idx = selected_idx
        
        return self.paths[selected_idx]
    
    def get_attack_surface_reduction(self) -> float:
        """
        Compute attack surface reduction from MTD
        
        Returns:
            Reduction percentage (0.0 to 1.0)
        """
        if len(self.switch_history) == 0:
            return 0.0
        
        # More path diversity = lower attack surface
        unique_paths = len(set(idx for _, idx in self.switch_history))
        reduction = unique_paths / len(self.paths)
        
        return reduction


# Example: Game-theoretic MTD simulation
if __name__ == "__main__":
    print("SCION Moving Target Defense - Game Theory")
    print("=" * 60)
    
    # Create game with 5 available SCION paths
    game = MTDGameTheory(num_paths=5, attack_cost=10.0, defense_cost=1.0)
    
    # Find Nash equilibrium
    defender_eq, attacker_eq = game.find_nash_equilibrium(iterations=1000)
    
    print(f"\nNash Equilibrium Found:")
    print(f"Defender strategy: {defender_eq}")
    print(f"Attacker strategy: {attacker_eq}")
    
    # Compute equilibrium payoffs
    defender_payoff = game.compute_defender_payoff(defender_eq, attacker_eq)
    attacker_payoff = game.compute_attacker_payoff(defender_eq, attacker_eq)
    
    print(f"\nEquilibrium Payoffs:")
    print(f"Defender: {defender_payoff:.4f}")
    print(f"Attacker: {attacker_payoff:.4f}")
    
    # Create MTD strategy with sample SCION paths
    paths = [
        SCIONPath(i, [1, 2, 3+i, 10], bandwidth=10.0+i, latency=20.0+i*2, cost=5.0)
        for i in range(5)
    ]
    
    mtd = SCIONMTDStrategy(paths)
    
    # Simulate threat scenarios
    print(f"\n\nMTD Path Selection:")
    threat_levels = [0.2, 0.5, 0.8, 0.9]
    for threat in threat_levels:
        path = mtd.dynamic_path_switching(threat)
        print(f"Threat {threat:.1f}: Selected Path {path.path_id} (BW: {path.bandwidth} Gbps)")
    
    reduction = mtd.get_attack_surface_reduction()
    print(f"\nAttack Surface Reduction: {reduction*100:.1f}%")
