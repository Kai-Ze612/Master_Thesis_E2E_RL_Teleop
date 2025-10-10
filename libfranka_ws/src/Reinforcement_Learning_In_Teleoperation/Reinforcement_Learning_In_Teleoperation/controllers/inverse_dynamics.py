"""
Inverse dynamics computation

The inverse dynamics computes required joint torques by:
tau = M(q)*q_dd + C(q, q_d)*q_d + G(q)

where:
- M(q) is the mass/inertia matrix
- C(q, q_d) is the Coriolis/centrifugal forces
- G(q) is the gravity vector
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DynamicsTerms:
    """ Define Dimensions of dynamics terms."""
    
    M: NDArray[np.float64]      # Mass matrix
    C_qd: NDArray[np.float64]   # Coriolis forces C(q,q̇)q̇
    G: NDArray[np.float64]      # Gravity forces
    
    def __post_init__(self) -> None:
        """Validate dimensions."""
        n = self.M.shape[0]
        if self.M.shape != (n, n):
            raise ValueError(f"M must be square, got {self.M.shape}")
        if self.C_qd.shape != (n,):
            raise ValueError(f"C_qd must be 1D with length {n}, got {self.C_qd.shape}")
        if self.G.shape != (n,):
            raise ValueError(f"G must be 1D with length {n}, got {self.G.shape}")
    
    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return self.M.shape[0]

class DynamicsComputer(ABC):
    """Interface for computing robot dynamics."""
    
    @abstractmethod
    def compute_dynamics(
        self,
        q: NDArray[np.float64],
        qd: NDArray[np.float64]
    ) -> DynamicsTerms:
        """Compute M, C_qd, G at given state."""
        pass
    
    @property
    @abstractmethod
    def n_joints(self) -> int:
        """Number of robot joints."""
        pass


class InverseDynamics:
    """Inverse dynamics solver using a dynamics computer backend."""
    
    def __init__(self, dynamics_computer: DynamicsComputer) -> None:
        """Initialize inverse dynamics solver."""
        self._dynamics = dynamics_computer
    
    @property
    def n_joints(self) -> int:
        """Number of robot joints."""
        return self._dynamics.n_joints
    
    def compute_torque(
        self,
        q: NDArray[np.float64],
        qd: NDArray[np.float64],
        qdd: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        
        # Get dynamics terms
        dynamics = self._dynamics.compute_dynamics(q, qd)
        
        # Inverse dynamics: τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
        tau = dynamics.M @ qdd + dynamics.C_qd + dynamics.G
        
        return tau