"""
Inverse dynamics computation

The inverse dynamics computes required joint torques by:
tau = M(q)*q_dd + C(q, q_d)*q_d + G(q)

dimensions:
- tau: n * 1 
- M: n * n
- C: n * n
- G: n * 1
- q: n * 1
- q_d: n * 1
- q_dd: n * 1

where:
- M(q) is the mass/inertia matrix
- C(q, q_d) is the Coriolis/centrifugal forces
- G(q) is the gravity vector
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)  # Once the class is created, the data will be forzen because the M, C, G should not be changed
class DynamicsTerms:
    """ Define Dimensions of dynamics terms."""
    
    M: NDArray[np.float64]      # Mass matrix
    C: NDArray[np.float64]      # Coriolis forces
    G: NDArray[np.float64]      # Gravity forces
    
    def __post_init__(self) -> None:
        """Validate dimensions."""
        n = self.M.shape[0]
        if self.M.shape != (n, n):
            raise ValueError(f"M must be square, got {self.M.shape}")
        if self.C.shape != (n, n):
            raise ValueError(f"C_qd must be square, got {self.C.shape}")
        if self.G.shape != (n,):
            raise ValueError(f"G must be 1D with length {n}, got {self.G.shape}")
    
    @property
    def n_joints(self) -> int:
        """Number of joints."""
        return self.M.shape[0]

class DynamicsComputer(ABC):
    """Interface for computing robot dynamics."""
    
    @abstractmethod
    def compute_dynamics(self,
                         q: NDArray[np.float64],
                         qd: NDArray[np.float64]) -> DynamicsTerms:
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
    
    def compute_torque(self,
                       q: NDArray[np.float64],
                       qd: NDArray[np.float64],
                       qdd: NDArray[np.float64]) -> NDArray[np.float64]:
        
        # Input validation
        n = self.n_joints
        if q.shape != (n,) or qd.shape != (n,) or qdd.shape != (n,):
            raise ValueError(
                f"All inputs must be ({n},), "
                f"got q={q.shape}, qd={qd.shape}, qdd={qdd.shape}"
            )
        
        # Get dynamics terms
        dynamics = self._dynamics.compute_dynamics(q, qd)
        
        # Inverse dynamics: τ = M(q)q̈ + C(q,q̇)q̇ + G(q)
        tau = dynamics.M @ qdd + dynamics.C @ qd + dynamics.G
        
        return tau