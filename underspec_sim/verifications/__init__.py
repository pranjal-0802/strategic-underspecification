"""
Verification scripts and runners.
"""

from underspec_sim.verifications.verify_prop3 import run_verification as verify_prop3
from underspec_sim.verifications.verify_prop4 import run_verification as verify_prop4
from underspec_sim.verifications.verify_cor2 import run_verification as verify_cor2
from underspec_sim.verifications.verify_cor3 import run_verification as verify_cor3
from underspec_sim.verifications.verify_prop5 import run_verification as verify_prop5
from underspec_sim.verifications.verify_prop6 import run_verification as verify_prop6
from underspec_sim.verifications.sweep_distortion import run_verification as sweep_distortion
from underspec_sim.verifications.compare_regimes_runner import run_verification as compare_regimes_runner

__all__ = [
    "verify_prop3",
    "verify_prop4",
    "verify_cor2",
    "verify_cor3",
    "verify_prop5",
    "verify_prop6",
    "sweep_distortion",
    "compare_regimes_runner",
]
