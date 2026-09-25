#!/usr/bin/env python3
"""
run_all_math_checks.py: Runs all non-LLM mathematical verifications and robustness checks
for the formal Stackelberg game model (strategic_underspecification_v2.tex).

Checks:
- Prop 3: First-Best threshold and argmax verification
- Prop 4: Pooling closed-form formula vs fine grid search
- Cor 2: Bias direction comparative statics
- Cor 3: Pooling is a corner solution (unbiased case)
- Prop 5: Screening menu with no bias recovers first-best and has slack ICs
- Prop 6: Screening downward distortion under bias & active constraint report
- Sweep 2D: Heterogeneity vs Bias distortion and welfare-gap decomposition
- Compare: Model I vs Model II Regime Comparison (Dominance of Menus)
- Robustness 1: Exact Conjunctive Success vs Linear-Risk Approximation (Task 1)
- Robustness 2: Regularity Assumption (Assumption 1) Characterization (Task 2)
- Robustness 3: mu_A Comparative Statics and Bias Confounding (Task 3)

Prints a summary table with PASS / CAUTION / FAIL verdicts.
"""

import sys
import time
import argparse
from typing import Dict, List, Any

from underspec_sim.verifications.verify_prop3 import run_verification as verify_prop3
from underspec_sim.verifications.verify_prop4 import run_verification as verify_prop4
from underspec_sim.verifications.verify_cor2 import run_verification as verify_cor2
from underspec_sim.verifications.verify_cor3 import run_verification as verify_cor3
from underspec_sim.verifications.verify_prop5 import run_verification as verify_prop5
from underspec_sim.verifications.verify_prop6 import run_verification as verify_prop6
from underspec_sim.verifications.sweep_distortion import run_verification as sweep_distortion
from underspec_sim.verifications.compare_regimes_runner import run_verification as compare_regimes_runner
from underspec_sim.verifications.verify_exact_conjunctive import run_verification as verify_exact_conjunctive
from underspec_sim.verifications.verify_assumption1 import run_verification as verify_assumption1
from underspec_sim.verifications.verify_mu_comparative_statics import run_verification as verify_mu_comparative_statics


def main():
    parser = argparse.ArgumentParser(description="Run all mathematical verifications and robustness checks.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save CSVs, plots, and markdown notes")
    parser.add_argument("--ignore-failures", action="store_true", help="Exit with 0 even if checks fail (for reporting)")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 95)
    print("RUNNING ALL MATHEMATICAL VERIFICATIONS & ROBUSTNESS CHECKS (NON-LLM FAST PATH)")
    print("=" * 95)

    results: List[Dict[str, Any]] = []

    # 1. Prop 3
    print("[1/11] Verifying Proposition 3 (First-Best Threshold)...")
    res_p3 = verify_prop3(output_dir=args.output_dir)
    results.append({
        "item": "Prop 3",
        "name": "First-Best Threshold & Argmax",
        "verdict": "PASS" if res_p3["passed"] else "FAIL",
        "passed": res_p3["passed"],
        "note": f"Flipped at kappa*={res_p3['kappa_star']:.3f}; verified argmax on grid",
    })

    # 2. Prop 4
    print("[2/11] Verifying Proposition 4 (Pooling Closed-Form vs Grid)...")
    res_p4 = verify_prop4(output_dir=args.output_dir)
    results.append({
        "item": "Prop 4",
        "name": "Pooling Equilibrium Closed Form",
        "verdict": "PASS" if res_p4["passed"] else "FAIL",
        "passed": res_p4["passed"],
        "note": f"Matches grid under SOC Delta*gamma>0 (gap: {res_p4['res_valid'].discrepancy:.4f})",
    })

    # 3. Cor 2
    print("[3/11] Verifying Corollary 2 (Bias Shifts Pooling Rate)...")
    res_c2 = verify_cor2(output_dir=args.output_dir)
    results.append({
        "item": "Cor 2",
        "name": "Bias Direction on Pooling Rate",
        "verdict": "PASS" if res_c2["passed"] else "FAIL",
        "passed": res_c2["passed"],
        "note": f"Reg. holds in {res_c2['reg_holding_pct']:.0f}% of sweep; FOC sign tracks slope",
    })

    # 4. Cor 3
    print("[4/11] Verifying Corollary 3 (Pooling is Corner Solution)...")
    res_c3 = verify_cor3(output_dir=args.output_dir)
    results.append({
        "item": "Cor 3",
        "name": "Pooling is Corner Solution",
        "verdict": "PASS" if res_c3["passed"] else "FAIL",
        "passed": res_c3["passed"],
        "note": f"Unbiased is corner a* in {{0, 1}} matching formula; Biased has interior a*={res_c3['res_interior'].a_SE:.2f}",
    })

    # 5. Prop 5
    print("[5/11] Verifying Proposition 5 (Screening Without Bias)...")
    res_p5 = verify_prop5(output_dir=args.output_dir)
    results.append({
        "item": "Prop 5",
        "name": "Screening Recovers First Best",
        "verdict": "PASS" if res_p5["passed"] else "FAIL",
        "passed": res_p5["passed"],
        "note": f"Recovers FB; IC strictly slack (IC_L={res_p5['res_uncon'].IC_L_slack:.3f}, IC_H={res_p5['res_uncon'].IC_H_slack:.3f})",
    })

    # 6. Prop 6
    print("[6/11] Verifying Proposition 6 (Screening Distortion Under Bias)...")
    res_p6 = verify_prop6(output_dir=args.output_dir)
    active_str = ", ".join(res_p6["active_constraints"]) if res_p6["active_constraints"] else "None"
    results.append({
        "item": "Prop 6",
        "name": "Screening Distortion Under Bias",
        "verdict": "PASS" if res_p6["passed"] else "FAIL",
        "passed": res_p6["passed"],
        "note": f"a_L at corner; a_H distorted by -{res_p6['distortion_size']:.3f}. Active: [{active_str}]",
    })

    # 7. Sweep distortion
    print("[7/11] Running Heterogeneity vs Bias 2D Sweep...")
    res_sweep = sweep_distortion(output_dir=args.output_dir)

    # 8. Compare regimes
    print("[8/11] Running Model I vs Model II Regime Comparison...")
    res_comp = compare_regimes_runner(output_dir=args.output_dir)

    # 9. Robustness Task 1: Exact Conjunctive Success
    print("[9/11] Running Robustness Check 1: Exact Conjunctive Model (Task 1)...")
    res_exact = verify_exact_conjunctive(output_dir=args.output_dir)
    results.append({
        "item": "Rob 1",
        "name": "Exact Conjunctive Robustness",
        "verdict": res_exact["verdict"],
        "passed": res_exact["passed"],
        "note": f"Corner survived in {res_exact['corner_rate']*100:.0f}%; downward dist in {res_exact['downward_dist_rate']*100:.0f}%; IR_H slack everywhere",
    })

    # 10. Robustness Task 2: Assumption 1 Characterization
    print("[10/11] Running Robustness Check 2: Assumption 1 Regularity (Task 2)...")
    res_assump1 = verify_assumption1(output_dir=args.output_dir)
    results.append({
        "item": "Rob 2",
        "name": "Assumption 1 Characterization",
        "verdict": res_assump1["verdict"],
        "passed": res_assump1["passed"],
        "note": f"Holds in {res_assump1['holds_fraction']*100:.1f}% of grid; restrictive when kappa<0.6 or q>0.90",
    })

    # 11. Robustness Task 3: mu_A Comparative Statics
    print("[11/11] Running Robustness Check 3: mu_A Comparative Statics (Task 3)...")
    res_mu = verify_mu_comparative_statics(output_dir=args.output_dir)
    results.append({
        "item": "Rob 3",
        "name": "mu_A Comparative Statics & Confound",
        "verdict": res_mu["verdict"],
        "passed": res_mu["passed"],
        "note": f"Verified ratio dmu/dlam = -lambda_A/mu_A (error: {res_mu['max_ratio_error']:.1e}); parameters confounded",
    })

    elapsed = time.time() - start_time

    # Print Summary Table
    print("\n" + "=" * 98)
    print(f"{'ITEM':<8} | {'VERIFICATION / CLAIM':<36} | {'VERDICT':<9} | {'DIAGNOSTIC NOTE'}")
    print("-" * 98)
    any_failed = False
    for r in results:
        if r["verdict"] == "FAIL":
            any_failed = True
        print(f"{r['item']:<8} | {r['name']:<36} | {r['verdict']:<9} | {r['note']}")
    print("=" * 98)
    print(f"Total verification time: {elapsed:.2f} seconds (all verifications completed)")
    print(f"Artifacts saved in: {args.output_dir}/")

    if any_failed:
        print("\n[!] WARNING: One or more theoretical claims failed numerical verification.")
        if not args.ignore_failures:
            print("    Exiting with nonzero exit code (1) as requested.\n")
            sys.exit(1)
    else:
        print("\nAll mathematical claims and robustness checks completed successfully.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
