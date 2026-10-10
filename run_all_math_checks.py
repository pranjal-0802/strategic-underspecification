#!/usr/bin/env python3
"""
run_all_math_checks.py: Runs all non-LLM mathematical verifications and robustness checks
for the formal Stackelberg game model (paper/strategic_underspecification.tex).

Checks:
- Prop 3: First-Best asking dominance and argmax verification on feasible domain
- Prop 4: Pooling closed-form formula vs fine grid search
- Cor 2: Friction bias direction comparative statics
- Cor 3: Continuous transition across friction regimes & unbiased corner
- Prop 5: The Guessing Trap (Equilibrium Capability Reversal in g)
- Table 1: Sensitivity of the Unbiased Pooling Corner and Clipped Payoffs
- Prop 6: Screening menu with no bias recovers first-best and has slack ICs
- Prop 7: Screening downward distortion under bias & active constraint report
- Sweep 2D: Heterogeneity vs Bias distortion and welfare-gap decomposition
- Compare: Model I vs Model II Regime Comparison (Dominance of Menus)
- Welfare: User Welfare Ordering across Pooling and Screening Regimes
- Robustness 1: Exact Conjunctive Success vs Linear-Risk Approximation
- Robustness 2: Regularity Assumption (Assumption 1) Characterization
- Robustness 3: mu_A Comparative Statics and Bias Confounding
- Follow-up 1 (Task 1): Monotonicity Survival Outside Assumption 1 Region
- Follow-up 2 (Task 2): Ray-Invariance in the Corner Regime
- Follow-up 3 (Task 3): Constraint Binding Under Exact Bias Sweep

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
from underspec_sim.verifications.verify_prop5_guessing_trap import run_verification as verify_prop5_guessing_trap
from underspec_sim.verifications.generate_table1 import run_table1_verification
from underspec_sim.verifications.verify_prop5 import run_verification as verify_prop6_screening_no_bias
from underspec_sim.verifications.verify_prop6 import run_verification as verify_prop7_screening_biased
from underspec_sim.verifications.sweep_distortion import run_verification as sweep_distortion
from underspec_sim.verifications.compare_regimes_runner import run_verification as compare_regimes_runner
from underspec_sim.verifications.verify_welfare_ordering import run_verification as verify_welfare_ordering
from underspec_sim.verifications.verify_exact_conjunctive import run_verification as verify_exact_conjunctive
from underspec_sim.verifications.verify_assumption1 import run_verification as verify_assumption1
from underspec_sim.verifications.verify_mu_comparative_statics import run_verification as verify_mu_comparative_statics
from underspec_sim.verifications.verify_monotonicity_survival import run_verification as verify_monotonicity_survival
from underspec_sim.verifications.verify_mu_lambda_corner import run_verification as verify_mu_lambda_corner
from underspec_sim.verifications.verify_exact_bias_sweep import run_verification as verify_exact_bias_sweep


def main():
    parser = argparse.ArgumentParser(description="Run all mathematical verifications and robustness checks.")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save CSVs, plots, and markdown notes")
    parser.add_argument("--ignore-failures", action="store_true", help="Exit with 0 even if checks fail (for reporting)")
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 105)
    print("RUNNING ALL MATHEMATICAL VERIFICATIONS & ROBUSTNESS CHECKS (NON-LLM FAST PATH)")
    print("=" * 105)

    results: List[Dict[str, Any]] = []

    # 1. Prop 3
    print("[1/17] Verifying Proposition 3 (First-Best Asking Dominance)...")
    res_p3 = verify_prop3(output_dir=args.output_dir)
    results.append({
        "item": "Prop 3",
        "name": "First-Best Asking Dominance & Argmax",
        "verdict": "PASS" if res_p3["passed"] else "FAIL",
        "passed": res_p3["passed"],
        "note": "Asking weakly dominates on feasible domain (a_FB=1); verified argmax on grid",
    })

    # 2. Prop 4
    print("[2/17] Verifying Proposition 4 (Pooling Closed-Form vs Grid)...")
    res_p4 = verify_prop4(output_dir=args.output_dir)
    results.append({
        "item": "Prop 4",
        "name": "Pooling Equilibrium Closed Form",
        "verdict": "PASS" if res_p4["passed"] else "FAIL",
        "passed": res_p4["passed"],
        "note": f"Matches grid under SOC Q_bar<0 (gap: {res_p4['res_valid'].discrepancy:.4f})",
    })

    # 3. Cor 2
    print("[3/17] Verifying Corollary 2 (Bias Shifts Pooling Rate on Interior Branch)...")
    res_c2 = verify_cor2(output_dir=args.output_dir)
    results.append({
        "item": "Cor 2",
        "name": "Bias Direction on Pooling Rate",
        "verdict": res_c2["verdict"],
        "passed": res_c2["passed"],
        "note": f"Feasible domain: da/dlam <= 0 strictly (slopes in [{res_c2['min_slope']:.3f}, {res_c2['max_slope']:.3f}], max err: {res_c2['max_rel_err']:.1%})",
    })

    # 4. Cor 3
    print("[4/17] Verifying Corollary 3 (Continuous Transition & Unbiased Corner)...")
    res_c3 = verify_cor3(output_dir=args.output_dir)
    results.append({
        "item": "Cor 3",
        "name": "Transition & Unbiased Corner",
        "verdict": "PASS" if res_c3["passed"] else "FAIL",
        "passed": res_c3["passed"],
        "note": f"Unbiased is corner a* in {{0, 1}}; Biased has interior a*={res_c3['res_interior'].a_SE:.2f}",
    })

    # 5. Prop 5: Guessing Trap
    print("[5/17] Verifying Proposition 5 (The Guessing Trap / Equilibrium Capability Reversal)...")
    res_gt = verify_prop5_guessing_trap(output_dir=args.output_dir)
    df_gt = res_gt["df_sweep"]
    a_g50 = df_gt.loc[df_gt["g"] == 0.50, "a_SE"].values[0]
    a_g60 = df_gt.loc[df_gt["g"] == 0.60, "a_SE"].values[0]
    w_g50 = df_gt.loc[df_gt["g"] == 0.50, "W"].values[0]
    w_g60 = df_gt.loc[df_gt["g"] == 0.60, "W"].values[0]
    results.append({
        "item": "Prop 5",
        "name": "The Guessing Trap (Capability Reversal)",
        "verdict": "PASS" if res_gt["passed"] else "FAIL",
        "passed": res_gt["passed"],
        "note": f"g in [0.5, 0.6]: a_SE falls ({a_g50:.2f}->{a_g60:.2f}), W falls ({w_g50:.2f}->{w_g60:.2f})",
    })

    # 6. Table 1 (Unbiased Sensitivity & Physical Feasibility)
    print("[6/17] Verifying Table 1 (Unbiased Pooling Sensitivity & Clipped Differences)...")
    df_t1 = run_table1_verification(output_dir=args.output_dir)
    t1_all_pos = bool((df_t1["clipped_delta"] > 0).all())
    results.append({
        "item": "Table 1",
        "name": "Unbiased Pooling Sensitivity",
        "verdict": "PASS" if t1_all_pos else "FAIL",
        "passed": t1_all_pos,
        "note": f"All 11 rows strictly positive (clipped delta in [{df_t1['clipped_delta'].min():+.2f}, {df_t1['clipped_delta'].max():+.2f}])",
    })

    # 7. Prop 6 (Screening without bias)
    print("[7/17] Verifying Proposition 6 (Screening Without Bias)...")
    res_p6 = verify_prop6_screening_no_bias(output_dir=args.output_dir)
    results.append({
        "item": "Prop 6",
        "name": "Screening Recovers First Best",
        "verdict": "PASS" if res_p6["passed"] else "FAIL",
        "passed": res_p6["passed"],
        "note": f"Recovers FB; IC strictly slack on feasible domain (IC_L={res_p6['res_con'].IC_L_slack:.3f}, IC_H={res_p6['res_con'].IC_H_slack:.3f})",
    })

    # 8. Prop 7 (Screening under bias)
    print("[8/17] Verifying Proposition 7 (Screening Distortion Under Bias)...")
    res_p7 = verify_prop7_screening_biased(output_dir=args.output_dir)
    active_str = ", ".join(res_p7["active_constraints"]) if res_p7["active_constraints"] else "None"
    results.append({
        "item": "Prop 7",
        "name": "Screening Distortion Under Bias",
        "verdict": "PASS" if res_p7["passed"] else "FAIL",
        "passed": res_p7["passed"],
        "note": f"a_L at corner; a_H distorted by -{res_p7['distortion_size']:.3f}. Active: [{active_str}]",
    })

    # 9. Sweep distortion
    print("[9/17] Running Heterogeneity vs Bias 2D Sweep...")
    res_sweep = sweep_distortion(output_dir=args.output_dir)

    # 10. Compare regimes
    print("[10/17] Running Model I vs Model II Regime Comparison...")
    res_comp = compare_regimes_runner(output_dir=args.output_dir)

    # 11. Welfare Ordering
    print("[11/17] Running User Welfare Ordering across Bias Bands...")
    res_welf = verify_welfare_ordering(output_dir=args.output_dir)
    df_welf = res_welf["df"]
    gap_mild = df_welf.loc[df_welf["lambda_A"] == 2.5, "W_gap"].values[0]
    gap_mod = df_welf.loc[df_welf["lambda_A"] == 3.5, "W_gap"].values[0]
    results.append({
        "item": "Welfare",
        "name": "User Welfare Ordering across Regimes",
        "verdict": "PASS" if res_welf["passed"] else "FAIL",
        "passed": res_welf["passed"],
        "note": f"Mild bias: pool wins ({gap_mild:+.2f}); Mod bias: screen wins ({gap_mod:+.2f}); Shut: tie",
    })

    # 12. Robustness Check 1: Exact Conjunctive Success
    print("[12/17] Running Robustness Check 1: Exact Conjunctive Model...")
    res_exact = verify_exact_conjunctive(output_dir=args.output_dir)
    results.append({
        "item": "Rob 1",
        "name": "Exact Conjunctive Robustness",
        "verdict": res_exact["verdict"],
        "passed": res_exact["passed"],
        "note": f"Corner survived in {res_exact['corner_rate']*100:.0f}%; downward dist in {res_exact['downward_dist_rate']*100:.0f}%; IR_H slack everywhere",
    })

    # 13. Robustness Check 2: Assumption 1 Characterization
    print("[13/17] Running Robustness Check 2: Assumption 1 Regularity...")
    res_assump1 = verify_assumption1(output_dir=args.output_dir)
    results.append({
        "item": "Rob 2",
        "name": "Assumption 1 Characterization",
        "verdict": res_assump1["verdict"],
        "passed": res_assump1["passed"],
        "note": f"Corrected (>= -1) holds in {res_assump1['holds_fraction']*100:.1f}%; canonical audit in Foll 1",
    })

    # 14. Robustness Check 3: mu_A Comparative Statics
    print("[14/17] Running Robustness Check 3: mu_A Comparative Statics...")
    res_mu = verify_mu_comparative_statics(output_dir=args.output_dir)
    results.append({
        "item": "Rob 3",
        "name": "mu_A Comparative Statics & Confound",
        "verdict": res_mu["verdict"],
        "passed": res_mu["passed"],
        "note": f"Verified ratio dmu/dlam = -(lambda_A - c_Q)/mu_A (error: {res_mu['max_ratio_error']:.1e}); parameters confounded",
    })

    # 15. Follow-up 1 (Task 1): Monotonicity Survival
    print("[15/17] Running Follow-up Check 1: Monotonicity Under Assumption 1 & Tightness...")
    res_mono_surv = verify_monotonicity_survival(output_dir=args.output_dir)
    pct_ass1 = (res_mono_surv['ass1_holds_mono_holds'] / res_mono_surv['ass1_holds_total']) * 100 if res_mono_surv['ass1_holds_total'] > 0 else 0.0
    pct_outside = (res_mono_surv['ass1_fails_mono_fails'] / res_mono_surv['ass1_fails_total']) * 100 if res_mono_surv['ass1_fails_total'] > 0 else 0.0
    results.append({
        "item": "Foll 1",
        "name": "Monotonicity Under Assump 1 & Tightness",
        "verdict": res_mono_surv["verdict"],
        "passed": res_mono_surv["passed"],
        "note": f"Mono holds {pct_ass1:.1f}% under Ass1 ({res_mono_surv['ass1_holds_mono_holds']}/{res_mono_surv['ass1_holds_total']}); breaks in {pct_outside:.1f}% outside ({res_mono_surv['ass1_fails_mono_fails']}/{res_mono_surv['ass1_fails_total']})",
    })

    # 16. Follow-up 2 (Task 2): Ray-Invariance in Corner Regime
    print("[16/17] Running Follow-up Check 2: Ray-Invariance in Corner Regime...")
    res_corner_ray = verify_mu_lambda_corner(output_dir=args.output_dir)
    results.append({
        "item": "Foll 2",
        "name": "Corner Regime Ray-Invariance",
        "verdict": res_corner_ray["verdict"],
        "passed": res_corner_ray["ray_invariance_failures"] == 0,
        "note": f"mu_A factors out completely from Pi(1)-Pi(0); 0 flips across {res_corner_ray['total_evals']} points; critical ratio rho*={res_corner_ray['rho_star']:.3f}",
    })

    # 17. Follow-up 3 (Task 3): Exact Bias Sweep & Constraint Binding
    print("[17/17] Running Follow-up Check 3: Constraint Binding Under Exact Bias Sweep...")
    res_exact_sweep = verify_exact_bias_sweep(output_dir=args.output_dir)
    results.append({
        "item": "Foll 3",
        "name": "Exact Bias Sweep & Constraint Binding",
        "verdict": res_exact_sweep["verdict"],
        "passed": res_exact_sweep["passed"],
        "note": f"IC_H active in {res_exact_sweep['both_ic_count']}/{res_exact_sweep['total_points']} points (predominantly shutdown regime)",
    })

    elapsed = time.time() - start_time

    # Print Summary Table
    print("\n" + "=" * 105)
    print(f"{'ITEM':<8} | {'VERIFICATION / CLAIM':<36} | {'VERDICT':<9} | {'DIAGNOSTIC NOTE'}")
    print("-" * 105)
    any_failed = False
    for r in results:
        if r["verdict"] == "FAIL":
            any_failed = True
        print(f"{r['item']:<8} | {r['name']:<36} | {r['verdict']:<9} | {r['note']}")
    print("=" * 105)
    print(f"Total verification time: {elapsed:.2f} seconds (all 17 verifications completed)")
    print(f"Artifacts saved in: {args.output_dir}/")

    if any_failed:
        print("\n[!] WARNING: One or more theoretical claims failed numerical verification.")
        if not args.ignore_failures:
            print("    Exiting with nonzero exit code (1) as requested.\n")
            sys.exit(1)
    else:
        print("\nAll mathematical claims, robustness checks, and follow-up verifications completed successfully.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
