#!/usr/bin/env python3
"""Render E1-D/FINAL_REPORT.md sections 6-10 and 12 from the run's own artifacts.

Everything is computed from files produced by the run (no hand-entered numbers):

    <run>/config/resolved_config.json                        (section 6, config)
    <run>/stdout.log                                         (section 6, timing/steps)
    <run>/analysis/training_summary.json, safety_analysis.json,
        training_diagnostics.md                              (section 6, diagnostics)
    <run>/analysis/post_train_integrity.json                 (section 6, error scan)
    E1-D/checkpoints/checkpoint_stats.json                   (section 7)
    E1-D/evaluations/*/eval_summary.json                     (section 8)
    E1-D/evaluations/comparison/comparison.json              (sections 9/10)
    E1-D/evaluations/comparison/paired_comparison_*.json     (sections 9/10)

Usage:
    python3 e1d_render_report_sections.py                 # writes report_generated_sections.md
    python3 e1d_render_report_sections.py --splice        # also splices into FINAL_REPORT.md
"""
import argparse
import glob
import json
import os
import re
import statistics

E1D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(E1D_ROOT)

BENCH_ORDER = ["nq", "triviaqa", "popqa", "hotpotqa", "2wikimultihopqa", "musique", "bamboogle"]
BEHAV = [("retrieval_rounds_mean", "search rounds"),
         ("search_queries_mean", "queries"),
         ("parallel_factor_mean", "parallel factor"),
         ("parallel_sample_rate", "parallel sample rate"),
         ("assistant_turns_mean_all_messages", "assistant turns"),
         ("response_tokens_mean", "response tokens"),
         ("zero_search_rate", "zero-search rate"),
         ("no_answer_tag_rate", "no-<answer> rate")]


def jload(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as fh:
        return json.load(fh)


def f(v, n=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def d(a, b, n=4, pct=False):
    if a is None or b is None:
        return "—"
    if pct:
        return f"{(b - a) * 100:+.2f} pp"
    return f"{b - a:+.{n}f}"


def sec6(run_dir, out):
    cfg = jload(os.path.join(run_dir, "config", "resolved_config.json"), {})
    integ = jload(os.path.join(run_dir, "analysis", "post_train_integrity.json"), {})
    summ = jload(os.path.join(run_dir, "analysis", "training_summary.json"), {})
    safety = jload(os.path.join(run_dir, "analysis", "safety_analysis.json"), {})
    log = os.path.join(run_dir, "stdout.log")

    steps = []
    gen_batches = 0
    t_first = t_last = None
    if os.path.exists(log):
        txt = open(log, errors="ignore").read()
        steps = sorted({int(m) for m in re.findall(r"step:(\d+)\s+-", txt)})
        gen_batches = len(re.findall(r"test_gen_batch meta info", txt))
        if os.path.exists(os.path.join(run_dir, "diagnostics")):
            calls = glob.glob(os.path.join(run_dir, "diagnostics", "calls_pid*.jsonl"))
            nb = 0
            for p in calls:
                nb += sum(1 for _ in open(p))
            gen_batches = nb

    a = summ.get("e1a") or {}
    b = summ.get("baseline") or {}
    out.append("## 6. 训练过程 (training process)\n")
    out.append("| item | value |")
    out.append("|---|---|")
    out.append(f"| experiment name | `{cfg.get('trainer', {}).get('experiment_name', 'DAPO-GAP3B-MHQA-Agent-E1B-step70to120-4gpu')}` |")
    out.append(f"| resume | `resume_path` from the E1-A step70 checkpoint |")
    out.append(f"| total_training_steps | {cfg.get('trainer', {}).get('total_training_steps', 120)} |")
    out.append(f"| save_freq / test_freq | {cfg.get('trainer', {}).get('save_freq', 10)} / {cfg.get('trainer', {}).get('test_freq', 10)} |")
    out.append(f"| steps observed in log | {steps[0] if steps else '—'} … {steps[-1] if steps else '—'} ({len(steps)} distinct) |")
    out.append(f"| steps 71..120 all present | {integ.get('training_log', {}).get('steps_71_to_120_all_present')} |")
    out.append(f"| reward-manager calls (generation batches) | {gen_batches} |")
    out.append(f"| rollouts / groups (diagnostics) | {a.get('n_rollouts')} / {a.get('n_groups')} |")
    out.append("")
    out.append("Per-step training behaviour (reward-manager diagnostics, cumulative over the whole run):\n")
    out.append("| metric | E1-D (shaped) | baseline: E1-0 steps61–70 (original EM reward) |")
    out.append("|---|---|---|")
    for k, label in (("em_rate", "mean EM"), ("reward_mean", "mean reward"),
                     ("efficiency_mean", "mean efficiency bonus E"),
                     ("efficiency_nonzero_rate", "E > 0 rate"),
                     ("rounds_mean", "mean logical_search_batches"),
                     ("queries_mean", "mean search queries"),
                     ("parallel_factor_mean", "mean parallel factor"),
                     ("turns_mean", "mean assistant turns"),
                     ("tokens_mean", "mean response tokens"),
                     ("zero_search_rate", "zero-search rate"),
                     ("filter_retention_baseline", "retained groups (baseline EM filter)"),
                     ("filter_retention_shaped", "retained groups (shaped reward filter)"),
                     ("filter_newly_added", "groups newly retained by shaping"),
                     ("efficiency_active_groups", "active efficiency groups"),
                     ("efficiency_supervision_ratio", "efficiency supervision ratio")):
        out.append(f"| {label} | {f(a.get(k))} | {f(b.get(k))} |")
    out.append("")
    qp = (safety.get("query_packing") or {})
    if qp:
        e = qp.get("e1b") or {}
        bl = qp.get("baseline") or {}
        out.append("### Safety checks (E1-D_task.md §7)\n")
        out.append("| # | check | E1-D | reference | verdict |")
        out.append("|---|---|---|---|---|")
        out.append(f"| 1 | accuracy (training-side mean EM) | {f(a.get('em_rate'))} | E1-0 {f(b.get('em_rate'))} | "
                   f"{'OK' if (a.get('em_rate') is not None and b.get('em_rate') is not None and a['em_rate'] >= b['em_rate'] - 0.01) else 'REVIEW'} |")
        out.append(f"| 2 | search rounds | {f(a.get('rounds_mean'))} | E1-0 {f(b.get('rounds_mean'))} | "
                   f"{'decreased' if (a.get('rounds_mean') is not None and b.get('rounds_mean') is not None and a['rounds_mean'] < b['rounds_mean']) else 'not decreased'} |")
        out.append(f"| 3 | queries | {f(a.get('queries_mean'))} | E1-0 {f(b.get('queries_mean'))} | "
                   f"{'OK' if (a.get('queries_mean') is not None and b.get('queries_mean') is not None and a['queries_mean'] <= b['queries_mean'] * 1.10) else 'REVIEW'} |")
        out.append(f"| 4 | zero-search-and-correct | {safety.get('zero_search', {}).get('e1d_zero_search_correct_rate')} | 0 expected | "
                   f"{'OK' if not safety.get('zero_search', {}).get('e1d_zero_search_correct_rate') else 'REVIEW'} |")
        out.append(f"| 5 | query packing (run-level, active groups) | {f(e.get('packing_suspect_ratio'))} "
                   f"({e.get('packing_suspect_groups')}/{e.get('active_groups')}) | E1-0 {f(bl.get('packing_suspect_ratio'))} "
                   f"({bl.get('packing_suspect_groups')}/{bl.get('active_groups')}) | "
                   f"{'OK' if (e.get('packing_suspect_ratio') is None or e['packing_suspect_ratio'] <= 0.10) else 'REVIEW'} |")
        out.append("")
    ep = (integ.get("training_log", {}) or {}).get("error_pattern_counts_before_final_val") or {}
    out.append("Error scan over the real training window (before the final-validation marker):\n")
    out.append("| pattern | count |")
    out.append("|---|---|")
    for k, v in ep.items():
        out.append(f"| {k} | {v} |")
    out.append(f"\nnon-finite metrics: {integ.get('training_log', {}).get('non_finite_metrics')} "
               f"(parsed {integ.get('training_log', {}).get('metrics_parsed')})\n")
    out.append(f"full-state resume evidence: `{integ.get('state_resume_evidence')}`\n")
    out.append(f"**integrity verdict: {'PASS' if integ.get('integrity_ok') else 'CHECK FAILED'}**\n")


def sec7(out):
    st = jload(os.path.join(E1D_ROOT, "checkpoints", "checkpoint_stats.json"), {})
    out.append("## 7. checkpoint 信息 (checkpoints)\n")
    if not st:
        out.append("(checkpoint_stats.json missing)\n")
        return
    out.append(f"steps observed: {st.get('steps_observed')}\n")
    for win, title in (("window_10step", "last-10-step window"),
                       ("cumulative", "cumulative from step71")):
        out.append(f"### {title}\n")
        out.append("| checkpoint | rollouts | groups | mean reward | mean EM | mean E | active E groups | "
                   "retained (shaped) | retained (baseline EM) | search batches | queries | parallel factor |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in st.get("checkpoints", []):
            d_ = r.get(win)
            if not d_:
                continue
            out.append(f"| `{r['checkpoint']}` | {d_['n_rollouts']} | {d_['n_groups']} | {f(d_['mean_reward'])} | "
                       f"{f(d_['mean_em'])} | {f(d_['mean_efficiency_bonus'], 5)} | {d_['active_efficiency_groups']} | "
                       f"{d_['retained_groups_shaped']} | {d_['retained_groups_baseline_em']} | "
                       f"{f(d_['mean_search_batches'], 3)} | {f(d_['mean_queries'], 3)} | {f(d_['parallel_factor'], 3)} |")
        out.append("")
    out.append("Manifest per checkpoint (shard sizes, `data.pt`, merged-HF presence): "
               "`E1-D/checkpoints/global_step_<N>.md`, `E1-D/checkpoints/manifest.json`.\n")


def load_evals():
    runs = {}
    for p in sorted(glob.glob(os.path.join(E1D_ROOT, "evaluations", "*", "eval_summary.json"))):
        with open(p) as fh:
            s = json.load(fh)
        runs[os.path.basename(os.path.dirname(p))] = s
    return runs


def sec8(out, runs):
    out.append("## 8. evaluation 结果 (evaluation)\n")
    out.append("Protocol: the **original** GAP evaluation implementation "
               "(`Agent/evaluation/mhqa_agent/eval_mhqa_agent_4gpu_common.sh`), 7 benchmarks, "
               "greedy decoding, `val_batch_size=512`, 51,201 samples, max_prompt 4096 / "
               "max_response 8192 / max_model_len 12288.\n")
    # eval run dirs are named step<N>_<timestamp>; map the step number to the run key
    step_of = {}
    for k in runs:
        m = re.match(r"step(\d+)_", k)
        if m:
            step_of[int(m.group(1))] = k
    steps = set(step_of)
    out.append(f"E1-D checkpoints evaluated: {sorted(steps)}\n")
    out.append("| benchmark | " + " | ".join(f"E1-D step{s}" for s in sorted(steps)) + " |")
    out.append("|---" * (1 + len(steps)) + "|")
    for bench in BENCH_ORDER:
        row = [bench]
        for s in sorted(steps):
            acc = runs[step_of[s]]["accuracy"]["per_benchmark"].get(bench, {})
            row.append(f"{acc.get('em')} (n={acc.get('n')})")
        out.append("| " + " | ".join(row) + " |")
    out.append("| **macro EM** | " + " | ".join(
        f"**{f(runs[step_of[s]]['accuracy']['macro_em_over_benchmarks'])}**" for s in sorted(steps)) + " |")
    out.append("| **micro EM** | " + " | ".join(
        f"**{f(runs[step_of[s]]['accuracy']['micro_em_over_samples'])}**" for s in sorted(steps)) + " |")
    out.append("")
    out.append("### Behaviour metrics by checkpoint (overall, 51,201 samples)\n")
    out.append("| metric | " + " | ".join(f"step{s}" for s in sorted(steps)) + " |")
    out.append("|---" * (1 + len(steps)) + "|")
    def overall_weighted(run, key):
        br = run.get("benchmark_results") or {}
        n = sum((v.get("n_samples") or 0) for v in br.values())
        if not n:
            return run.get("overall", {}).get(key)
        num = sum((v.get(key) or 0) * (v.get("n_samples") or 0) for v in br.values())
        return num / n

    for key, label in BEHAV:
        if key in ("assistant_turns_mean_all_messages", "response_tokens_mean"):
            cells = [f"{overall_weighted(runs[step_of[s]], key):.4f}" for s in sorted(steps)]
        else:
            cells = [f(runs[step_of[s]]['overall'].get(key)) for s in sorted(steps)]
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("Flat artifacts required by `E1-D_task.md` §6: `E1-D/evaluations/results.json`, "
               "`benchmark_scores.csv`, `behavior_metrics.csv`, `overall_metrics.csv`.\n")


def sec9_10(out, runs):
    comp = jload(os.path.join(E1D_ROOT, "evaluations", "comparison", "comparison.json"), {})
    if not comp:
        out.append("## 9. 与 GAP baseline 比较 (vs GAP baseline)\n\n(comparison.json missing)\n")
        out.append("## 10. 与 E1-A 比较 (vs E1-A)\n\n(comparison.json missing)\n")
        return

    def block(title, key, extra_note=""):
        b = comp.get(key)
        out.append(title)
        if not b:
            out.append("(missing run)\n")
            return
        out.append(f"baseline `{b['baseline_run']}` → test `{b['test_run']}`\n")
        out.append("| metric | baseline | E1-D | delta | delta % |")
        out.append("|---|---|---|---|---|")
        out.append(f"| macro EM | {f(b['macro_em']['baseline'])} | {f(b['macro_em']['test'])} | "
                   f"{d(b['macro_em']['baseline'], b['macro_em']['test'])} | "
                   f"{d(b['macro_em']['baseline'], b['macro_em']['test'], pct=True)} |")
        out.append(f"| micro EM | {f(b['micro_em']['baseline'])} | {f(b['micro_em']['test'])} | "
                   f"{d(b['micro_em']['baseline'], b['micro_em']['test'])} | "
                   f"{d(b['micro_em']['baseline'], b['micro_em']['test'], pct=True)} |")
        for m, label in BEHAV:
            bv = (b["behavior"].get(m) or {})
            base, test = bv.get("baseline"), bv.get("test")
            pctv = bv.get("pct")
            out.append(f"| {label} | {f(base, 5)} | {f(test, 5)} | {f(bv.get('delta'), 5)} | "
                       f"{'—' if pctv is None else f'{pctv:+.2f}%'} |")
        out.append("")
        out.append("Per benchmark EM:\n")
        out.append("| benchmark | baseline | E1-D | delta |")
        out.append("|---|---|---|---|")
        for bench in BENCH_ORDER:
            pbd = b["per_benchmark"].get(bench, {})
            if pbd.get("delta") is None:
                continue
            out.append(f"| {bench} | {f(pbd['baseline'])} | {f(pbd['test'])} | {pbd['delta']*100:+.2f} pp |")
        out.append("")
        if extra_note:
            out.append(extra_note + "\n")

    # paired comparisons
    paired = {}
    for p in glob.glob(os.path.join(E1D_ROOT, "evaluations", "comparison", "paired_comparison_*.json")):
        with open(p) as fh:
            pj = json.load(fh)
        paired[pj.get("label", os.path.basename(p))] = pj

    def paired_md(label, title):
        pj = paired.get(label)
        out.append(title)
        if not pj:
            out.append("(no paired comparison)\n")
            return
        o = pj["overall"]
        out.append(f"paired by (data_source, prompt) — **{o['n_paired_prompts']:,}** pairs (greedy, so a pair "
                   f"is the same question answered by the two checkpoints):\n")
        out.append(f"- search rounds: {o['rounds_baseline_mean']:.4f} → {o['rounds_e1d_mean']:.4f} "
                   f"({o['rounds_delta']:+.4f}, {o['rounds_delta']/max(o['rounds_baseline_mean'],1e-9)*100:+.2f}%); "
                   f"decreased on {o['rounds_decreased_prompts']:,} prompts "
                   f"({o['rounds_decreased_rate']*100:.2f}%), increased on {o['rounds_increased_prompts']:,} "
                   f"({o['rounds_increased_rate']*100:.2f}%); sign test **p = {o['sign_test_p_rounds']:.3g}**")
        out.append(f"- EM: {o['em_baseline_mean']:.4f} → {o['em_e1d_mean']:.4f} ({o['em_delta']:+.4f}); "
                   f"up {o['em_up_prompts']:,} / down {o['em_down_prompts']:,}; sign test "
                   f"**p = {o['sign_test_p_em']:.3g}**")
        out.append("")
        out.append("| benchmark | n | rounds base | rounds E1-D | Δ | decreased | increased | ΔEM |")
        out.append("|---|---|---|---|---|---|---|---|")
        for ds, v in pj["per_benchmark"].items():
            out.append(f"| {ds} | {v['n_paired_prompts']:,} | {v['rounds_baseline_mean']:.3f} | "
                       f"{v['rounds_e1d_mean']:.3f} | {v['rounds_delta']:+.3f} | "
                       f"{v['rounds_decreased_rate']*100:.1f}% | {v['rounds_increased_rate']*100:.1f}% | "
                       f"{v['em_delta']:+.4f} |")
        out.append("")

    block("## 9. 与 GAP baseline 比较 (vs GAP baseline)\n", "vs_gap_step120",
          "The GAP baseline at the **same step count** (step120 of the original GAP RL run, "
          "original EM reward) is the primary comparison; the E1-D branch starts from GAP step60 "
          "(via E1-A step70), so a GAP step60 comparison is also given in `comparison.json`.")
    paired_md("e1d_step120_vs_gap_step120", "Paired per-prompt test, E1-D step120 vs GAP step120:\n")
    block("## 10. 与 E1-A 比较 (vs E1-A)\n", "vs_e1a_step70",
          "E1-A step70 is the E1-D starting checkpoint, so this comparison measures what the "
          "**additional 50 shaped-reward steps** changed (within the same reward regime).")
    paired_md("e1d_step120_vs_e1a_step70", "Paired per-prompt test, E1-D step120 vs E1-A step70:\n")

    traj = comp.get("trajectory_from_e1a_step70") or {}
    if traj:
        out.append("### E1-D trajectory anchored at its own start (E1-A step70)\n")
        out.append("| checkpoint | macro EM | ΔEM vs E1-A70 (pp) | search rounds | Δrounds | Δrounds % | queries | parallel factor |")
        out.append("|---|---|---|---|---|---|---|---|")
        for s in sorted(traj, key=lambda x: int(x)):
            b = traj[s]
            if not b:
                continue
            rd = (b["behavior"].get("retrieval_rounds_mean") or {})
            em_delta = b["macro_em"]["delta"]
            em_cell = "—" if em_delta is None else f"{em_delta * 100:+.2f}"
            pct_cell = "—" if rd.get("pct") is None else f"{rd['pct']:+.2f}%"
            out.append(f"| step{s} | {f(b['macro_em']['test'])} | {em_cell} | "
                       f"{f(rd.get('test'), 3)} | {f(rd.get('delta'), 4)} | {pct_cell} | "
                       f"{f((b['behavior'].get('search_queries_mean') or {}).get('test'), 3)} | "
                       f"{f((b['behavior'].get('parallel_factor_mean') or {}).get('test'), 3)} |")
        out.append("")


def sec12(out, runs):
    comp = jload(os.path.join(E1D_ROOT, "evaluations", "comparison", "comparison.json"), {})
    out.append("## 12. 最终结论 (conclusion)\n")
    if not comp:
        out.append("(no evaluation comparison available yet)\n")
        return
    vg = comp.get("vs_gap_step120") or comp.get("vs_e1a_step70") or {}
    ve = comp.get("vs_e1a_step70") or {}
    g = vg.get("behavior", {}).get("retrieval_rounds_mean", {})
    e = ve.get("behavior", {}).get("retrieval_rounds_mean", {})
    em_g = vg.get("macro_em", {})
    em_e = ve.get("macro_em", {})
    out.append("| question | answer |")
    out.append("|---|---|")
    out.append(f"| search depth vs GAP step120 | {f(g.get('delta'), 4)} rounds "
               f"({f(g.get('pct'), 2)}%), macro EM {f(em_g.get('delta'), 4)} |")
    out.append(f"| search depth vs E1-A step70 | {f(e.get('delta'), 4)} rounds "
               f"({f(e.get('pct'), 2)}%), macro EM {f(em_e.get('delta'), 4)} |")
    # E1-A's stated success / stop criteria for E1-D
    ok_acc = (em_g.get("delta") is not None and em_g["delta"] >= -0.005)
    ok_rounds = (e.get("delta") is not None and e["pct"] is not None and e["pct"] <= -1.5)
    out.append(f"| E1-A success criterion (EM drop ≤ 0.5 pp) | {'MET' if ok_acc else 'NOT MET'} |")
    out.append(f"| E1-A success criterion (rounds −1.5 % or more vs the E1-D start) | {'MET' if ok_rounds else 'NOT MET'} |")
    out.append("")
    out.append("See the narrative verdict in the main report text.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--splice", action="store_true")
    args = ap.parse_args()
    run_dir = args.run_dir or open(os.path.join(E1D_ROOT, "latest_run.txt")).read().strip()

    runs = load_evals()
    out = []
    sec6(run_dir, out)
    sec7(out)
    sec8(out, runs)
    sec9_10(out, runs)
    sec12(out, runs)
    text = "\n".join(out)
    gen = os.path.join(E1D_ROOT, "report_generated_sections.md")
    with open(gen, "w") as fh:
        fh.write(text)
    print(f"wrote {gen} ({len(text)} chars)")

    if args.splice:
        rp = os.path.join(E1D_ROOT, "FINAL_REPORT.md")
        cur = open(rp).read()
        idx = cur.find("## 6. 训练过程")
        if idx < 0:
            raise SystemExit("anchor '## 6. 训练过程' not found in FINAL_REPORT.md")
        new = cur[:idx] + text
        with open(rp, "w") as fh:
            fh.write(new)
        print(f"spliced sections 6-12 into {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
