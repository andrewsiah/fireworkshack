"""Standalone reports that keep unmeasured results distinct from zero scores."""

import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wilson(successes, count):
    if not count:
        return None
    z = 1.96
    p = successes / count
    scale = 1 + z * z / count
    center = (p + z * z / (2 * count)) / scale
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / scale
    return max(0, center - half), min(1, center + half)


def build_report(state, output_dir: Path, *, evaluations=None, readiness=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = evaluations or []
    for evaluation in evaluations:
        if evaluation.get("source") != "measured":
            raise ValueError(
                "report source must be measured; external or synthetic scores require separate reports"
            )
    account = state.summary()
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source": "measured" if evaluations else "readiness",
        "accounting": account,
        "readiness": readiness or {},
        "arms": {arm: "not run" for arm in ("A", "B", "C")},
        "metrics": [],
        "coverage": [],
    }
    for evaluation in evaluations:
        arm = evaluation["arm"]
        scores = evaluation.get("scores", [])
        report["arms"][arm] = "measured" if scores else "not run"
        report["coverage"].append(
            {
                "arm": arm,
                "total": evaluation.get("total", len(scores)),
                "completed": evaluation.get(
                    "completed", sum(s.get("status") == "completed" for s in scores)
                ),
                "missing": evaluation.get("missing", 0),
                "errors": evaluation.get("errors", 0),
                "pending": evaluation.get("pending", 0),
            }
        )
        for suite in sorted(
            {s["suite"] for s in scores} | set(evaluation.get("by_suite", {}))
        ):
            suite_scores = [s for s in scores if s["suite"] == suite]
            total = (
                evaluation.get("by_suite", {})
                .get(suite, {})
                .get("total", len(suite_scores))
            )
            for metric, field in (
                ("capability", "capability_correct"),
                ("refusal", "refusal"),
                ("harmful_compliance", "harmful_compliance"),
                ("complete", "answer_complete"),
                ("truncated", "truncated"),
            ):
                if metric == "capability" and suite != "mmlu":
                    continue
                known = [
                    s[field] for s in suite_scores if isinstance(s.get(field), bool)
                ]
                count, good = len(known), sum(known)
                denominator = (
                    total
                    if metric in ("capability", "complete", "truncated")
                    else count
                )
                report["metrics"].append(
                    {
                        "arm": arm,
                        "suite": suite,
                        "metric": metric,
                        "successes": good,
                        "scored": count,
                        "total": total,
                        "unknown": total - count,
                        "rate_denominator": denominator,
                        "rate": good / denominator if denominator else None,
                        "interval": wilson(good, denominator),
                    }
                )
    paths = {
        key: output_dir / name
        for key, name in (
            ("markdown", "report.md"),
            ("json", "summary.json"),
            ("csv", "metrics.csv"),
            ("chart", "overview.png"),
        )
    }
    paths["json"].write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "arm",
        "suite",
        "metric",
        "successes",
        "scored",
        "total",
        "unknown",
        "rate_denominator",
        "rate",
        "interval",
    ]
    with paths["csv"].open("w") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(report["metrics"])
    lines = [
        "# FireworksHack experiment report",
        "",
        "**Measured pilot results**"
        if evaluations
        else "**Readiness report — no model performance measured.**",
        "",
        "| Arm | Status |",
        "|---|---|",
    ]
    lines += [f"| {a} | {status} |" for a, status in report["arms"].items()]
    lines += [
        "",
        f"Settled usage estimate: ${account['spent_usd']}; unsettled reservations: ${account['reserved_usd']}.",
        "Reservations remain held until actual usage is known. Billing may settle later.",
        "",
        "## Verification",
        "",
    ]
    lines += [
        f"- {name}: "
        + (
            "verified"
            if value is True
            else "not run / unresolved"
            if value is False
            else "unverified"
        )
        for name, value in (readiness or {}).items()
    ]
    if report["metrics"]:
        lines += [
            "",
            "## Coverage",
            "",
            "| Arm | Planned | Completed | Missing | Error | Pending |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {c['arm']} | {c['total']} | {c['completed']} | {c['missing']} | {c['errors']} | {c['pending']} |"
            for c in report["coverage"]
        ]
        lines += [
            "",
            "## Measurements",
            "",
            "| Arm | Suite | Metric | Known / total | Rate (95% Wilson interval) |",
            "|---|---|---|---|---|",
        ]
        for row in report["metrics"]:
            interval = row["interval"]
            value = (
                "unscored"
                if interval is None
                else f"{row['rate']:.1%} ({interval[0]:.1%}–{interval[1]:.1%})"
            )
            lines.append(
                f"| {row['arm']} | {row['suite']} | {row['metric']} | {row['scored']}/{row['total']} | {value} |"
            )
        lines += [
            "",
            "Capability/completeness rates use all planned items; missing outputs count as incomplete/incorrect.",
            "Behavioral refusal/compliance rates use judged items only; unjudged coverage is shown separately.",
        ]
    lines += [
        "",
        "Behavioral compliance, capability and completeness are separate measurements.",
        "This pilot cannot establish statistical superiority. External author claims are not measured results.",
        "",
        "![Experiment status](overview.png)",
        "",
    ]
    paths["markdown"].write_text("\n".join(lines))
    _chart(report, paths["chart"])
    return paths


def _chart(report, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5.2),
        facecolor="#f4f5f7",
        gridspec_kw={"width_ratios": [1.6, 1]},
    )
    fig.suptitle(
        "FireworksHack · "
        + (
            "Pilot measurements"
            if report["source"] == "measured"
            else "Verified preparation"
        ),
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color="#12263a",
    )
    ax = axes[0]
    metrics = [m for m in report["metrics"] if m["rate"] is not None]
    if metrics:
        labels = [
            f"{m['arm']} · {m['suite']} · {m['metric']} ({m['scored']}/{m['total']})"
            for m in metrics
        ]
        ax.barh(labels, [m["rate"] for m in metrics], color="#267d91")
        ax.set_xlim(0, 1)
        ax.set_title("Distinct metrics · known denominators")
    else:
        labels = list(report["readiness"]) or ["Model measurements"]
        values = list(report["readiness"].values()) or [False]
        for n, (label, value) in enumerate(zip(labels, values)):
            ax.barh(n, 1, color="#24796f" if value is True else "#d8dce1", height=0.65)
            ax.text(
                0.03,
                n,
                label,
                va="center",
                color="white" if value is True else "#23364a",
                fontsize=10,
            )
            ax.text(
                0.98,
                n,
                "VERIFIED" if value is True else "PENDING",
                ha="right",
                va="center",
                fontsize=8,
                color="white" if value is True else "#526476",
            )
        ax.set_yticks([])
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_title(
            "Readiness evidence · no performance scores", loc="left", fontsize=11
        )
    right = axes[1]
    right.axis("off")
    right.text(
        0.05,
        0.89,
        "A   Original DeepSeek\nB   Our Fireworks LoRA\nC   Refuseless comparison",
        transform=right.transAxes,
        fontsize=12,
        linespacing=2,
        color="#12263a",
    )
    right.text(
        0.05,
        0.47,
        "\n".join(f"{a}: {s.upper()}" for a, s in report["arms"].items()),
        transform=right.transAxes,
        fontsize=12,
        linespacing=1.7,
        fontweight="bold",
        color="#526476",
    )
    right.text(
        0.05,
        0.11,
        f"Settled usage estimate: ${report['accounting']['spent_usd']}\n"
        f"Unsettled: ${report['accounting']['reserved_usd']}",
        transform=right.transAxes,
        fontsize=11,
        color="#12263a",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f4f5f7")
    fig.text(
        0.07,
        0.035,
        "Preparation is verified independently from training and model quality.",
        color="#526476",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.04, 0.07, 0.98, 0.9))
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
