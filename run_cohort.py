#!/usr/bin/env python3
"""Run svr_gpu's public CLI over a local cohort and build Nilearn screenshots."""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def get_stacks(case: dict) -> list[str]:
    stacks = case.get("raw_stacks", case.get("stacks"))
    if not isinstance(stacks, list) or not stacks:
        raise ValueError(f"{case.get('case', '<unknown>')}: raw_stacks must be a nonempty list")
    missing = [str(path) for path in stacks if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input stack(s) for {case.get('case')}: {missing}")
    return [str(Path(path).resolve()) for path in stacks]


def write_report(out: Path, rows: list[dict]) -> None:
    report = out / "report"
    report.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        name = html.escape(row["case"])
        ref = row.get("reference_screenshot")
        reference = (f'<figure><img src="{html.escape(ref)}" alt="Reference for {name}">'
                     f'<figcaption>Reference</figcaption></figure>') if ref else ""
        screenshot = row.get("screenshot")
        output = (f'<figure><img src="{html.escape(screenshot)}" alt="SVR GPU result for {name}">'
                  f'<figcaption>SVR GPU</figcaption></figure>') if screenshot else "<p>No screenshot: reconstruction failed.</p>"
        cards.append(f'<section id="{name}"><h2>{name}</h2><p>Status: {html.escape(row["status"])}</p>'
                     f'<div class="images">{reference}{output}</div><p class="path">{html.escape(row.get("reconstruction", ""))}</p></section>')
    options = "".join(f'<option value="{html.escape(r["case"])}">{html.escape(r["case"])}</option>' for r in rows)
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SVR GPU cohort report</title><style>
body{{font:16px system-ui,sans-serif;margin:0;background:#eef2f7;color:#1a2940}}header{{background:#10233f;color:white;padding:28px 5vw}}main{{max-width:1200px;margin:auto;padding:20px}}nav{{position:sticky;top:0;background:white;padding:12px 5vw}}section{{background:white;padding:20px;margin:20px 0;border-radius:10px;scroll-margin-top:65px}}.images{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}}figure{{margin:0}}img{{width:100%;height:auto}}figcaption{{font-weight:600;padding:6px 0}}.path{{font-size:12px;color:#52647d;overflow-wrap:anywhere}}</style>
<header><h1>SVR GPU cohort report</h1><p>{len(rows)} cases · screenshots generated with Nilearn plot_anat</p></header>
<nav>Jump to case <select onchange="location.hash=this.value">{options}</select> · <a href="../results.csv">CSV results</a></nav>
<main>{''.join(cards)}</main></html>'''
    (report / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Local JSON list of case records")
    parser.add_argument("--svr-gpu-dir", type=Path, default=Path("../svr_gpu"), help="Path to the svr_gpu checkout")
    parser.add_argument("--output", required=True, type=Path, help="Directory for reconstructions and report")
    parser.add_argument("--python", dest="python", type=Path, help="Python interpreter used for svr_cli.py")
    parser.add_argument("--device", default="0", help="Device argument passed to svr_cli.py")
    parser.add_argument("--limit", type=int, help="Run only the first N manifest cases")
    parser.add_argument("--force", action="store_true", help="Rerun cases with existing outputs")
    args = parser.parse_args()

    # Import plotting dependencies only after argument parsing so --help works
    # even in a reconstruction-only environment.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import plotting

    manifest = args.manifest.resolve()
    cases = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("The manifest must contain a nonempty JSON list")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        cases = cases[:args.limit]

    svr_dir = args.svr_gpu_dir.resolve()
    cli = svr_dir / "svr_cli.py"
    if not cli.is_file():
        raise FileNotFoundError(f"svr_cli.py not found: {cli}")
    python = args.python.resolve() if args.python else (svr_dir / ".venv/bin/python")
    if not python.is_file():
        python = Path(sys.executable)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.csv"
    rows: list[dict] = []

    for index, case in enumerate(cases, start=1):
        name = str(case.get("case", f"case_{index:03d}"))
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Unsafe case label: {name!r}")
        stacks = get_stacks(case)
        case_dir = out / name
        case_dir.mkdir(parents=True, exist_ok=True)
        reconstruction = case_dir / "reconstruction.nii.gz"
        log_path = case_dir / "cli.log"
        command = [str(python), str(cli), "--input-stacks", *stacks,
                   "--output", str(reconstruction), "--device", str(args.device)]
        start = time.monotonic()
        if reconstruction.exists() and not args.force:
            return_code = 0
            status = "reused"
        else:
            environment = os.environ.copy()
            environment.setdefault("OMP_NUM_THREADS", "8")
            environment.setdefault("OPENBLAS_NUM_THREADS", "4")
            environment["SVR_TEMP_DIR"] = str(case_dir / "temp")
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=svr_dir, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            return_code = result.returncode
            status = "completed" if return_code == 0 and reconstruction.is_file() else "failed"
        elapsed = time.monotonic() - start

        screenshot_rel = None
        if status in {"completed", "reused"} and reconstruction.is_file():
            screenshot = out / "report" / f"{name}.png"
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            display = plotting.plot_anat(str(reconstruction), display_mode="ortho", dim=0,
                                        annotate=True, draw_cross=False,
                                        title=f"{name} · SVR GPU")
            display.savefig(str(screenshot), dpi=150)
            display.close()
            screenshot_rel = f"{name}.png"

        reference_screenshot_rel = None
        reference = case.get("reference")
        if reference:
            reference_path = Path(reference)
            if reference_path.is_file():
                reference_screenshot = out / "report" / f"{name}_reference.png"
                display = plotting.plot_anat(str(reference_path), display_mode="ortho", dim=0,
                                            annotate=True, draw_cross=False,
                                            title=f"{name} · reference (independent coordinates)")
                display.savefig(str(reference_screenshot), dpi=150)
                display.close()
                reference_screenshot_rel = f"{name}_reference.png"

        rows.append({"case": name, "status": status, "exit_code": return_code,
                     "elapsed_seconds": round(elapsed, 3), "command": shlex.join(command),
                     "reconstruction": str(reconstruction), "screenshot": screenshot_rel,
                     "reference_screenshot": reference_screenshot_rel,
                     "log": str(log_path) if log_path.exists() else ""})
        print(f"[{index}/{len(cases)}] {name}: {status} ({elapsed:.1f}s)", flush=True)
        with results_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_report(out, rows)

        if status == "failed":
            print(f"  CLI failed; see {log_path}", file=sys.stderr)
            return return_code or 1

    print(f"Report: {out / 'report/index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
