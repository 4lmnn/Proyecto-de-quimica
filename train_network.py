#!/usr/bin/env python3
"""Entrena varias inicializaciones de la red C# y archiva los mejores modelos."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / "src" / "99.4.csproj"
ASSEMBLY = ROOT / "src" / "bin" / "Release" / "net10.0" / "99.4.dll"
DOTNET = Path("/usr/lib/dotnet/dotnet")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data_curated.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "best_runs.json")
    parser.add_argument("--runs", type=int, default=5, help="Cantidad de inicializaciones.")
    parser.add_argument("--top", type=int, default=3, help="Cantidad de mejores modelos a guardar.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1, help="Semilla inicial; se incrementa por corrida.")
    parser.add_argument("--split-seed", type=int, default=42, help="Semilla fija para separar los datos.")
    parser.add_argument(
        "--test-ids",
        default="I-19,A-1694,E-1274",
        help="IDs separados por coma que se reservan exclusivamente para prueba.",
    )
    args = parser.parse_args()
    if args.runs < 1 or args.top < 1 or args.top > args.runs:
        parser.error("--runs debe ser positivo y --top debe estar entre 1 y --runs.")
    if args.epochs < 1 or args.patience < 1 or args.hidden_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, patience, hidden-size y learning-rate deben ser positivos.")
    if not args.data.is_file():
        parser.error(f"No existe el dataset: {args.data}")
    if not args.test_ids.strip():
        parser.error("--test-ids debe incluir al menos un ID.")
    return args


def main():
    args = parse_args()
    subprocess.run(
        [str(DOTNET), "build", str(PROJECT), "--configuration", "Release", "--nologo"],
        cwd=ROOT,
        check=True,
    )

    runs = []
    for seed in range(args.seed, args.seed + args.runs):
        command = [
            str(DOTNET), str(ASSEMBLY),
            "--data", str(args.data.resolve()),
            "--seed", str(seed),
            "--split-seed", str(args.split_seed),
            "--test-ids", args.test_ids,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--hidden-size", str(args.hidden_size),
            "--learning-rate", str(args.learning_rate),
        ]
        completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
        runs.append(json.loads(completed.stdout))
        print(
            f"semilla={seed} época={runs[-1]['best_epoch']} "
            f"RMSE_validación={runs[-1]['metrics']['validation']['rmse']:.6f}",
            file=sys.stderr,
        )

    best_runs = sorted(runs, key=lambda run: run["metrics"]["validation"]["rmse"])[:args.top]
    archive = {
        "target": "Solubility",
        "selection_metric": "validation.rmse",
        "dataset": str(args.data.resolve()),
        "split_seed": args.split_seed,
        "reserved_test_ids": [item.strip() for item in args.test_ids.split(",") if item.strip()],
        "architecture": [17, args.hidden_size, 1],
        "saved_runs": len(best_runs),
        "runs": best_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(archive, indent=2) + "\n", encoding="utf-8")
    print(f"Guardadas {len(best_runs)} mejores corridas en {args.output.resolve()}")


if __name__ == "__main__":
    main()
