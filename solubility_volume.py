#!/usr/bin/env python3
"""Estimate water volume from saved molar logS predictions and medicine doses."""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MEDICINES = (
    ("A-1694", "Paracetamol", "paracetamol_mg"),
    ("E-1274", "Aspirina", "aspirin_mg"),
    ("I-19", "Clorfenamina", "chlorphenamine_mg"),
)


def load_csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return {row["ID"]: row for row in csv.DictReader(file)}


def predict_log_s(model, row):
    preprocessing = model["preprocessing"]
    feature_names = preprocessing["feature_names"]
    means = preprocessing["x_mean"]
    stds = preprocessing["x_std"]
    if not (len(feature_names) == len(means) == len(stds)):
        raise ValueError("Inconsistencia en los datos de normalización del modelo.")

    values = []
    for name, mean, std in zip(feature_names, means, stds):
        value = float(row[name])
        values.append((value - mean) / std if std else 0.0)

    for layer in model["layers"]:
        next_values = []
        if len(layer["weights"]) != len(layer["biases"]):
            raise ValueError("Inconsistencia entre pesos y sesgos del modelo.")
        for weights, bias in zip(layer["weights"], layer["biases"]):
            if len(weights) != len(values):
                raise ValueError("Dimensiones de pesos incompatibles con la entrada de la capa.")
            z = float(bias) + sum(float(weight) * value for weight, value in zip(weights, values))
            activation = layer["activation"]
            if activation == "tanh":
                next_values.append(math.tanh(z))
            elif activation == "linear":
                next_values.append(z)
            else:
                raise ValueError(f"Activación no soportada: {activation}")
        values = next_values

    if len(values) != 1:
        raise ValueError(f"Se esperaba una salida logS y se obtuvieron {len(values)}.")
    return values[0] * preprocessing["y_std"] + preprocessing["y_mean"]


def volume_ml(dose_mg, molecular_weight, log_s):
    if dose_mg <= 0 or molecular_weight <= 0:
        raise ValueError("La dosis y el peso molecular deben ser positivos.")
    # V(mL) = dose(mg) / (MW(g/mol) * S(mol/L)); logS = log10(S).
    log_volume = math.log10(dose_mg / molecular_weight) - log_s
    try:
        return 10.0**log_volume
    except OverflowError as error:
        raise ValueError("El volumen calculado excede el rango numérico.") from error


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "best_runs.json")
    parser.add_argument("--data", type=Path, default=ROOT / "data_curated.csv")
    parser.add_argument("--paracetamol-mg", type=float, default=500.0)
    parser.add_argument("--aspirin-mg", type=float, default=10.0)
    parser.add_argument("--chlorphenamine-mg", type=float, default=4.0)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"No existe el archivo de corridas: {args.model}")
    if not args.data.is_file():
        parser.error(f"No existe el dataset: {args.data}")
    if any(value <= 0 for value in (args.paracetamol_mg, args.aspirin_mg, args.chlorphenamine_mg)):
        parser.error("Las dosis deben ser positivas.")
    return args


def main():
    args = parse_args()
    archive = json.loads(args.model.read_text(encoding="utf-8"))
    if not archive.get("runs"):
        raise ValueError("El archivo de modelos no contiene corridas.")
    model = min(archive["runs"], key=lambda run: run["metrics"]["validation"]["rmse"])
    rows = load_csv_rows(args.data)

    doses = {
        "paracetamol_mg": args.paracetamol_mg,
        "aspirin_mg": args.aspirin_mg,
        "chlorphenamine_mg": args.chlorphenamine_mg,
    }
    results = []
    for medicine_id, name, dose_key in DEFAULT_MEDICINES:
        if medicine_id not in rows:
            raise ValueError(f"No se encontró {medicine_id} en {args.data}.")
        row = rows[medicine_id]
        dose = doses[dose_key]
        log_s = predict_log_s(model, row)
        solubility_mol_l = 10.0**log_s
        mw = float(row["MolWt"])
        results.append(
            {
                "name": name,
                "id": medicine_id,
                "dose_mg": dose,
                "molecular_weight_g_mol": mw,
                "predicted_logS": log_s,
                "predicted_solubility_mol_l": solubility_mol_l,
                "estimated_volume_ml": volume_ml(dose, mw, log_s),
            }
        )

    # For one shared solution, every compound must remain below its own saturation limit.
    shared_volume_ml = max(result["estimated_volume_ml"] for result in results)
    print(f"Mejor corrida: semilla {model['seed']} (RMSE validación "
          f"{model['metrics']['validation']['rmse']:.4f})")
    print("Estimación en agua pura a saturación:")
    for result in results:
        print(
            f"- {result['name']} ({result['id']}): dosis={result['dose_mg']:g} mg, "
            f"MW={result['molecular_weight_g_mol']:.3f} g/mol, "
            f"logS={result['predicted_logS']:.4f}, "
            f"S={result['predicted_solubility_mol_l']:.6g} mol/L, "
            f"agua estimada={result['estimated_volume_ml']:.3f} mL"
        )
    print(
        f"Volumen común teórico mínimo para las tres dosis: "
        f"{shared_volume_ml:.3f} mL (máximo de los volúmenes individuales)."
    )
    print(
        "Aviso: estimación basada en logS del modelo para las formas químicas del dataset; "
        "no considera pH, temperatura, sales, excipientes ni interacciones entre solutos."
    )


if __name__ == "__main__":
    main()
