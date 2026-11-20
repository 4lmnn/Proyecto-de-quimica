# Red neuronal para predecir solubilidad

Este proyecto entrena una red neuronal implementada en C# para predecir la
columna `Solubility` de `data_curated.csv`. El script `train_network.py`
compila y ejecuta el entrenador C#, repite el entrenamiento con distintas
semillas y guarda las mejores corridas en JSON.

## Requisitos

- Python 3.8 o posterior. El script usa solo la biblioteca estándar de Python.
- .NET SDK 10 instalado en `/usr/lib/dotnet/dotnet` (la ruta configurada en el
  script).
- El archivo `data_curated.csv` en la raíz del repositorio, o una ruta
  alternativa indicada con `--data`.

No es necesario instalar paquetes de Python. El script compila el proyecto
automáticamente antes de entrenar.

## Uso rápido

Desde la raíz del repositorio:

```bash
python3 train_network.py
```

La ejecución predeterminada:

- Entrena cinco inicializaciones, con semillas de 1 a 5.
- Guarda las tres mejores, ordenadas por RMSE de validación.
- Escribe pesos, sesgos, parámetros de normalización y métricas en
  `best_runs.json`.
- Reserva por defecto los IDs `I-19` (clorfenamina), `A-1694` (paracetamol) y
  `E-1274` (aspirina) exclusivamente para prueba; no se usan al ajustar los
  pesos ni al seleccionar la época.

Para consultar todas las opciones:

```bash
python3 train_network.py --help
```

## Opciones

| Opción | Predeterminado | Descripción |
|---|---:|---|
| `--data` | `data_curated.csv` | Ruta al CSV de entrada. |
| `--output` | `best_runs.json` | Archivo JSON de salida. |
| `--runs` | `5` | Cantidad de semillas/conjuntos de pesos que se entrenan. |
| `--top` | `3` | Cantidad de mejores corridas que se guardan (`1` a `runs`). |
| `--epochs` | `200` | Máximo de épocas por corrida. |
| `--patience` | `25` | Épocas sin mejora de validación antes de detenerse. |
| `--hidden-size` | `24` | Neuronas de la capa oculta. |
| `--learning-rate` | `0.005` | Tasa de aprendizaje. |
| `--seed` | `1` | Primera semilla de inicialización; las siguientes son consecutivas. |
| `--split-seed` | `42` | Semilla que fija la partición aleatoria de los datos restantes. |
| `--test-ids` | `I-19,A-1694,E-1274` | IDs separados por coma reservados para prueba. |

Ejemplo con más inicializaciones y otro archivo de salida:

```bash
python3 train_network.py \
  --runs 20 \
  --top 5 \
  --epochs 400 \
  --patience 40 \
  --output resultados/solubilidad.json
```

Para usar otros IDs como prueba, pásalos separados por comas. Si se desea
desactivar la reserva especial de medicamentos, se puede proporcionar otro
ID o conjunto de IDs que sí existan en el CSV; todos los IDs especificados
deben estar presentes.

## Repetir una corrida guardada

Cada corrida del JSON incluye su semilla. Para volver a entrenar la mejor
corrida actualmente guardada (semilla `2`) con los valores predeterminados:

```bash
python3 train_network.py \
  --runs 1 \
  --top 1 \
  --seed 2 \
  --split-seed 42 \
  --output corrida_repetida.json
```

El comando usa también los IDs de prueba predeterminados. Para reproducir otra
entrada, sustituye `2` por su valor de `seed`. La repetición exacta requiere
el mismo dataset, el mismo SDK y los mismos hiperparámetros. El JSON guarda la
semilla y la partición, pero no todos los hiperparámetros; para corridas con
opciones personalizadas, hay que volver a proporcionar los mismos valores
`--epochs`, `--patience`, `--hidden-size`, `--learning-rate` y `--test-ids`.

Este comando vuelve a entrenar desde una inicialización aleatoria reproducible.
Para reproducir predicciones sin volver a entrenar, utiliza directamente los
pesos y sesgos guardados en la entrada del modelo.

## Partición y evaluación

Los IDs indicados en `--test-ids` se separan antes de dividir el resto de las
filas. El resto se baraja de forma reproducible usando `--split-seed` y se
divide aproximadamente en 80 % entrenamiento, 10 % validación y 10 % prueba.
Las estadísticas de normalización se calculan solo con el conjunto de
entrenamiento.

La mejor época se elige por RMSE de validación. El archivo presenta RMSE, MAE y
MSE para entrenamiento, validación y prueba. `metrics.reserved_medicines`
incluye el valor real, la predicción y el error absoluto de cada ID reservado.
El RMSE global de prueba incluye tanto la fracción aleatoria de prueba como los
IDs reservados.

La arquitectura predeterminada es `17 → 24 → 1`: 17 descriptores moleculares,
una capa oculta `tanh` y una salida lineal. La salida se transforma de vuelta
a las unidades originales de `Solubility` antes de calcular y registrar los
errores.
