using System.Globalization;
using System.Text.Json;
using LibNeuronalNetwork;

namespace ProyectoQuimixa;

class Program
{
    private sealed record MetricResult(double rmse, double mae, double mse);

    private static readonly string[] FeatureNames =
    {
        "MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHAcceptors",
        "NumHDonors", "NumHeteroatoms", "NumRotatableBonds", "NumValenceElectrons",
        "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings", "RingCount",
        "TPSA", "LabuteASA", "BalabanJ", "BertzCT"
    };

    static void Main(string[] args)
    {
        var options = ParseOptions(args);
        string dataPath = options.GetValueOrDefault("data", "./data_curated.csv");
        int seed = ParseInt(options, "seed", 1);
        int splitSeed = ParseInt(options, "split-seed", 42);
        int epochs = ParseInt(options, "epochs", 200);
        int patience = ParseInt(options, "patience", 25);
        int hiddenSize = ParseInt(options, "hidden-size", 24);
        double learningRate = ParseDouble(options, "learning-rate", 0.005);
        string[] testIds = options.GetValueOrDefault("test-ids", "I-19,A-1694,E-1274")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        if (testIds.Length == 0) throw new ArgumentException("Debe indicarse al menos un ID para el conjunto de prueba.");
        if (epochs <= 0 || patience <= 0 || hiddenSize <= 0 || learningRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(args), "Épocas, paciencia, tamaño oculto y tasa de aprendizaje deben ser positivos.");
        }

        var (foundTestIds, x, y, xReservedTest, yReservedTest) = Data.Extraer(dataPath, testIds);
        var missingTestIds = testIds.Except(foundTestIds, StringComparer.Ordinal).ToArray();
        if (missingTestIds.Length > 0)
        {
            throw new InvalidDataException($"No se encontraron los IDs de prueba: {string.Join(", ", missingTestIds)}");
        }
        if (x.Length < 10) throw new InvalidDataException("Se requieren al menos 10 filas para separar entrenamiento, validación y prueba.");

        var indexes = Enumerable.Range(0, x.Length).ToArray();
        Barajar(indexes, new Random(splitSeed));
        int trainCount = (int)(indexes.Length * 0.8);
        int validationCount = (int)(indexes.Length * 0.1);
        int[] trainIndexes = indexes[..trainCount];
        int[] validationIndexes = indexes[trainCount..(trainCount + validationCount)];
        int[] testIndexes = indexes[(trainCount + validationCount)..];

        var xTrain = trainIndexes.Select(i => x[i]).ToArray();
        var yTrain = trainIndexes.Select(i => y[i]).ToArray();
        var (xMean, xStd) = Data.Zscore(xTrain);
        Data.AplicarZscore(xTrain, xMean, xStd);
        var xValidation = validationIndexes.Select(i => x[i]).ToArray();
        Data.AplicarZscore(xValidation, xMean, xStd);
        var xRandomTest = testIndexes.Select(i => x[i]).ToArray();
        Data.AplicarZscore(xRandomTest, xMean, xStd);
        var xTest = xRandomTest.Concat(xReservedTest).ToArray();
        Data.AplicarZscore(xReservedTest, xMean, xStd);
        var yTest = testIndexes.Select(i => y[i]).Concat(yReservedTest).ToArray();

        double yMean = yTrain.Average();
        double yStd = Math.Sqrt(yTrain.Average(value => (value - yMean) * (value - yMean)));
        if (yStd == 0) throw new InvalidDataException("La variable objetivo no tiene variación.");
        var yTrainScaled = yTrain.Select(value => (value - yMean) / yStd).ToArray();

        var network = new NeuralNetwork(learningRate)
            .Add(
                new Layer(FeatureNames.Length, hiddenSize, Neuron.Tanh, seed),
                new Layer(hiddenSize, 1, Neuron.Linear, seed + 1));

        double bestValidationRmse = double.PositiveInfinity;
        int bestEpoch = 0;
        int epochsWithoutImprovement = 0;
        double[][][]? bestWeights = null;
        double[][]? bestBiases = null;
        var trainOrder = Enumerable.Range(0, xTrain.Length).ToArray();
        var trainingRandom = new Random(seed + 2);

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            Barajar(trainOrder, trainingRandom);
            foreach (int i in trainOrder)
            {
                network.Train(xTrain[i], new[] { yTrainScaled[i] });
            }

            double validationRmse = Evaluate(network, xValidation, validationIndexes.Select(i => y[i]).ToArray(), yMean, yStd);
            if (validationRmse < bestValidationRmse)
            {
                bestValidationRmse = validationRmse;
                bestEpoch = epoch;
                epochsWithoutImprovement = 0;
                bestWeights = network.Layers.Select(layer => layer.Weights.Select(row => row.ToArray()).ToArray()).ToArray();
                bestBiases = network.Layers.Select(layer => layer.Biases.ToArray()).ToArray();
            }
            else if (++epochsWithoutImprovement >= patience)
            {
                break;
            }
        }

        if (bestWeights is null || bestBiases is null) throw new InvalidOperationException("No se pudo conservar un modelo entrenado.");
        for (int layerIndex = 0; layerIndex < network.Layers.Count; layerIndex++)
        {
            for (int neuron = 0; neuron < network.Layers[layerIndex].Biases.Length; neuron++)
            {
                network.Layers[layerIndex].Biases[neuron] = bestBiases[layerIndex][neuron];
                for (int input = 0; input < network.Layers[layerIndex].Weights[neuron].Length; input++)
                {
                    network.Layers[layerIndex].Weights[neuron][input] = bestWeights[layerIndex][neuron][input];
                }
            }
        }

        var result = new
        {
            seed,
            best_epoch = bestEpoch,
            metrics = new
            {
                train = Metrics(network, xTrain, yTrain, yMean, yStd),
                validation = Metrics(network, xValidation, validationIndexes.Select(i => y[i]).ToArray(), yMean, yStd),
                test = Metrics(network, xTest, yTest, yMean, yStd),
                reserved_medicines = foundTestIds.Select((id, i) => new
                {
                    id,
                    actual = yReservedTest[i],
                    predicted = Predict(network, xReservedTest[i], yMean, yStd),
                    absolute_error = Math.Abs(yReservedTest[i] - Predict(network, xReservedTest[i], yMean, yStd))
                }).ToArray()
            },
            preprocessing = new
            {
                feature_names = FeatureNames,
                x_mean = xMean,
                x_std = xStd,
                y_mean = yMean,
                y_std = yStd
            },
            layers = network.Layers.Select((layer, index) => new
            {
                activation = index == network.Layers.Count - 1 ? "linear" : "tanh",
                weights = layer.Weights,
                biases = layer.Biases
            }).ToArray()
        };
        Console.WriteLine(JsonSerializer.Serialize(result));
    }

    private static double Evaluate(NeuralNetwork network, double[][] x, double[] y, double yMean, double yStd)
    {
        return Metrics(network, x, y, yMean, yStd).rmse;
    }

    private static MetricResult Metrics(NeuralNetwork network, double[][] x, double[] y, double yMean, double yStd)
    {
        double squaredError = 0;
        double absoluteError = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double prediction = Predict(network, x[i], yMean, yStd);
            double error = prediction - y[i];
            squaredError += error * error;
            absoluteError += Math.Abs(error);
        }
        double mse = squaredError / x.Length;
        return new MetricResult(Math.Sqrt(mse), absoluteError / x.Length, mse);
    }

    private static double Predict(NeuralNetwork network, double[] x, double yMean, double yStd) =>
        network.ForWard(x)[0] * yStd + yMean;

    private static Dictionary<string, string> ParseOptions(string[] args)
    {
        var options = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < args.Length; i++)
        {
            if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
            {
                throw new ArgumentException($"Argumento no válido: {args[i]}");
            }
            options[args[i][2..]] = args[++i];
        }
        return options;
    }

    private static int ParseInt(Dictionary<string, string> options, string name, int defaultValue) =>
        options.TryGetValue(name, out string? value) ? int.Parse(value, CultureInfo.InvariantCulture) : defaultValue;

    private static double ParseDouble(Dictionary<string, string> options, string name, double defaultValue) =>
        options.TryGetValue(name, out string? value) ? double.Parse(value, CultureInfo.InvariantCulture) : defaultValue;

    private static void Barajar(int[] array, Random rng)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
