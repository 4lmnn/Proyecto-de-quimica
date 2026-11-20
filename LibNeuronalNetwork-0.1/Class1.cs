namespace LibNeuronalNetwork;

public enum Neuron {
    Sigmoide,
    ReLU,
    Tanh,
    Linear
}

public class NeuralNetwork {
    private List<Layer> layers = new List<Layer>();
    private double alfa;
    public IReadOnlyList<Layer> Layers => layers;

    // Alfa se asigna aqui porque es constante en todas las
    // capas de nuestra red
    public NeuralNetwork (double alfa) {
        this.alfa = alfa;
    }

    // Agrega nuevas Layer a nuestra lista
    public NeuralNetwork Add(params Layer[] newLayers) {
        // Verifica en cada capa que nuestras salidas y entradas en nuestra red sesan coherentes
        foreach(Layer layer in newLayers){
            if (layers.Count > 0) {     // Checa cuando hay mas de una capa
                Layer prevLayer = layers[layers.Count - 1];
                if (prevLayer.OutputSize != layer.InputSize) {
                    throw new ArgumentException($"Error de dimensiones: La capa anterior tiene {prevLayer.OutputSize}salidas y tu siguiente capa espera {layer.InputSize} entradas.");
                }
            }
            // Agrega nuestra capa cuando las entradas y salidas coinciden
            layers.Add(layer);
        }
        return this;
    }

    // Hace ForWard en cada capa de la red
    // Se puede leer como un pipeline donde nuestra primera capa 'x'
    // entra a esta funcion y regresa su 'yPred', despues de una pasada
    // usamos esa salida como nueva 'x' y esto se repite hasta neuestra
    // ultima capa, una vez termina retorna 'yPred' de la capa [^1]
    public double[] ForWard(double[] inputs) {
        if (layers.Count == 0) {
            throw new InvalidOperationException("La red no tiene capas.");
        }
        if (inputs.Length != layers[0].InputSize) {
            throw new ArgumentException($"La red espera {layers[0].InputSize} entradas y recibió {inputs.Length}.", nameof(inputs));
        }

        double[] current = inputs;
        foreach (var layer in layers)
        {
            current = layer.ForWard(current);
        }
        return current;
    }

    public void Train(double[] inputs, double[] yReal) {
        double[] yPred = ForWard(inputs);
        if (yReal.Length != yPred.Length) {
            throw new ArgumentException($"La red produce {yPred.Length} salidas y recibió {yReal.Length} valores objetivo.", nameof(yReal));
        }

        // Conseguimos nuestro error en nuestra ultima capa
        double[] error = new double[yReal.Length];
        for (int i = 0; i < yReal.Length; i++) {
            error[i] = yReal[i] - yPred[i];
        }

        // Recorre hacia atras capa a capa haciendo backpropagation
        for (int i = layers.Count - 1; i >= 0; i--) {
            error = layers[i].BackPropagation(error, this.alfa);
        }
    }
}
public class Layer {
    // Variables necesarias
    private double[][] weights;
    private double[] biases;
    private double[] outputs;
    private double[] sums;
    private double[]? inputs;
    // Solo lectura
    public double[][] Weights => weights;
    public double[]   Biases  => biases;
    public double[]   Outputs => outputs;

    public int InputSize  => weights[0].Length;
    public int OutputSize => outputs.Length;

    private Func<double, double> Activate;
    private Func<double, double, double> Derive;

    private readonly Random random;

    public Layer(int inputSize, int outputSize, Neuron neuron, int? seed = null) {
        if (inputSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize));
        if (outputSize <= 0) throw new ArgumentOutOfRangeException(nameof(outputSize));
        random = seed.HasValue ? new Random(seed.Value) : Random.Shared;

        weights = new double[outputSize][];
        biases  = new double[outputSize];
        outputs = new double[outputSize];
        sums    = new double[outputSize];

        double lim = 0.0;

        switch (neuron) {
            case Neuron.Sigmoide:
                Activate = z => 1.0 / (1.0 + Math.Exp(-z));
                Derive = (z, y) => y * (1.0 - y);
                lim = Math.Sqrt(6.0 / (inputSize + outputSize));
                break;
            case Neuron.ReLU:
                Activate = z => z > 0 ? z : 0.0;
                Derive = (z, y) => z > 0 ? 1.0 : 0.0;
                lim = Math.Sqrt(6.0 / inputSize);
                break;
            case Neuron.Tanh:
                Activate = z => Math.Tanh(z);
                Derive = (z, y) => 1.0 - (y * y);
                lim = Math.Sqrt(6.0 / (inputSize + outputSize));
                break;
            case Neuron.Linear:
                Activate = z => z;
                Derive = (z, y) => 1.0;
                lim = Math.Sqrt(2.0 / inputSize);
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(neuron));
        }

        // Inicializamos los pesos de cada neurona en nuestra capa de manera aleatoria
        for (int i = 0; i < outputSize; i++) {
            double[] wi = new double[inputSize];
            for (int j = 0; j < inputSize; j++) {
                wi[j] = (random.NextDouble() * 2.0 - 1.0) * lim;
            }
            weights[i] = wi;
            biases[i] = 0.0;
        }
    }

    // Propagacion hacia atras
    public double[] BackPropagation(double[] error, double alfa) {
        if (error.Length != outputs.Length) {
            throw new ArgumentException($"La capa produce {outputs.Length} salidas y recibió {error.Length} errores.", nameof(error));
        }
        if (inputs is null) {
            throw new InvalidOperationException("Debe ejecutarse ForWard antes de BackPropagation.");
        }

        double[] yPred = outputs;

        // Consigo delta de cada neurona: error * derivada
        double[] delta = new double[outputs.Length];
        for (int i = 0; i < outputs.Length; i++) {
            delta[i] = error[i] * Derive(sums[i], yPred[i]);
        }

        // Consigue nuestro array de errores para capas ocultas
        // Nuestra manera de corregir capas ocultas que no tienen algo tangible
        // como podria ser 'yReal' es hacer una sumatoria de todo sus errores acumulados
        // regresando a neurona 'a^1' una sumatoria de: delta * w[i][j]
        // Dicho de otra manera regreso una suma de cada peso que es asignado
        // una capa adelante por delta de esa misma neurona
        double[] backPropagation = new double[inputs.Length];
        for (int j = 0; j < inputs.Length; j++) {
            double sum = 0.0;
            for (int i = 0; i < outputs.Length; i++) {
                sum += delta[i] * weights[i][j];
            }
            backPropagation[j] = sum;
        }

        // Corrijo los pesos y sesgos
        for (int i = 0; i < outputs.Length; i++) {
            biases[i] += alfa * delta[i];
            for (int j = 0; j < inputs.Length; j++) {
                weights[i][j] += alfa * delta[i] * inputs[j];
            }
        }

        return backPropagation;
    }

    // Conseguimos 'yPred'
    public double[] ForWard(double[] inputs) {
        if (inputs.Length != InputSize) {
            throw new ArgumentException($"La capa espera {InputSize} entradas y recibió {inputs.Length}.", nameof(inputs));
        }
        this.inputs = inputs;

        for (int i = 0; i < outputs.Length; i++) {
            double sum = biases[i];
            for (int j = 0; j < inputs.Length; j++) {
                sum += weights[i][j] * inputs[j];
            }
            sums[i] = sum;
            outputs[i] = Activate(sums[i]);
        }
        return outputs;
    }
}
