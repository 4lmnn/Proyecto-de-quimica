namespace  ProyectoQuimixa;

class Program
{
    static void Main()
    {
        var (x, y) = Data.Extraer("./data_curated.csv");

        var rng = new Random(42);
        var (xTrain, yTrain, xTest, yTest) = Split(x, y, 80, rng);

        var molWtTest = new double[xTest.Length];
        for (int i = 0; i < xTest.Length; i++) molWtTest[i] = xTest[i][0];

        var (med, des) = Data.Zscore(xTrain);
        Data.AplicarZscore(xTrain, med, des);
        Data.AplicarZscore(xTest , med, des);

        var indexTrain = new int[xTrain.Length];
        for (int i = 0; i < xTrain.Length; i++) indexTrain[i] = i;


    
    }
    public static void Barajar(int[] array, Random rng)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    public static (double[][], double[], double[][], double[]) Split(double[][] x, double[]y, int div, Random rng)
    {
        var index = new int[x.Length];
        // Inicializa los indices
        for (int i = 0; i < x.Length; i ++) index[i] = i;

        Barajar(index, rng);
        
        int split = (x.Length * div) / 100;
        var (indexTrain, indexTest) = (index[..split], index[split..]);

        var xTrain = new double[indexTrain.Length][];
        var yTrain = new double[indexTrain.Length];
        for (int i = 0; i < indexTrain.Length; i++)
        {
            xTrain[i] = x[indexTrain[i]];
            yTrain[i] = y[indexTrain[i]];
        }

        var xTest = new double[indexTest.Length][];
        var yTest = new double[indexTest.Length];
        for (int i = 0; i < indexTest.Length; i++)
        {
            xTest[i] = x[indexTest[i]];
            yTest[i] = y[indexTest[i]];
        }

        return (xTrain, yTrain, xTest, yTest);
    }
}
