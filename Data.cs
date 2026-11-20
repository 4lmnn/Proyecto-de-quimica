using System.ComponentModel;
using System.Diagnostics.Contracts;

namespace ProyectoQuimixa;

class Data {
    static public (string[], double[][], double[], double[][], double[]) Extraer(string path, string[] ids)
    {
        var xList = new List<double[]>(10000);
        var yList = new List<double>(10000);
        var xSearch = new List<double[]>();
        var ySearch = new List<double>();
        var idFound = new List<string>();

        bool lineOne = true;

        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrEmpty(line)) continue;
            if (lineOne)
            {
                lineOne = false;
                continue;
            }
            
            string[] parts = line.Split(',');
            int n = parts.Length;

            string id = parts[0];

            var xCurrent = new double[17];
            for (int i = 0; i < 17; i++)
            {
                xCurrent[i] = double.Parse(parts[n - 17 + i]);
            }
            
            if (Array.IndexOf(ids, id) != -1)
            {
                idFound.Add(id);
                xSearch.Add(xCurrent);
                ySearch.Add(double.Parse(parts[n - 21]));
            }
            else
            {
                xList.Add(xCurrent);
                yList.Add(double.Parse(parts[n - 21]));
            }
        }
        
        if (idFound.Count == 0)
        {
            throw new Exception("No se encontró ninguno de los IDs buscados.");
        }
        // Avisa cuando no se encuentran elementos
        if (idFound.Count < ids.Length)
        {
            Console.WriteLine($"No se encontraron {ids.Length - idFound.Count} elementos");
        }
        return (idFound.ToArray(), xList.ToArray(), yList.ToArray(), xSearch.ToArray(), ySearch.ToArray());
    }

    static public (double[][], double[]) Extraer(string path)
    {
         var xList = new List<double[]>(10000);
        var yList = new List<double>(10000);

        bool lineOne = true;

        foreach (string line in File.ReadLines(path))
        {
            if (string.IsNullOrEmpty(line)) continue;
            if (lineOne)
            {
                lineOne = false;
                continue;
            }
            
            string[] parts = line.Split(',');
            int n = parts.Length;
            var xCurrent = new double[17];

            yList.Add(double.Parse(parts[n - 21]));

            for (int i = 0; i < 17; i++)
            {
                xCurrent[i] = double.Parse(parts[n - 17 + i]);
            }
            xList.Add(xCurrent);
        }

        return (xList.ToArray(), yList.ToArray()); 
    }

    static public (double[], double[]) MinMax(double[][] x)
    {
        var min = new double[x[0].Length];
        var max = new double[x[0].Length];

        Array.Fill(min, double.MaxValue);
        Array.Fill(max, double.MinValue);

        for (int i = 0; i < x.Length; i++)
        {
            for (int j = 0; j < x[0].Length; j++)
            {
                if (min[j] > x[i][j]) min[j] = x[i][j];
                if (max[j] < x[i][j]) max[j] = x[i][j];
            }
        }
        return (min, max);
    }

    static public (double[], double[]) Zscore(double[][] x)
    {
        int row = x.Length;
        int col = x[0].Length;

        var med = new double[col];
        var des = new double[col];

        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                med[j] += x[i][j];
            }
        }

        for (int j = 0; j < col; j ++)
        {
            med[j] /= row;
        }

        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                double diff = x[i][j] - med[j];
                des[j] += diff * diff;
            }
        }

        for (int j = 0; j < col; j++)
        {
            des[j] = Math.Sqrt(des[j] / row);
        }

        return (med, des);
    }

    public static void AplicarMinMax(double[][] x, double[] min, double[] max) {
        for (int i = 0; i < x.Length; i++) {
            for (int j = 0; j < x[0].Length; j++) {
                double range = max[j] - min[j];
                x[i][j] = range > 0 ? ((x[i][j] - min[j]) / range) : 0.0;
            }
        }
    }

    public static void AplicarZscore(double[][] x, double[] med, double[] des)
    {
        for (int i = 0; i < x.Length; i++)
        {
            for (int j = 0; j < x[0].Length; j++)
            {
                x[i][j] = des[j] > 0 ? (x[i][j] - med[j]) / des[j] : 0.0;
            }
        }
    }
}