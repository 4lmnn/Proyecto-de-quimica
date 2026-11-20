using System.Globalization;

namespace ProyectoQuimixa;

class Data {
    private static readonly string[] FeatureNames =
    {
        "MolWt", "MolLogP", "MolMR", "HeavyAtomCount", "NumHAcceptors",
        "NumHDonors", "NumHeteroatoms", "NumRotatableBonds", "NumValenceElectrons",
        "NumAromaticRings", "NumSaturatedRings", "NumAliphaticRings", "RingCount",
        "TPSA", "LabuteASA", "BalabanJ", "BertzCT"
    };

    private static (List<string> ids, List<double[]> x, List<double> y) Load(string path)
    {
        using var reader = new StreamReader(path);
        string? headerLine = reader.ReadLine();
        if (headerLine is null) throw new InvalidDataException("El archivo CSV está vacío.");

        var headers = ParseCsvLine(headerLine);
        var columns = headers.Select((name, index) => (name, index))
            .ToDictionary(item => item.name, item => item.index, StringComparer.OrdinalIgnoreCase);
        int idColumn = columns["ID"];
        int targetColumn = columns["Solubility"];
        int[] featureColumns = FeatureNames.Select(name => columns[name]).ToArray();

        var ids = new List<string>();
        var x = new List<double[]>();
        var y = new List<double>();
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            string[] parts = ParseCsvLine(line);
            ids.Add(parts[idColumn]);
            y.Add(double.Parse(parts[targetColumn], CultureInfo.InvariantCulture));

            var features = new double[featureColumns.Length];
            for (int i = 0; i < featureColumns.Length; i++)
            {
                features[i] = double.Parse(parts[featureColumns[i]], CultureInfo.InvariantCulture);
            }
            x.Add(features);
        }

        return (ids, x, y);
    }

    private static string[] ParseCsvLine(string line)
    {
        var fields = new List<string>();
        var field = new System.Text.StringBuilder();
        bool quoted = false;
        for (int i = 0; i < line.Length; i++)
        {
            char current = line[i];
            if (current == '"')
            {
                if (quoted && i + 1 < line.Length && line[i + 1] == '"')
                {
                    field.Append('"');
                    i++;
                }
                else
                {
                    quoted = !quoted;
                }
            }
            else if (current == ',' && !quoted)
            {
                fields.Add(field.ToString());
                field.Clear();
            }
            else
            {
                field.Append(current);
            }
        }
        fields.Add(field.ToString());
        return fields.ToArray();
    }

    static public (string[], double[][], double[], double[][], double[]) Extraer(string path, string[] ids)
    {
        var (allIds, allX, allY) = Load(path);
        var xSearch = new List<double[]>();
        var ySearch = new List<double>();
        var idFound = new List<string>();
        var xList = new List<double[]>(allIds.Count);
        var yList = new List<double>(allIds.Count);

        for (int i = 0; i < allIds.Count; i++)
        {
            if (Array.IndexOf(ids, allIds[i]) != -1)
            {
                idFound.Add(allIds[i]);
                xSearch.Add(allX[i]);
                ySearch.Add(allY[i]);
            }
            else
            {
                xList.Add(allX[i]);
                yList.Add(allY[i]);
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
        var (_, x, y) = Load(path);
        return (x.ToArray(), y.ToArray());
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