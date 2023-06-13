using System.Globalization;
using System.Text;

namespace Diplom.Lib.Extensions;

public static class ArduinoMelodyUnitExtensions
{
    public static string MapToPySine(this IEnumerable<ArduinoMelodyUnit> converted)
    {
        var sb = new StringBuilder();
        sb.AppendLine("import pysine")
            .AppendLine()
            .AppendLine("def play(hz, time):")
            .AppendLine("\tpysine.sine(hz, time / 1000)")
            .AppendLine().AppendLine();
        
        foreach (var unit in converted)
        {
            sb.AppendLine($"play({unit.Frequency.ToString(CultureInfo.InvariantCulture)}, {unit.Delay.ToString(CultureInfo.InvariantCulture)})");
        }

        return sb.ToString();
    }

    public static string MapToArduino(this IEnumerable<ArduinoMelodyUnit> converted)
    {
        var sb = new StringBuilder();
        foreach (var unit in converted)
        {
            sb.AppendLine($"tone(13, {unit.Frequency.ToString(CultureInfo.InvariantCulture)}, {unit.Delay.ToString(CultureInfo.InvariantCulture)})");
        }


        return sb.ToString();
    }
}