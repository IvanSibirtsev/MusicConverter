namespace Diplom.Lib;

public static class MidiToArduinoConverter
{
    public static ArduinoMelodyUnit Convert(Note noteLib)
    {
        return new ArduinoMelodyUnit(
            Math.Floor(Math.Pow(2, (noteLib.Midi - 69.0) / 12.0) * 440),
            Math.Round(noteLib.Duration * 1000));
    }
}