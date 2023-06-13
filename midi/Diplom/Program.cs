using Diplom.Lib;
using Diplom.Lib.Extensions;
using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Interaction;
using Note = Diplom.Lib.Note;


namespace Diplom;

public static class Program
{
    public static void Main(string[] args)
    {
        var path = args[0];
        
        Console.WriteLine("Parsing: {0}\n", path);

        var midi = MidiFile.Read(path);
        var header = new Header(midi);

        var melodyUnits = midi
            .GetTrackChunks()
            .Select(track => track.Events)
            .Select(track => track.GetTimedEvents().ToArray())
            .Select(track => new Track(track, header))
            .Select(MergeNoteInTrack)
            .Select(notes => notes.Select(MidiToArduinoConverter.Convert).ToArray())
            .ToArray();


        var arduino = melodyUnits.Select(melodyUnit => melodyUnit.MapToArduino()).ToArray();

        Write(arduino);
    }

    private static void Write(string[] tracks)
    {
        Directory.CreateDirectory("Arduino");
        for (var i = 0; i < tracks.Length; i++)
        {
            File.WriteAllText($"Arduino\\{i}.txt", tracks[i]);
        }
    }

    private static Note[] MergeNoteInTrack(Track tracks)
    {
        return tracks.Notes
            .Select(note => new Interval(note))
            .ToArray()
            .Merge()
            .Map(tracks.Notes);
    }
}