using Diplom.Lib.Extensions;
using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Interaction;

namespace Diplom.Lib;


public class Track
{
    public List<Note> Notes { get; } = new();

    public Track(TimedEvent[] events, Header header)
    {
        var notesOn = events.GetNoteOnEvents();
        var notesOff = events.GetNoteOffEvents();

        foreach (var noteOn in notesOn)
        {
            if (!notesOff.TryGetIndex(noteOff => IsThisNoteNeededNoteOff(noteOff, noteOn), out var index))
                continue;

            var noteOff = notesOff.RemoveAndGet(index);
            
            AddNote(noteOff, noteOn, header);
        }
    }

    private static bool IsThisNoteNeededNoteOff(TimedEvent noteOff, TimedEvent noteOn)
    {
        var noteOffEvent = noteOff.As<NoteOffEvent>();
        var noteOnEvent = noteOn.As<NoteOnEvent>();
        return noteOff.Time >= noteOn.Time && noteOffEvent.NoteNumber == noteOnEvent.NoteNumber;
    }

    private void AddNote(TimedEvent noteOff, TimedEvent noteOn, Header header)
    {
        var note = new Note
        {
            DurationTicks = noteOff.Time - noteOn.Time,
            Midi = noteOn.As<NoteOnEvent>().NoteNumber,
            NoteOffVelocity = (double)noteOff.As<NoteOffEvent>().Velocity / 127,
            Ticks = noteOn.Time,
            Velocity = ((double)noteOn.As<NoteOnEvent>().Velocity) / 127,
            Header = header
        };
        Utils.Insert(Notes, js => js.Ticks, note);
    }
}