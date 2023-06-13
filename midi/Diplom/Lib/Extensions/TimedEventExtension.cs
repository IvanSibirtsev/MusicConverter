using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Interaction;

namespace Diplom.Lib.Extensions;

public static class TimedEventExtension
{
    public static T As<T>(this TimedEvent timedEvent) where T : MidiEvent
    {
        return (T)timedEvent.Event;
    }

    public static TimedEvent RemoveAndGet(this List<TimedEvent> timedEvents, int index)
    {
        var noteOff = timedEvents[index];
        timedEvents.RemoveAt(index);
        return noteOff;
    }

    public static bool TryGetIndex(this List<TimedEvent> timedEvents, Predicate<TimedEvent> match, out int index)
    {
        index = timedEvents.FindIndex(match);
        return index != -1;
    }

    public static List<TimedEvent> GetNoteOnEvents(this ICollection<TimedEvent> events) => events
        .GetTypedEvents(MidiEventType.NoteOn);

    public static List<TimedEvent> GetNoteOffEvents(this ICollection<TimedEvent> events) => events
        .GetTypedEvents(MidiEventType.NoteOff);

    public static List<TimedEvent> GetSetTempoEvents(this ICollection<TimedEvent> events) => events
        .GetTypedEvents(MidiEventType.SetTempo);

    public static List<TimedEvent> GetTypedEvents(this ICollection<TimedEvent> events, MidiEventType midiEventTypeLib)
        => events
            .Where(@event => @event.Event.EventType == midiEventTypeLib)
            .ToList();
}