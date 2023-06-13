namespace Diplom.Lib.Extensions;

public static class IntervalExtensions
{
    public static IReadOnlyList<Interval> Merge(this IReadOnlyList<Interval> intervals)
    {
        for (var index = 1; index < intervals.Count; index++)
        {
            var note = intervals[index];
            var previousNote = intervals[index - 1];
            if (previousNote.Right <= note.Left || Math.Abs(note.Weight - previousNote.Weight) < 0.01)
            {
                continue;
            }

            if (note.Weight > previousNote.Weight)
            {
                note.Left = previousNote.Right;
            }

            if (note.Weight < previousNote.Weight)
            {
                previousNote.Right = note.Left;
            }
        }

        return intervals;
    }
    
    public static Note[] Map(this IEnumerable<Interval> intervals, IEnumerable<Note> notes)
    {
        var newNotes = new List<Note>();
        foreach (var (note, interval) in notes.Zip(intervals))
        {
            newNotes.Add(new Note
            {
                Ticks = interval.Left,
                DurationTicks = interval.Right - interval.Left,
                Header = note.Header,
                Midi = note.Midi,
                NoteOffVelocity = note.NoteOffVelocity,
                Velocity = note.Velocity
            });
        }

        return newNotes.ToArray();
    }
}