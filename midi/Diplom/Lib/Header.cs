using Diplom.Lib.Extensions;
using Melanchall.DryWetMidi.Core;
using Melanchall.DryWetMidi.Interaction;

namespace Diplom.Lib;

public class Header
{
    private List<TempoEvent> TempoEvents { get; } = new();
    private int Ppq { get; }

    public Header(MidiFile midiFile)
    {
        Ppq = ((TicksPerQuarterNoteTimeDivision)midiFile.TimeDivision).TicksPerQuarterNote;

        Initialize(midiFile);
    }

    public double TicksToSeconds(long ticks)
    {
        long Selector(TempoEvent tempoEvent) => tempoEvent.Ticks;
        if (Utils.TryGetIndexInArray(TempoEvents, Selector, ticks, out var index))
        {
            var tempo = TempoEvents[index];
            var tempoTime = tempo.Time;
            var elapsedBeats = (double)(ticks - tempo.Ticks) / Ppq;
            
            return tempoTime + 60 / tempo.Bpm * elapsedBeats;
        }

        var beats = (double)ticks;
        return 0.5 * beats;
    }

    private void Initialize(MidiFile midiFile)
    {
        foreach (var track in midiFile.GetTrackChunks())
        {
            var timedEvents = track.Events.GetTimedEvents();

            TempoEvents
                .AddRange(timedEvents
                    .GetSetTempoEvents()
                    .Select(timedEvent =>
                    {
                        var setTempoEvent = timedEvent.As<SetTempoEvent>();
                        return new TempoEvent
                        {
                            Bpm = 60000000 / (double)setTempoEvent.MicrosecondsPerQuarterNote,
                            Ticks = timedEvent.Time,
                        };
                    }));
        }

        Update();
    }

    private void Update()
    {
        var currentTime = 0.0;
        var lastEventBeats = 0.0;
        TempoEvents.Sort((x, y) => (int)(x.Ticks - y.Ticks));
        
        for (var i = 0; i < TempoEvents.Count; i++)
        {
            var tempoEvent = TempoEvents[i];
            var previousBpm = i > 0 ? TempoEvents[i - 1].Bpm : TempoEvents[0].Bpm;
            var beats = (double)tempoEvent.Ticks / Ppq - lastEventBeats;
            var elapsedSeconds = 60 / previousBpm * beats;

            tempoEvent.Time = elapsedSeconds + currentTime;
            currentTime = tempoEvent.Time;
            lastEventBeats += beats;
        }
    }
}