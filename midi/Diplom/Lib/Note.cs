namespace Diplom.Lib;

public class Note
{
    public long DurationTicks { get; set; }
    public int Midi { get; set; }
    public double NoteOffVelocity { get; set; }
    public long Ticks { get; set; }
    public double Velocity { get; set; }
    
    public Header Header { get; init; }

    public double Duration => Header.TicksToSeconds(Ticks + DurationTicks) - Header.TicksToSeconds(Ticks);
}