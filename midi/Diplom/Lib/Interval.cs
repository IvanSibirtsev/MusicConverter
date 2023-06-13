namespace Diplom.Lib;

public class Interval
{
    public long Left { get; set; }
    public long Right { get; set; }
    public double Weight { get; set; }

    public Interval(Note note)
    {
        Left = note.Ticks;
        Right = note.Ticks + note.DurationTicks;
        Weight = note.Velocity;
    }
}