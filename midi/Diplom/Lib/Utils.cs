namespace Diplom.Lib;

public static class Utils
{
    public static bool TryGetIndexInArray<T>(List<T> array, Func<T, long> selector, long value, out int index)
    {
        index = 0;
        var beginning = 0;
        var end = array.Count;
        if (array.Count > 0 && selector(array[^1]) <= value)
        {
            index = array.Count - 1;
            return true;
        }
        while (beginning < end)
        {
            var midPoint = (int)Math.Floor(beginning + (double)(end - beginning) / 2);
            var item = array[midPoint];
            var nextItem = array[midPoint + 1];
            if (selector(item) == value)
            {
                for (var i = midPoint; i < array.Count; i++)
                {
                    var testItem = array[i];
                    if (selector(testItem) == value)
                        midPoint = i;
                }

                index = midPoint;
                return true;
            }

            if (selector(item) < value && selector(nextItem) > value)
            {
                index = midPoint;
                return true;
            }
            if (selector(item) > value)
                end = midPoint;
            if (selector(item) < value)
                beginning = midPoint + 1;

        }

        return false;
    }
    
    public static void Insert<T>(List<T> array, Func<T, long> selector, T value)
    {
        if (array.Count > 0)
        {
            var _ = TryGetIndexInArray(array, selector, selector(value), out var index);
            array.Insert(index + 1, value);
        }
        else
        {
            array.Add(value);
        }
    }
}