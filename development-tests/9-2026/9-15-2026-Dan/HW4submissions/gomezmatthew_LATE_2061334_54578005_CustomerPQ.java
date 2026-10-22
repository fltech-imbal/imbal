/*
  Author: Matthew Gomez
  Email: matthewgomez2025@fit.edu
  Course: CSE 2010
  Section:
  Description of this file: Priority queue (heap) whose entries have
  key = distance (Double) and value = customer name (String).
  It reuses HeapPriorityQueue (Code Fragment 9.8) and changes how two
  entries are compared: a tie in the key is broken by the value
  (alphabetically).
 */

public class CustomerPQ extends HeapPriorityQueue<Double, String>
{
    public static final int MAX_ENTRIES = 100;

    /*
      Compares two entries by key (distance); if the keys are equal,
      compares by value (customer name, alphabetical order).
      a, b: the two entries to compare
    */
    @Override
    protected int compare(Entry<Double, String> a, Entry<Double, String> b)
    {
        int c = a.getKey().compareTo(b.getKey());
        if (c != 0) return c;
        return a.getValue().compareTo(b.getValue());
    }

    /* Returns true if the queue already holds the maximum number of entries. */
    public boolean isFull() { return size() >= MAX_ENTRIES; }

    /* Returns (without removing) the minimum entry, or null if empty. */
    public Entry<Double, String> getMin() { return min(); }

    /* insert(key, value), removeMin(), isEmpty() are inherited. */
}
