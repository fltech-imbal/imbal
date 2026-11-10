/*
    Author: Matthew Gomez
    Course: CSE 2010
    Date: October 29, 2026
    Description of this file:
    A sorted map (key -> value) implemented as a skip list. The skip list
    is a stack of sorted, doubly linked lists S0, S1, ..., Sh. S0 holds
    every entry; each higher list holds a random subset of the list below
    it. Every list starts with a -infinity sentinel and ends with a
    +infinity sentinel, and the top list Sh always holds only the two
    sentinels (it is empty). The height of each new entry comes from
    FakeRandHeight.get(), which makes the output predictable.

    Supported operations (expected O(log n) time each):
      get(key), put(key, value), remove(key),
      floorEntry(key), ceilingEntry(key), subMap(key1, key2)
*/

public class SkipListMap<K extends Comparable<K>, V>
{
    /* ---------------- nested Entry class ---------------- */

    /*
      A read-only key-value pair that is returned to the caller by
      floorEntry, ceilingEntry and subMap.
    */
    public static class Entry<K, V>
    {
        private final K key;     // key of this entry
        private final V value;   // value associated with the key

        /*
          Creates an entry.
          key: the key of the entry
          value: the value of the entry
        */
        public Entry(K key, V value)
        {
            this.key = key;
            this.value = value;
        }

        /* Returns the key of this entry. */
        public K getKey() { return key; }

        /* Returns the value of this entry. */
        public V getValue() { return value; }
    }

    /* ---------------- nested Node class ---------------- */

    /*
      One position of the skip list (a "quad node"). It knows its
      neighbors on the same list (prev, next) and the positions of the
      same key on the list above and below (above, below).
      The -infinity sentinel of a list has prev == null and the
      +infinity sentinel has next == null; both have a null key.
    */
    private static class Node<K, V>
    {
        K key;               // key stored at this position (null for sentinels)
        V value;             // value stored at this position (null for sentinels)
        Node<K, V> prev;     // left neighbor on the same list
        Node<K, V> next;     // right neighbor on the same list
        Node<K, V> above;    // same key on the list above (null if none)
        Node<K, V> below;    // same key on the list below (null on S0)

        /*
          Creates a node that is not linked to anything yet.
          key: the key of the node
          value: the value of the node
        */
        Node(K key, V value)
        {
            this.key = key;
            this.value = value;
        }
    }

    /* ---------------- instance variables ---------------- */

    private Node<K, V> top;              // -infinity sentinel of the top list Sh
    private Node<K, V> topRight;         // +infinity sentinel of the top list Sh
    private int height;                  // index h of the top list (the top list is always empty)
    private int size;                    // number of entries in the map
    private FakeRandHeight randHeight;   // source of the heights of new entries

    /*
      Creates an empty skip list that has only the list S0, which holds
      just the two sentinels.
      randHeight: the source of the heights of new entries
    */
    public SkipListMap(FakeRandHeight randHeight)
    {
        this.randHeight = randHeight;
        top = new Node<K, V>(null, null);       // -infinity of S0
        topRight = new Node<K, V>(null, null);  // +infinity of S0
        top.next = topRight;
        topRight.prev = top;
        height = 0;
        size = 0;
    }

    /* ---------------- small helpers ---------------- */

    /* Returns true if n is the -infinity sentinel of its list. */
    private boolean isNegInf(Node<K, V> n) { return n.prev == null; }

    /* Returns true if n is the +infinity sentinel of its list. */
    private boolean isPosInf(Node<K, V> n) { return n.next == null; }

    /* Returns the number of entries in the map. */
    public int size() { return size; }

    /* Returns the index of the top list (which is always empty). */
    public int getHeight() { return height; }

    /*
      Searches for a key. Starts at the top-left sentinel; on each list it
      moves right while the next key is <= key, then drops down one list.
      key: the key to search for
      Returns the node on S0 with the largest key <= key (this is the
      -infinity sentinel of S0 if every key is larger than key).
    */
    private Node<K, V> skipSearch(K key)
    {
        Node<K, V> p = top;        // current position
        Node<K, V> found = null;   // last position visited on the list that was just scanned
        while (p != null)
        {
            while (!isPosInf(p.next) && p.next.key.compareTo(key) <= 0)
                p = p.next;        // scan right on this list
            found = p;
            p = p.below;           // drop down (null after S0)
        }
        return found;
    }

    /*
      Links a new node right after node p on p's list.
      p: the node that will be the left neighbor of the new node
      key, value: the key and value of the new node
      Returns the new node.
    */
    private Node<K, V> insertAfter(Node<K, V> p, K key, V value)
    {
        Node<K, V> n = new Node<K, V>(key, value);
        n.prev = p;
        n.next = p.next;
        p.next.prev = n;
        p.next = n;
        return n;
    }

    /*
      Adds a new empty list (just two sentinels) above the current top
      list, so it becomes the new top list.
    */
    private void addLevel()
    {
        Node<K, V> newLeft = new Node<K, V>(null, null);    // -infinity of the new list
        Node<K, V> newRight = new Node<K, V>(null, null);   // +infinity of the new list
        newLeft.next = newRight;
        newRight.prev = newLeft;
        newLeft.below = top;
        top.above = newLeft;
        newRight.below = topRight;
        topRight.above = newRight;
        top = newLeft;
        topRight = newRight;
        height++;
    }

    /* Removes the top list (it must be empty and have a list below it). */
    private void removeTopLevel()
    {
        top = top.below;
        top.above = null;
        topRight = topRight.below;
        topRight.above = null;
        height--;
    }

    /* ---------------- map operations ---------------- */

    /*
      Returns the value associated with key, or null if key is not in the map.
      key: the key to look for
    */
    public V get(K key)
    {
        Node<K, V> p = skipSearch(key);
        if (isNegInf(p) || p.key.compareTo(key) != 0)
            return null;
        return p.value;
    }

    /*
      Adds the entry if key is not in the map and returns null; otherwise
      replaces the value and returns the old value. A new entry is placed
      on S0 and on as many higher lists as FakeRandHeight.get() says, and
      new lists are created so that the top list stays empty.
      key: the key of the entry
      value: the value of the entry
    */
    public V put(K key, V value)
    {
        Node<K, V> p = skipSearch(key);   // largest key <= key on S0

        /* key already exists: replace the value on every list of its tower */
        if (!isNegInf(p) && p.key.compareTo(key) == 0)
        {
            V old = p.value;
            for (Node<K, V> n = p; n != null; n = n.above)
                n.value = value;
            return old;
        }

        /* key is new: get its height and make sure the lists exist */
        int r = randHeight.get();         // the new entry is on S0 .. Sr
        while (height <= r)
            addLevel();                   // keeps one empty list above Sr

        /* insert on S0, then on S1 .. Sr, linking each node to the one below */
        Node<K, V> lower = insertAfter(p, key, value);   // new node on the list below
        Node<K, V> base = p;                             // left neighbor on the current list
        for (int level = 1; level <= r; level++)
        {
            while (base.above == null)
                base = base.prev;         // walk left until a position that reaches the next list
            base = base.above;
            Node<K, V> n = insertAfter(base, key, value);
            n.below = lower;
            lower.above = n;
            lower = n;
        }
        size++;
        return null;
    }

    /*
      Removes the entry with the given key from every list and returns its
      value; returns null if key is not in the map. Afterwards all but one
      empty list at the top are removed.
      key: the key to remove
    */
    public V remove(K key)
    {
        Node<K, V> p = skipSearch(key);
        if (isNegInf(p) || p.key.compareTo(key) != 0)
            return null;
        V old = p.value;

        /* unlink the whole tower, from S0 upward */
        Node<K, V> n = p;
        while (n != null)
        {
            Node<K, V> up = n.above;
            n.prev.next = n.next;
            n.next.prev = n.prev;
            n = up;
        }

        /* keep only one empty list at the top */
        while (height > 0 && isPosInf(top.below.next))
            removeTopLevel();
        size--;
        return old;
    }

    /*
      Returns the entry with the largest key <= key, or null if no such
      entry exists.
      key: the key to compare with
    */
    public Entry<K, V> floorEntry(K key)
    {
        Node<K, V> p = skipSearch(key);
        if (isNegInf(p))
            return null;
        return new Entry<K, V>(p.key, p.value);
    }

    /*
      Returns the entry with the smallest key >= key, or null if no such
      entry exists.
      key: the key to compare with
    */
    public Entry<K, V> ceilingEntry(K key)
    {
        Node<K, V> p = skipSearch(key);
        if (!isNegInf(p) && p.key.compareTo(key) == 0)
            return new Entry<K, V>(p.key, p.value);   // exact match
        Node<K, V> n = p.next;                         // first key > key
        if (isPosInf(n))
            return null;
        return new Entry<K, V>(n.key, n.value);
    }

    /*
      Returns all entries with key1 <= key <= key2 in increasing key order
      (an empty list if there are none).
      key1: the smallest key to include
      key2: the largest key to include
    */
    public DoublyLinkedList<Entry<K, V>> subMap(K key1, K key2)
    {
        DoublyLinkedList<Entry<K, V>> result = new DoublyLinkedList<Entry<K, V>>();
        if (key1.compareTo(key2) > 0)
            return result;

        /* find the first entry with key >= key1 on S0 */
        Node<K, V> p = skipSearch(key1);
        if (isNegInf(p) || p.key.compareTo(key1) < 0)
            p = p.next;

        /* walk right along S0 while key <= key2 */
        while (!isPosInf(p) && p.key.compareTo(key2) <= 0)
        {
            result.addLast(new Entry<K, V>(p.key, p.value));
            p = p.next;
        }
        return result;
    }

    /* ---------------- printing ---------------- */

    /*
      Returns the text of one list, for example "(S1) 0926:BeachDay 1126:Thanksgiving",
      or "(S3) empty" if the list has no entries.
      level: the index of the list (0 .. getHeight())
    */
    public String levelToString(int level)
    {
        if (level < 0 || level > height)
            throw new IllegalArgumentException("no such list: S" + level);
        Node<K, V> p = top;                   // will become the -infinity sentinel of S(level)
        for (int i = height; i > level; i--)
            p = p.below;
        StringBuilder sb = new StringBuilder("(S" + level + ")");
        if (isPosInf(p.next))
        {
            sb.append(" empty");
        }
        else
        {
            for (p = p.next; !isPosInf(p); p = p.next)
                sb.append(" ").append(p.key).append(":").append(p.value);
        }
        return sb.toString();
    }
}
