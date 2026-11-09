/*
 Author: Parker Blue
 Email: pblue2025@my.fit.edu
 Course: CSE2010
 Section: 9:30 AM Lab
 Description: Skip list implementation of a sorted map using quad linked nodes with a negative
 and positive infinity sentinels on every level. Heights for new entries come from FakeRandHeight.
 The list always keeps exactly one empty level on top (s_h), so the number of levels is max entry
 height plus 2.
*/
import java.util.ArrayList;

public class SkipListMap<K extends Comparable<K>, V> {
  // An entry (key, value) returned to callers
  public static class Entry<K, V> {
    private final K key;
    private final V value;
    public Entry (K k, V v) { key = k; value = v; }
    public K getKey() { return key; }
    public V getValue() { return value; }
  }

  // Quad linked node; sentinels have isMinus or isPlus set and null key
  private static class Node<K, V> {
    K key;
    V value;
    Node<K, V> prev, next, above, below;
    boolean isMinus, isPlus;
    Node(K k, V v) { key = k; value = v; }
  }

  private Node<K, V> topLeft;     // negative infinity sentinel of the top level
  private int topLevel;           // index of the top level
  private int size;               // number of entries
  private final FakeRandHeight randHeight = new FakeRandHeight();

  // Creates an empty skip list with a single level S0
  public SkipListMap() {
    topLeft = newLevel(null, null);
    topLevel = 0;
    size = 0;
  }

  /* Builds one empty level from negative infinity to positive infinity and then links it vertically
  above the given sentinels. Will be null when building the bottom level. Then returns the new
  negative infinity sentinel.
  */
  private Node<K, V> newLevel(Node<K, V> belowMinus, Node<K, V> belowPlus) {
    Node<K, V> minus = new Node<>(null, null);
    Node<K, V> plus = new Node<>(null, null);
    minus.isMinus = true;
    plus.isPlus = true;
    minus.next = plus;
    plus.prev = minus;
    if (belowMinus != null) {
      minus.below = belowMinus;
      belowMinus.above = minus;
      plus.below = belowPlus;
      belowPlus.above = plus;
    }
  return minus;
  }

  // Adds one empty level on top of the current top level
  private void addTopLevel() {
    Node<K, V> plus = topLeft;
    while (!plus.isPlus) plus = plus.next;
    topLeft = newLevel(topLeft, plus);
    topLevel++;
  }

  // Removes the top level which is only called when it is empty
  private void removeTopLevel() {
    Node<K, V> newTop = topLeft.below;
    newTop.above = null;
    Node<K, V> plus = newTop;
    while (!plus.isPlus) plus = plus.next;
    plus.above = null;
    topLeft = newTop;
    topLevel--;
  }

  // Compare a node's key to k, treating sentinels as negative infinity and positive infinity
  private int compare(Node<K, V> p, K k) {
    if (p.isMinus) return -1;
    if (p.isPlus) return 1;
    return p.key.compareTo(k);
  }

  /* Standard skip search which returns the bottom level not with the largest key. Is less than or
  equal to k. Could possibly be the negative infinite sentinel.
  */
  private Node<K, V> skipSearch(K k) {
    Node<K, V> p = topLeft;
    while (true) {
      while (compare(p.next, k) <= 0) {
        p = p.next;
      }
      if (p.below == null) break;
      p = p.below;
    }
    return p;
  }

  public int size() { return size; }
  public boolean isEmpty() { return size == 0; }

  // get(key) is the vallue associated with key, or null
  public V get(K k) {
    Node<K, V> p = skipSearch(k);
    if (!p.isMinus && p.key.compareTo(k) == 0) return p.value;
    return null;
  }

  /* put(key, value) will insert a new entry and return null. If the key exists, replace its value
  in the whole tower and return the old value.
  */
  public V put(K k, V v) {
    Node<K, V> p = skipSearch(k);
    // key already present, so will update every not in its tower
    if (!p.isMinus && p.key.compareTo(k) == 0) {
      V old = p.value;
      for (Node<K, V> q = p; q != null; q = q.above) q.value = v;
      return old;
    }

    int h = randHeight.get();
    while (topLevel < h + 1) addTopLevel();
    Node<K, V> below = null;
    for (int i = 0; i <= h; i++) {
      if (i > 0) {
        while (p.above == null) p = p.prev;
        p = p.above;
      }
      Node<K, V> q = new Node<>(k, v);
      q.prev = p;
      q.next = p.next;
      p.next.prev = q;
      p.next = q;
      if (below != null) {
        q.below = below;
        below.above = q;
      }
      below = q;
    }
    size++;
    return null;
  }

  /* remove(key) will remove the entire tower and return its value. Null though if the key is not
  present. Shrinks the list so that only one empty level is left on the top of the list.
  */
  public V remove(K k) {
    Node<K, V> p = skipSearch(k);
    if (p.isMinus || p.key.compareTo(k) != 0) return null;
    V old = p.value;
    for (Node<K, V> q = p; q != null; q = q.above) {
      q.prev.next = q.next;
      q.next.prev = q.prev;
    }
    size--;

    // drop extra empty levels
    while (topLevel > 0 && topLeft.below.next.isPlus) {
      removeTopLevel();
    }
    return old;
  }

  // floorEntry(key) is the entry with the largest key which is <= key or null
  public Entry<K, V> floorEntry(K k) {
    Node<K, V> p = skipSearch(k);
    if (p.isMinus) return null;
    return new Entry<>(p.key, p.value);
  }

  // ceilingEntry(key) is the entry with the smallest key which is >= key or null
  public Entry<K, V> ceilingEntry(K k) {
    Node<K, V> p = ceilingNode(k);
    if (p.isPlus) return null;
    return new Entry<>(p.key, p.value);
  }

  // bottom level node with the smallest key which is >= k, could be positive infinity
  private Node<K, V> ceilingNode(K k) {
    Node<K, V> p = skipSearch(k);
    if (!p.isMinus && p.key.compareTo(k) == 0) return p;
    return p.next;
  }

  // subMap(key1, key2) is all the entrues with key1 which is <= key <= key2, in order
  public ArrayList<Entry<K, V>> subMap(K k1, K k2) {
    ArrayList<Entry<K, V>> result = new ArrayList<>();
    for (Node<K, V> p = ceilingNode(k1); !p.isPlus && p.key.compareTo(k2) <= 0; p = p.next) {
      result.add(new Entry<>(p.key, p.value));
    }
    return result;
  }

  // first and last entries, null if empty
  public Entry<K, V> firstEntry() {
    Node<K, V> p = bottomLeft().next;
    return p.isPlus ? null : new Entry<>(p.key, p.value);
  }

  public Entry<K, V> lastEntry() {
    Node<K, V> p = bottomLeft();
    while (!p.next.isPlus) p = p.next;
    return p.isMinus ? null : new Entry<>(p.key, p.value);
  }

  private Node<K, V> bottomLeft() {
    Node<K, V> p = topLeft;
    while (p.below != null) p = p.below;
    return p;
  }

  // Returns the levels from the top down to S0, one string per level
  public ArrayList<String> levelStrings() {
    ArrayList<String> lines = new ArrayList<>();
    Node<K, V> left = topLeft;
    for (int i = topLevel; i >= 0; i--, left = left.below) {
      StringBuilder sb = new StringBuilder("(S" + i + ")");
      if (left.next.isPlus) {
        sb.append(" empty");
      } else {
          for (Node<K, V> p = left.next; !p.isPlus; p = p.next) {
            sb.append(' ').append(p.key).append(':').append(p.value);
          }
      }
      lines.add(sb.toString());
    }
    return lines;
  }
}
