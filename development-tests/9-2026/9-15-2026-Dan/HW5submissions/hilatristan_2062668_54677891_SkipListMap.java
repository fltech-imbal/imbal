import java.util.ArrayList;
import java.util.List;

/** An ordered map implemented with horizontally and vertically linked nodes. */
public class SkipListMap<K extends Comparable<K>, V> {
    /** An immutable key/value pair returned by ordered queries. */
    public static class Entry<K, V> {
        private final K key;
        private final V value;

        public Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public K getKey() { return key; }
        public V getValue() { return value; }
    }

    /** A null key marks either end sentinel; sentinels are never compared. */
    private static class Node<K, V> {
        K key;
        V value;
        Node<K, V> left, right, up, down;

        Node(K key, V value) {
            this.key = key;
            this.value = value;
        }
    }

    private Node<K, V> head;
    private Node<K, V> tail;
    private int height;
    private int size;
    private final FakeRandHeight random = new FakeRandHeight();

    /** Initially level zero is the empty top level. */
    public SkipListMap() {
        head = new Node<>(null, null);
        tail = new Node<>(null, null);
        head.right = tail;
        tail.left = head;
    }

    public int size() { return size; }
    public boolean isEmpty() { return size == 0; }

    /** Descend from the top to the bottom predecessor of key, including equality. */
    private Node<K, V> find(K key) {
        Node<K, V> node = head;
        while (true) {
            while (node.right.key != null && node.right.key.compareTo(key) <= 0) {
                node = node.right;
            }
            if (node.down == null) return node;
            node = node.down;
        }
    }

    /** Return the associated value, or null when the key is absent. */
    public V get(K key) {
        Node<K, V> node = find(key);
        return node.key != null && node.key.compareTo(key) == 0 ? node.value : null;
    }

    /** Add an empty level above the current top, linking both sentinels. */
    private void addLevel() {
        Node<K, V> newHead = new Node<>(null, null);
        Node<K, V> newTail = new Node<>(null, null);
        newHead.right = newTail;
        newTail.left = newHead;
        newHead.down = head;
        newTail.down = tail;
        head.up = newHead;
        tail.up = newTail;
        head = newHead;
        tail = newTail;
        height++;
    }

    /** Insert a new tower, or replace every value in an existing tower. */
    public V put(K key, V value) {
        Node<K, V> predecessor = find(key);
        if (predecessor.key != null && predecessor.key.compareTo(key) == 0) {
            V old = predecessor.value;
            for (Node<K, V> node = predecessor; node != null; node = node.up) {
                node.value = value;
            }
            return old;
        }

        // Reserve an empty level above the tallest data tower.
        int towerHeight = random.get();
        while (height <= towerHeight) addLevel();
        Node<K, V> below = null;
        for (int level = 0; level <= towerHeight; level++) {
            Node<K, V> node = new Node<>(key, value);
            node.left = predecessor;
            node.right = predecessor.right;
            predecessor.right.left = node;
            predecessor.right = node;
            node.down = below;
            if (below != null) below.up = node;
            below = node;

            // Move left to the closest tower that reaches the next level.
            if (level < towerHeight) {
                while (predecessor.up == null) predecessor = predecessor.left;
                predecessor = predecessor.up;
            }
        }
        size++;
        return null;
    }

    /** Remove a complete tower and retain exactly one empty top level. */
    public V remove(K key) {
        Node<K, V> node = find(key);
        if (node.key == null || node.key.compareTo(key) != 0) return null;
        V old = node.value;
        while (node != null) {
            node.left.right = node.right;
            node.right.left = node.left;
            node = node.up;
        }
        size--;
        while (height > 0 && head.down.right.key == null) {
            head = head.down;
            tail = tail.down;
            head.up = null;
            tail.up = null;
            height--;
        }
        return old;
    }

    private Entry<K, V> entry(Node<K, V> node) {
        return node.key == null ? null : new Entry<>(node.key, node.value);
    }

    /** Return the greatest entry whose key is at most key. */
    public Entry<K, V> floorEntry(K key) { return entry(find(key)); }

    /** Return the least entry whose key is at least key. */
    public Entry<K, V> ceilingEntry(K key) {
        Node<K, V> node = find(key);
        if (node.key == null || node.key.compareTo(key) < 0) node = node.right;
        return entry(node);
    }

    /** Return entries in the inclusive interval in O(log n + result size) expected time. */
    public List<Entry<K, V>> subMap(K first, K last) {
        List<Entry<K, V>> result = new ArrayList<>();
        Node<K, V> node = find(first);
        if (node.key == null || node.key.compareTo(first) < 0) node = node.right;
        while (node.key != null && node.key.compareTo(last) <= 0) {
            result.add(entry(node));
            node = node.right;
        }
        return result;
    }

    /** Print levels from the empty top down; the formatter controls date padding. */
    public String formatLevels(java.util.function.Function<K, String> formatKey) {
        StringBuilder result = new StringBuilder();
        Node<K, V> levelHead = head;
        for (int level = height; level >= 0; level--) {
            if (result.length() > 0) result.append('\n');
            result.append("(S").append(level).append(")");
            Node<K, V> node = levelHead.right;
            if (node.key == null) result.append(" empty");
            while (node.key != null) {
                result.append(' ').append(formatKey.apply(node.key))
                      .append(':').append(node.value);
                node = node.right;
            }
            levelHead = levelHead.down;
        }
        return result.toString();
    }
}
