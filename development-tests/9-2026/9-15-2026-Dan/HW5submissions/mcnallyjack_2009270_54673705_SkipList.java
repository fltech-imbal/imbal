
import java.util.ArrayList;
import java.util.List;

public class SkipList {
    private static final int MAX_HEIGHT = 10;

    private static class Node {
        int key;
        String value;
        Node[] forward;

        Node(int key, String value, int height) {
            this.key = key;
            this.value = value;
            this.forward = new Node[height + 1];
        }
    }

    private final Node head;
    private int currentHeight;
    private int size;

    // Use the instructor-provided random-height generator.
    private final FakeRandHeight randHeight = new FakeRandHeight();

    public SkipList() {
        head = new Node(0, null, MAX_HEIGHT);
        currentHeight = 0;
        size = 0;
    }

    public int size() {
        return size;
    }

    // Return the value associated with key, or null if absent.
    public String get(int key) {
        Node current = head;

        for (int level = currentHeight; level >= 0; level--) {
            while (current.forward[level] != null
                    && current.forward[level].key < key) {
                current = current.forward[level];
            }
        }

        current = current.forward[0];

        if (current != null && current.key == key) {
            return current.value;
        }

        return null;
    }

    // Insert a new entry or replace an existing entry.
    // Return null for insertion, or the old value for replacement.
    public String put(int key, String value) {
        Node[] update = new Node[MAX_HEIGHT + 1];
        Node current = head;

        for (int level = currentHeight; level >= 0; level--) {
            while (current.forward[level] != null
                    && current.forward[level].key < key) {
                current = current.forward[level];
            }

            update[level] = current;
        }

        current = current.forward[0];

        if (current != null && current.key == key) {
            String oldValue = current.value;
            current.value = value;
            return oldValue;
        }

        // IMPORTANT: FakeRandHeight defines get(), not getRandHeight().
        int newHeight = randHeight.get();

        if (newHeight < 0) {
            newHeight = 0;
        }

        if (newHeight > MAX_HEIGHT) {
            newHeight = MAX_HEIGHT;
        }

        if (newHeight > currentHeight) {
            for (int level = currentHeight + 1;
                    level <= newHeight; level++) {
                update[level] = head;
            }

            currentHeight = newHeight;
        }

        Node newNode = new Node(key, value, newHeight);

        for (int level = 0; level <= newHeight; level++) {
            newNode.forward[level] = update[level].forward[level];
            update[level].forward[level] = newNode;
        }

        size++;
        return null;
    }

    // Remove an entry and return its value, or null if absent.
    public String remove(int key) {
        Node[] update = new Node[MAX_HEIGHT + 1];
        Node current = head;

        for (int level = currentHeight; level >= 0; level--) {
            while (current.forward[level] != null
                    && current.forward[level].key < key) {
                current = current.forward[level];
            }

            update[level] = current;
        }

        current = current.forward[0];

        if (current == null || current.key != key) {
            return null;
        }

        for (int level = 0; level <= currentHeight; level++) {
            if (update[level].forward[level] != current) {
                break;
            }

            update[level].forward[level] = current.forward[level];
        }

        while (currentHeight > 0
                && head.forward[currentHeight] == null) {
            currentHeight--;
        }

        size--;
        return current.value;
    }

    // Return the entry with the greatest key <= the given key.
    public Entry floorEntry(int key) {
        Node current = head;

        for (int level = currentHeight; level >= 0; level--) {
            while (current.forward[level] != null
                    && current.forward[level].key <= key) {
                current = current.forward[level];
            }
        }

        if (current == head) {
            return null;
        }

        return new Entry(current.key, current.value);
    }

    // Return the entry with the smallest key >= the given key.
    public Entry ceilingEntry(int key) {
        Node current = head;

        for (int level = currentHeight; level >= 0; level--) {
            while (current.forward[level] != null
                    && current.forward[level].key < key) {
                current = current.forward[level];
            }
        }

        current = current.forward[0];

        if (current == null) {
            return null;
        }

        return new Entry(current.key, current.value);
    }

    // Return all entries with key1 <= key <= key2.
    public List<Entry> subMap(int key1, int key2) {
        List<Entry> result = new ArrayList<>();

        if (key1 > key2) {
            return result;
        }

        Node current = head;

        // Find the first entry whose key is >= key1.
        for (int level = currentHeight; level >= 0; level--) {
            while (current.forward[level] != null
                    && current.forward[level].key < key1) {
                current = current.forward[level];
            }
        }

        current = current.forward[0];

        // Traverse the bottom level until the end of the range.
        while (current != null && current.key <= key2) {
            result.add(new Entry(current.key, current.value));
            current = current.forward[0];
        }

        return result;
    }

    // Return all entries in ascending key order.
    public List<Entry> allEntries() {
        List<Entry> result = new ArrayList<>();
        Node current = head.forward[0];

        while (current != null) {
            result.add(new Entry(current.key, current.value));
            current = current.forward[0];
        }

        return result;
    }

    // Print from the highest active level down to level zero.
    public String printSkipList() {
        StringBuilder result = new StringBuilder();

        for (int level = currentHeight; level >= 0; level--) {
            result.append("(S").append(level).append(")");

            Node current = head.forward[level];

            while (current != null) {
                result.append(" ")
                      .append(formatDate(current.key))
                      .append(":")
                      .append(current.value);

                current = current.forward[level];
            }

            if (level > 0) {
                result.append(System.lineSeparator());
            }
        }

        return result.toString();
    }

    public static String formatDate(int date) {
        return String.format("%04d", date);
    }

    public static class Entry {
        private final int key;
        private final String value;

        public Entry(int key, String value) {
            this.key = key;
            this.value = value;
        }

        public int getKey() {
            return key;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return formatDate(key) + ":" + value;
        }
    }
}
