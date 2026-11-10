/*

  Author: Joseph Venech
  Email: jvenech2025@my.fit.edu
  Course: CSE2010
  Section: 1
  Description of this file: implements skip list that is able to search, insert, remove, and retrieve entries within given range.

 */


import java.util.ArrayList;
import java.util.List;

public class SkipListMap<K extends Comparable<K>, V> {

    //Max number of levels allowed
    private static final int MAX_HEIGHT = 10;

    //Stores one key value pair
    //Key = date, value = event
    public static class Entry<K, V> {
        private final K key;
        private final V value;

        //Creates entry with given key and value
        public Entry(K key, V value) {
            this.key = key;
            this.value = value;
        }
        
        //returns key of this entry
        public K getKey() {
            return key;
        }

        //returns value of this entry
        public V getValue() {
            return value;
        }
    }

    //Stores key value pair and refereneces to next node at each level
    private class Node {
        K key;
        V value;
        List<Node> next;

        //Creates a node with references for levels 0 through height
        Node(K key, V value, int height) {
            this.key = key;
            this.value = value;
            next = new ArrayList<>();

            //Initalize each forward reference as empty
            for (int i = 0; i <= height; i++) {
                next.add(null);
            }
        }
    }

    //starting point for every level
    private final Node head;
    // generate node heights
    private final FakeRandHeight randHeight;
    //Highest level currently
    private int currentHeight;
    //Number of events stored
    private int size;

    //Creates empty skip list
    public SkipListMap() {
        head = new Node(null, null, MAX_HEIGHT - 1);
        randHeight = new FakeRandHeight();
        currentHeight = 0;
        size = 0;
    }

    //Compares two keys for sorted order
    private int compare(K a, K b) {
        return a.compareTo(b);
    }

    //returns size
    public int size() {
        return size;
    }

    //Searches for key
    //Forward when next is smaller than destination
    //Down when greater
    public V get(K key) {
        Node node = head;

        for (int level = currentHeight; level >= 0; level--) {
            while (node.next.get(level) != null && compare(node.next.get(level).key, key) < 0) {
                node = node.next.get(level);
            }
        }
        
        //Check the next node at the bottom level
        node = node.next.get(0);

        if (node != null && compare(node.key, key) == 0) {
            return node.value;
        }

        return null;
    }

    //insert a new entry
    public V put(K key, V value) {
        List<Node> update = new ArrayList<>();

        //prepare a positon at every level
        for (int i = 0; i < MAX_HEIGHT; i++) {
            update.add(null);
        }

        Node node = head;

        //Find the insertion positon at each active level
        for (int level = currentHeight; level >= 0; level--) {
            while (node.next.get(level) != null && compare(node.next.get(level).key, key) < 0) {
                node = node.next.get(level);
            }

            update.set(level, node);
        }

        node = node.next.get(0);
    
        //Replace value if already exists
        if (node != null && compare(node.key, key) == 0) {
            V oldValue = node.value;
            node.value = value;
            return oldValue;
        }

        //Generate height
        int height = Math.min(randHeight.get(), MAX_HEIGHT - 1);

        if (height > currentHeight) {
            for (int level = currentHeight + 1;level <= height; level++) {
                update.set(level, head);
            }

            currentHeight = height;
        }

        //Create the node and connect it at each of its levels
        Node newNode = new Node(key, value, height);

        for (int level = 0; level <= height; level++) {
            newNode.next.set(level, update.get(level).next.get(level));
            update.get(level).next.set(level, newNode);
        }

        size++;
        return null;
    }

    //Removes an entry
    public V remove(K key) {
        List<Node> update = new ArrayList<>();

        for (int i = 0; i < MAX_HEIGHT; i++) {
            update.add(null);
        }

        Node node = head;

        //Locate node before the target
        for (int level = currentHeight; level >= 0; level--) {
            while (node.next.get(level) != null && compare(node.next.get(level).key, key) < 0) {
                node = node.next.get(level);
            }

            update.set(level, node);
        }

        node = node.next.get(0);
        
        //Return null if does not exist
        if (node == null || compare(node.key, key) != 0) {
            return null;
        }
        
        //Bypass the removed node at every level it is on
        for (int level = 0; level < node.next.size(); level++) {
            if (update.get(level).next.get(level) == node) {
                update.get(level).next.set(level, node.next.get(level));
            }
        }

        //Lower the active height if the top levels become empty
        while (currentHeight > 0 && head.next.get(currentHeight) == null) {
            currentHeight--;
        }

        size--;
        return node.value;
    }

    //Finds entry with largest key less than or equal to requested key
    public Entry<K, V> floorEntry(K key) {
        Node node = head;

        //Move forward as long as next does not exceed key
        for (int level = currentHeight; level >= 0; level--) {
            while (node.next.get(level) != null && compare(node.next.get(level).key, key) <= 0) {
                node = node.next.get(level);
            }
        }

        //If never left the head, no floor entry exists
        if (node == head) {
            return null;
        }

        return new Entry<>(node.key, node.value);
    }

    //Finds the smallest key greater than or equal to the requested key
    public Entry<K, V> ceilingEntry(K key) {
        Node node = head;

        //Stop before the first key greater than or equal to key
        for (int level = currentHeight; level >= 0; level--) {
            while (node.next.get(level) != null && compare(node.next.get(level).key, key) < 0) {
                node = node.next.get(level);
            }
        }

        //The next bottom-level node is the ceiling
        node = node.next.get(0);

        if (node == null) {
            return null;
        }

        return new Entry<>(node.key, node.value);
    }

    //Returns all entries whose keys are between key1 and key2
    public Iterable<Entry<K, V>> subMap(K key1, K key2) {
        List<Entry<K, V>> result = new ArrayList<>();

        //inverted range thus no entries
        if (compare(key1, key2) > 0) {
            return result;
        }

        //First entri that is at least key1
        Entry<K, V> first = ceilingEntry(key1);

        if (first == null) {
            return result;
        }

        Node node = head;

        //Get to first entry's position
        for (int level = currentHeight; level >= 0; level--) {
            while (node.next.get(level) != null && compare(node.next.get(level).key, first.getKey()) < 0) {
                node = node.next.get(level);
            }
        }

        node = node.next.get(0);

        //Collect entries until upper endpoint is exceeded
        while (node != null && compare(node.key, key2) <= 0) {
            result.add(new Entry<>(node.key, node.value));
            node = node.next.get(0);
        }

        return result;
    }

    //Print each skiplist level top to bottom
    public void printSkipList() {
        int topLevel = Math.max(3, currentHeight);

        for (int level = topLevel; level >= 0; level--) {
            System.out.print("(S" + level + ")");

            Node node = head.next.get(level);

            //Print empty if no entries
            if (node == null) {
                System.out.println(" empty");
                continue;
            }

            System.out.print(" ");

            //Print entries in sorted order at this level
            while (node != null) {
                System.out.print(node.key + ":" + node.value);

                node = level < node.next.size()
                        ? node.next.get(level) : null;

                if (node != null) {
                    System.out.print(" ");
                }
            }

            System.out.println();
        }
    }
}