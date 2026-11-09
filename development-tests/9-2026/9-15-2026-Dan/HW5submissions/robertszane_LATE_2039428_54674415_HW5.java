/*
Author: Zane Roberts
Email: zroberts2025@fit.edu
Course: CES2002
Section: 3
Description of this file: Displays the timeline/photos of events a user has attended
*/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Social Network Timeline Manager using a Skip List.
 * Dates are in MMDD format and events are up to 100 characters
 */
public class HW5 {

    /**
     * provides a known sequence of random heights for skip list insertions
     */
    static class FakeRandHeight {
        private static int[] height = {0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4,
                                       0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0}; 
        private int count;  // number of times get() has been called

        /**
         *  Initializes the height sequence counter to zero.
         */
        public FakeRandHeight() {
            count = 0;
        }

        /**
         * Returns the next height number in the sequence, 
         * recycling the sequence automatically if called more than 31 times
         */
        public int get() {
            return height[count++ % 31];
        }
    }

    /**
     * Node class representing an entry in the Skip List, containing a key (date), 
     * a value (event), and forward pointers for multiple levels.
     */
    static class Node {
        String key;
        String value;
        Node[] forward;

        /**
         * Initializes a Skip List node with a specific key, value, 
         * and height (number of levels), allocating space for its forward pointers.
         */
        public Node(String key, String value, int level) {
            this.key = key;
            this.value = value;
            this.forward = new Node[level + 1];
        }
    }

    /**
     * SkipList class implementation
     */
    static class SkipList {
        private static final int MAX_HEIGHT = 10; // Skip list has a height of at most 10[cite: 2]
        private int level;
        private Node header;
        private FakeRandHeight randHeight;

        /**
         * Initializes an empty Skip List with a header node, sets the 
         * initial level to 0, and instantiates the FakeRandHeight generator.
         */
        public SkipList() {
            this.level = 0;
            this.header = new Node(null, null, MAX_HEIGHT);
            this.randHeight = new FakeRandHeight();
        }

        /**
         * Retrieves a deterministic random height for a new node using FakeRandHeight
         */
        public int getRandHeight() {
            int h = randHeight.get();
            return Math.min(h, MAX_HEIGHT - 1);
        }

        /**
         * Searches the skip list for a given key. If the key exists, 
         * it returns the associated value otherwise, it returns null
         */
        public String get(String key) {
            Node curr = header;
            for (int i = level; i >= 0; i--) {
                while (curr.forward[i] != null && curr.forward[i].key.compareTo(key) < 0) {
                    curr = curr.forward[i];
                }
            }
            curr = curr.forward[0];
            if (curr != null && curr.key.equals(key)) {
                return curr.value;
            }
            return null;
        }

        /**
         * Adds a new key-value entry if the key does not exist and returns null; 
         * if the key already exists, replaces its value and returns the old value
         * Uses FakeRandHeight to determine the node's height
         */
        public String put(String key, String value) {
            Node[] update = new Node[MAX_HEIGHT];
            Node curr = header;
            for (int i = level; i >= 0; i--) {
                while (curr.forward[i] != null && curr.forward[i].key.compareTo(key) < 0) {
                    curr = curr.forward[i];
                }
                update[i] = curr;
            }
            curr = curr.forward[0];

            if (curr != null && curr.key.equals(key)) {
                String oldValue = curr.value;
                curr.value = value;
                return oldValue;
            }

            int newLevel = getRandHeight();
            if (newLevel > level) {
                for (int i = level + 1; i <= newLevel; i++) {
                    update[i] = header;
                }
                level = newLevel;
            }

            Node newNode = new Node(key, value, newLevel);
            for (int i = 0; i <= newLevel; i++) {
                newNode.forward[i] = update[i].forward[i];
                update[i].forward[i] = newNode;
            }
            return null;
        }

        /**
         * Removes the entry corresponding to the specified key if it exists, 
         * returning its value; otherwise, returns null
         */
        public String remove(String key) {
            Node[] update = new Node[MAX_HEIGHT];
            Node curr = header;
            for (int i = level; i >= 0; i--) {
                while (curr.forward[i] != null && curr.forward[i].key.compareTo(key) < 0) {
                    curr = curr.forward[i];
                }
                update[i] = curr;
            }
            curr = curr.forward[0];

            if (curr != null && curr.key.equals(key)) {
                for (int i = 0; i <= level; i++) {
                    if (update[i].forward[i] != curr) break;
                    update[i].forward[i] = curr.forward[i];
                }
                while (level > 0 && header.forward[level] == null) {
                    level--;
                }
                return curr.value;
            }
            return null;
        }

        /**
         * Returns the entry with the smallest key greater than or equal to the 
         * specified key; returns null if no such entry exists
         */
        public Node ceilingEntry(String key) {
            Node curr = header;
            for (int i = level; i >= 0; i--) {
                while (curr.forward[i] != null && curr.forward[i].key.compareTo(key) < 0) {
                    curr = curr.forward[i];
                }
            }
            return curr.forward[0];
        }

        /**
         * Returns a list of all entries whose keys fall within the
         * range from key1 to key2
         */
        public List<Node> subMap(String key1, String key2) {
            List<Node> result = new ArrayList<>();
            Node curr = ceilingEntry(key1);
            while (curr != null && curr.key.compareTo(key2) <= 0) {
                result.add(curr);
                curr = curr.forward[0];
            }
            return result;
        }

        /**
         * Prints the entire structure of the skip list level by level doen to S0
         */
        public void printSkipList() {
            for (int i = level; i >= 0; i--) {
                System.out.print("(S" + i + ") ");
                Node curr = header.forward[i];
                if (curr == null) {
                    System.out.print("empty");
                } else {
                    while (curr != null) {
                        System.out.print(curr.key + ":" + curr.value);
                        curr = curr.forward[i];
                        if (curr != null) {
                            System.out.print(" ");
                        }
                    }
                }
                System.out.println();
            }
        }
    }

    /**
     * Reads actions from the input file via command-line arguments
     */
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java HW5 <filename>");
            return;
        }

        SkipList skipList = new SkipList();
        String filename = args[0];

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                int firstSpace = line.indexOf(' ');
                String action = firstSpace == -1 ? line : line.substring(0, firstSpace);
                String arguments = firstSpace == -1 ? "" : line.substring(firstSpace + 1).trim();

                processAction(skipList, action, arguments);
            }
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
    }

    /**
     * Parses and processes whats read from the input file, 
     * executing the correct skip list method and printing output in the required format
     */
    public static void processAction(SkipList skipList, String action, String arguments) {
        switch (action) {
            case "DisplayEvent": {
                String date = arguments.trim();
                String event = skipList.get(date);
                if (event != null) {
                    System.out.println("DisplayEvent " + date + " " + event);
                } else {
                    System.out.println("DisplayEvent " + date + " none");
                }
                break;
            }
            case "AddEvent": {
                int spaceIdx = arguments.indexOf(' ');
                String date = spaceIdx == -1 ? arguments : arguments.substring(0, spaceIdx);
                String event = spaceIdx == -1 ? "" : arguments.substring(spaceIdx + 1).trim();
                skipList.put(date, event);
                System.out.println("AddEvent " + date + " " + event + " success");
                break;
            }
            case "DeleteEvent": {
                String date = arguments.trim();
                String removed = skipList.remove(date);
                if (removed != null) {
                    System.out.println("DeleteEvent " + date + " success");
                } else {
                    System.out.println("DeleteEvent " + date + " noDateError");
                }
                break;
            }
            case "DisplayEventsBetweenDates": {
                String[] subParts = arguments.split("\\s+");
                String startDate = subParts[0];
                String endDate = subParts[1];
                List<Node> entries = skipList.subMap(startDate, endDate);
                System.out.print("DisplayEventsBetweenDates " + startDate + " " + endDate + " ");
                if (entries.isEmpty()) {
                    System.out.println("none");
                } else {
                    for (int i = 0; i < entries.size(); i++) {
                        Node node = entries.get(i);
                        System.out.print(node.key + ":" + node.value);
                        if (i < entries.size() - 1) System.out.print(" ");
                    }
                    System.out.println();
                }
                break;
            }
            case "DisplayEventsFromStartDate": {
                String startDate = arguments.trim();
                List<Node> entries = skipList.subMap(startDate, "1231");
                System.out.print("DisplayEventsFromStartDate " + startDate + " ");
                if (entries.isEmpty()) {
                    System.out.println("none");
                } else {
                    for (int i = 0; i < entries.size(); i++) {
                        Node node = entries.get(i);
                        System.out.print(node.key + ":" + node.value);
                        if (i < entries.size() - 1) System.out.print(" ");
                    }
                    System.out.println();
                }
                break;
            }
            case "DisplayEventsToEndDate": {
                String endDate = arguments.trim();
                List<Node> entries = skipList.subMap("0101", endDate);
                System.out.print("DisplayEventsToEndDate " + endDate + " ");
                if (entries.isEmpty()) {
                    System.out.println("none");
                } else {
                    for (int i = 0; i < entries.size(); i++) {
                        Node node = entries.get(i);
                        System.out.print(node.key + ":" + node.value);
                        if (i < entries.size() - 1) System.out.print(" ");
                    }
                    System.out.println();
                }
                break;
            }
            case "DisplayAllEvents": {
                List<Node> entries = skipList.subMap("0101", "1231");
                System.out.print("DisplayAllEvents ");
                if (entries.isEmpty()) {
                    System.out.println("none");
                } else {
                    for (int i = 0; i < entries.size(); i++) {
                        Node node = entries.get(i);
                        System.out.print(node.key + ":" + node.value);
                        if (i < entries.size() - 1) System.out.print(" ");
                    }
                    System.out.println();
                }
                break;
            }
            case "PrintSkipList": {
                System.out.println("PrintSkipList");
                skipList.printSkipList();
                break;
            }
            default:
                break;
        }
    }
}
