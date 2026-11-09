/**
 Author: Joshua Davis
 Email: joshuadavis2025@my.fit.edu
 Course: CSE 2010 - Algorithms & Data Structures
 Section: 4

 Description:
 This Program implements a skip list to map out dates across the year
 */
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class HW5 {
    //Main method calls every other method to do the operations required
    public static void main(String[] args) throws FileNotFoundException {
        //Declaration of File and the scanner to read the file
        File inputFile = new File(args[0]);
        Scanner input = new Scanner(inputFile);
        //Here a skiplist object is made
        SkipList list = new SkipList();

        //while there is a next line read the next line save the event and choose the case that matches the command and
        //execute
        while (input.hasNextLine()) {
            String line = input.nextLine();

            Scanner lineScanner = new Scanner(line);
            String command = lineScanner.next();

            switch (command) {
                case "AddEvent": {
                    String date = lineScanner.next();
                    String event = lineScanner.nextLine().trim();

                    list.put(date, event);

                    System.out.println("AddEvent " + date + " " + event + " success");
                    break;
                }
                case "DisplayEvent": {
                    String date = lineScanner.next();
                    String event = list.get(date);

                    if (event == null) {
                        System.out.println("DisplayEvent " + date + " none");
                    } else {
                        System.out.println("DisplayEvent " + date + " " + event);
                    }
                    break;
                }
                case "DeleteEvent": {
                    String date = lineScanner.next();
                    String removedEvent = list.remove(date);

                    if (removedEvent == null) {
                        System.out.println("DeleteEvent " + date + " noDateError");
                    } else {
                        System.out.println("DeleteEvent " + date + " success");
                    }
                    break;
                }
                case "PrintSkipList": {
                    System.out.println("PrintSkipList");
                    list.printSkipList();
                    break;
                }
                case "DisplayAllEvents": {
                    System.out.println("DisplayAllEvents "
                            + list.formatEvents(list.subMap("0000", "9999")));
                    break;
                }

                case "DisplayEventsBetweenDates": {
                    String startDate = lineScanner.next();
                    String endDate = lineScanner.next();

                    System.out.println("DisplayEventsBetweenDates "
                            + startDate + " " + endDate + " "
                            + list.formatEvents(list.subMap(startDate, endDate)));
                    break;
                }

                case "DisplayEventsFromStartDate": {
                    String startDate = lineScanner.next();

                    System.out.println("DisplayEventsFromStartDate "
                            + startDate + " "
                            + list.formatEvents(list.subMap(startDate, "9999")));
                    break;
                }

                case "DisplayEventsToEndDate": {
                    String endDate = lineScanner.next();

                    System.out.println("DisplayEventsToEndDate "
                            + endDate + " "
                            + list.formatEvents(list.subMap("0000", endDate)));
                    break;
                }
            }
        }

        input.close();
    }
}
//The skip List class with the the the nested Node class, sentinels are the beginning of every level of the skiplist
class SkipList {
    //Fields of SkipList class
    private Node sentinel;
    private int currentHeight;
    private FakeRandHeight randHeight;

    public class Node {
        //Fields of Node class
        String key;
        String value;
        Node[] next;

        //Nodes constructor
        Node(String key, String value, int height) {
            this.key = key;
            this.value = value;

            this.next = new Node[height + 1];
        }

        Node(int maxHeight) {
            this(null, null, maxHeight);
        }
    }
    //SkipList constuctor
    SkipList() {
        this.sentinel = new Node(10);
        this.currentHeight = 0;
        this.randHeight = new FakeRandHeight();
    }
    //this is the get method for a node in a skip list, start with current at a sentinel and your starting level at the
    //current height. while your starting level is greater than 0 if you can move right in ther skip list move right, otherwise
    //move down until you find your target
    public String get(String key) {
        Node current = sentinel;
        int startingLevel = currentHeight;

        while (startingLevel >= 0) {
            if (current.next[startingLevel] != null && current.next[startingLevel].key.compareTo(key) < 0) {
                current = current.next[startingLevel];
                continue;
            }
            else if (current.next[startingLevel] != null && current.next[startingLevel].key.equals(key)) {
                return current.next[startingLevel].value;
            }
            startingLevel--;
        }
        return null;
    }
    //The put method inserts a node where it belongs, at every level it finds the node that should be directly before it
    //and saves it, then does the same until it reaches the bottom, then every previouses next pointer gets updated and the
    //and the insertion node points to the old next of the previous node
    public String put(String key, String value) {
        Node[] update = new Node[11];

        Node current = sentinel;
        int startingLevel = currentHeight;

        while (startingLevel >= 0) {
            while (current.next[startingLevel] != null
                    && current.next[startingLevel].key.compareTo(key) < 0) {
                current = current.next[startingLevel];
            }
            update[startingLevel] = current;
            startingLevel--;
        }

        if (current.next[0] != null && current.next[0].key.equals(key)) {
            String oldValue = current.next[0].value;
            current.next[0].value = value;
            return oldValue;
        }

        int newHeight = randHeight.get();
        Node newNode = new Node(key, value, newHeight);

        if (newHeight > currentHeight) {
            for (int i = currentHeight + 1; i <= newHeight; i++) {
                update[i] = sentinel;
            }
            currentHeight = newHeight;
        }

        for (int i = 0; i <= newHeight; i++) {
            newNode.next[i] = update[i].next[i];
            update[i].next[i] = newNode;
        }

        return null;
    }
    public String remove(String key) {
        Node[] update = new Node[11];

        Node current = sentinel;
        int startingLevel = currentHeight;

        while (startingLevel >= 0) {
            while (current.next[startingLevel] != null
                    && current.next[startingLevel].key.compareTo(key) < 0) {
                current = current.next[startingLevel];
            }

            update[startingLevel] = current;
            startingLevel--;
        }

        if (current.next[0] == null || !current.next[0].key.equals(key)) {
            return null;
        }

        Node target = current.next[0];
        String oldValue = target.value;

        for (int i = 0; i <= currentHeight; i++) {
            if (update[i].next[i] == target) {
                update[i].next[i] = target.next[i];
            }
        }

        while (currentHeight > 0 && sentinel.next[currentHeight] == null) {
            currentHeight--;
        }

        return oldValue;
    }
    //This method prints the skip list it the required format at each level of the skip list
    public void printSkipList() {
        for (int i = currentHeight + 1; i >= 0; i--) {
            Node current = sentinel.next[i];
            if (current == null) {
                System.out.println("(S" + i + ") empty");
            } else {
                System.out.print("(S" + i + ") ");
                while (current != null) {
                    System.out.print(current.key + ":" + current.value);

                    current = current.next[i];

                    if (current != null) {
                        System.out.print(" ");
                    }
                }
                System.out.println();
            }
        }
    }
    //this method given a key, find the keys node and then returns the node right before it
    public Node floorEntry(String key) {
        Node current = sentinel;
        int startingLevel = currentHeight;

        while (startingLevel >= 0) {
            while (current.next[startingLevel] != null
            && current.next[startingLevel].key.compareTo(key) <= 0) {
                current = current.next[startingLevel];
            }
            startingLevel--;
        }

        return current == sentinel ? null : current;
    }
    //this method given a key, find the keys node and returns the node right after it
    public Node ceilingEntry(String key) {
        Node current = sentinel;
        int startingLevel = currentHeight;

        while (startingLevel >= 0) {
            while (current.next[startingLevel] != null
            && current.next[startingLevel].key.compareTo(key) < 0) {
                current = current.next[startingLevel];
            }
            startingLevel--;
        }
        return current.next[0];
    }
    //Given two bounds the method finds every event between these bounds including the bounds themselves
    public ArrayList<Node> subMap(String startKey, String endKey) {
        ArrayList<Node> results = new ArrayList<>();

        Node  current = ceilingEntry(startKey);
        while (current != null
        && current.key.compareTo(endKey) <= 0) {
            results.add(current);
            current = current.next[0];
        }
        return results;
    }
    //this is a helper method to make sure everything is in the desired format with the output
    public String formatEvents(ArrayList<Node> events) {
        if (events.isEmpty()) {
            return "none";
        }

        StringBuilder result = new StringBuilder();

        for (Node event : events) {
            if (result.length() > 0) {
                result.append(" ");
            }

            result.append(event.key).append(":").append(event.value);
        }

        return result.toString();
    }
}
