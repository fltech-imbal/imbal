/*

  Author: Milena Koullapi
  Email: mkoullapi2025@my.fit.edu
  Course: CSE 2010
  Section: 01 
  Description of this file: Implements a SkipListMap data structure to store and manage event timelines,
    supporting efficient search, insertion, deletion, and range-based event queries.

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;

public class HW5 {
  
  /*
  * This method reads the action file from command-line arguments, initializes the SkipListMap 
  * and random height generator, and processes each command to manage events.
  * @param args - command-line arguments; args[0] is the path to the actions file.
  */

  public static void main(String[] args) {
    // Filename stores the path of the file that is going to be read from command-line argument args[0]
    String filename = args[0];
    SkipListMap skipList = new SkipListMap();
    FakeRandHeight randHeight = new FakeRandHeight();

    // Opens the file and reads it line by line
    try (Scanner scanner = new Scanner(new File(filename))) {
      while (scanner.hasNextLine()) {
        String line = scanner.nextLine().trim();
        if (line.isEmpty()) continue;

          String[] parts = line.split("\\s+", 3);
          String command = parts[0];

          // Adds a new event date and description to the skip list
          if (command.equals("AddEvent")) {
            String date = parts[1];
            String event = parts[2];
            skipList.put(date, event, randHeight);
            System.out.println("AddEvent " + date + " " + event + " success");

          // Displays the event description for a given date, or none if it doesn't exist
          } else if (command.equals("DisplayEvent")) {
            String date = parts[1];
            String val = skipList.get(date);
            if (val != null) {
              System.out.println("DisplayEvent " + date + " " + val);
            } else {
              System.out.println("DisplayEvent " + date + " none");
            }

          // Removes an event by its date from the skip list
          } else if (command.equals("DeleteEvent")) {
            String date = parts[1];
            String val = skipList.remove(date);
              if (val != null) {
                System.out.println("DeleteEvent " + date + " success");
              } else {
                System.out.println("DeleteEvent " + date + " noDateError");
              }

          // Prints all levels of the skip list from top to bottom
          } else if (command.equals("PrintSkipList")) {
            System.out.println("PrintSkipList");
            skipList.printSkipList();

          // Displays all events falling between two given dates
          } else if (command.equals("DisplayEventsBetweenDates")) {
            String start = parts[1];
            String end = parts[2];
            ArrayList<Entry> entries = skipList.subMap(start, end);
            System.out.print("DisplayEventsBetweenDates " + start + " " + end);
              if (entries.isEmpty()) {
                System.out.println(" none");
              } else {
                for(int i = 0; i < entries.size(); i++){
                  System.out.print(" " + entries.get(i).key + ":" + entries.get(i).value);
                }
                System.out.println();
              }

          // Displays all events starting from a given date onwards
          } else if (command.equals("DisplayEventsFromStartDate")) {
            String start = parts[1];
            ArrayList<Entry> entries = skipList.fromStartDate(start);
            System.out.print("DisplayEventsFromStartDate " + start);
              if (entries.isEmpty()) {
                System.out.println(" none");
              } else {
                for(int i = 0; i < entries.size(); i++){
                  System.out.print(" " + entries.get(i).key + ":" + entries.get(i).value);
                }
                System.out.println();
              }

          // Displays all events up to a given end date
          } else if (command.equals("DisplayEventsToEndDate")) {
            String end = parts[1];
            ArrayList<Entry> entries = skipList.toEndDate(end);
            System.out.print("DisplayEventsToEndDate " + end);
            if (entries.isEmpty()) {
              System.out.println(" none");
            } else {
              for(int i = 0; i < entries.size(); i++){
                System.out.print(" " + entries.get(i).key + ":" + entries.get(i).value);
              }
              System.out.println();
            }

          // Displays all stored events in chronological order
          } else if (command.equals("DisplayAllEvents")) {
            ArrayList<Entry> entries = skipList.subMap("0000", "9999");
            System.out.print("DisplayAllEvents");
            if (entries.isEmpty()) {
              System.out.println(" none");
            } else {
              for(int i = 0; i < entries.size(); i++){
                System.out.print(" " + entries.get(i).key + ":" + entries.get(i).value);
              }
              System.out.println();
            }
          }
      }
    // This block runs if the actions file cannot be found or opened
    } catch (FileNotFoundException e) {
      System.out.println("File not found.");
      e.printStackTrace();
    }
  }

    // Represents a key-value entry holding a date key and an event value string.
    public static class Entry {
      public String key;
      public String value;

      public Entry(String k, String v) {
        key = k;
        value = v;
      }
    }

    // Represents a node in the multi-level skip list with directional pointers.
    public static class Node {
      public String key;
      public String value;
      public Node next;
      public Node prev;
      public Node down;
      public Node up;

      public Node(String k, String v) {
        key = k;
        value = v;
      }
    }

    // Implements the Skip-List Map data structure with multi-level linkage and sentinel boundary checks.
    public static class SkipListMap {
      private Node head;
      private Node tail;
      private int height;

      // Helper method to check if a node is the negative infinity sentinel
      private boolean isStart(Node n) { return n.key.equals("-INF"); }

      // Helper method to check if a node is the positive infinity sentinel
      private boolean isEnd(Node n)   { return n.key.equals("+INF"); }

      // Constructor to initialize an empty skip list with head and tail sentinels at level 0.
      public SkipListMap() {
        head = new Node("-INF", null);
        tail = new Node("+INF", null);
        head.next = tail;
        tail.prev = head;
        height = 0;
      }

      /*
        * Searches for the position of a key starting from the top-left corner and dropping down.
        * @param k - key to search for
        * @return node at level 0 with the largest key less than or equal to k
      */
      private Node search(String k) {
        Node p = head;
        while (p.down != null) {
          p = p.down;
          while (!isEnd(p.next) && p.next.key.compareTo(k) <= 0) {
            p = p.next;
          }
        }
          return p;
      }

      /*
        * Retrieves the value associated with a given key, or null if it does not exist.
        * @param k - key to look up
        * @return event value string, or null
      */
      public String get(String k) {
        Node p = search(k);
        if (!isStart(p) && p.key.equals(k)) {
          return p.value;
        }
          return null;
      }

      /*
        * Returns the entry with the largest key less than or equal to the given key.
        * @param k - target key
        * @return Entry object, or null if none exists
      */
      public Entry floorEntry(String k) {
        Node p = search(k);
        if (!isStart(p)) {
          return new Entry(p.key, p.value);
        }
         return null;
      }

      /*
        * Returns the entry with the smallest key greater than or equal to the given key.
        * @param k - target key
        * @return Entry object, or null if none exists
      */
      public Entry ceilingEntry(String k) {
        Node p = search(k);
        if (!isStart(p) && p.key.equals(k)) {
          return new Entry(p.key, p.value);
        }
        Node s0 = search(k);
        Node candidate = s0.next;
        if (!isEnd(candidate)) {
          return new Entry(candidate.key, candidate.value);
        }
          return null;
      }

      /*
        * Inserts a key-value pair into the skip list with a randomly determined height.
        * @param k - key
        * @param v - value
        * @param randHeight - height generator utility
        * @return old value if key already existed, or null if newly added
       */
      public String put(String k, String v, FakeRandHeight randHeight) {
        Node p = search(k);
        if (!isStart(p) && p.key.equals(k)) {
          String oldVal = p.value;
          Node curr = p;
          while (curr != null) {
            curr.value = v;
            curr = curr.up;
          }
            return oldVal;
          }

          int newHeight = randHeight.get();

          // Add new empty levels at the top if necessary
          while (height <= newHeight) {
            height++;
            Node newHead = new Node("-INF", null);
            Node newTail = new Node("+INF", null);
            newHead.next = newTail;
            newTail.prev = newHead;

            newHead.down = head;
            head.up = newHead;
            

            newTail.down = tail;
            tail.up = newTail;

            head = newHead;
            tail = newTail;
          }

          // Find predecessor nodes at each level
          Node[] preds = new Node[height + 1];
          Node trav = head;
          int lvl = height;
          while (lvl >= 0) {
            while (!isEnd(trav.next) && trav.next.key.compareTo(k) < 0) {
              trav = trav.next;
            }
            preds[lvl] = trav;
            trav = trav.down;
            lvl--;
          }

          // Insert new nodes across levels and link them vertically
          Node downNode = null;
          for (int i = 0; i <= newHeight; i++) {
            Node succ = preds[i].next;
            Node newest = new Node(k, v);
                
            newest.prev = preds[i];
            newest.next = succ;
            preds[i].next = newest;
            succ.prev = newest;

              if (downNode != null) {
                newest.down = downNode;
                downNode.up = newest;
              }
                downNode = newest;
          }

          return null;
      }

      /*
        * Removes an entry matching the given key from all levels of the skip list.
        * @param k - key to remove
        * @return removed value string, or null if key was not found
      */
      public String remove(String k) {
        Node p = search(k);
        if (isStart(p) || !p.key.equals(k)) {
          return null;
        }

        String val = p.value;
        Node curr = p;
        while (curr != null) {
          Node prevNode = curr.prev;
          Node nextNode = curr.next;
          prevNode.next = nextNode;
          nextNode.prev = prevNode;
          curr = curr.up;
        }

        // Trim any extra empty top levels
        while (height > 0 && head.next == tail && head.down.next == tail.down) {
          head = head.down;
          head.up = null;
          tail = tail.down;
          tail.up = null;
          height--;
        }

        return val;
      }

      /*
        * Returns all entries within a specified key range [k1, k2].
        * @param k1 - start key
        * @param k2 - end key
        * @return ArrayList of Entry objects
      */
      public ArrayList<Entry> subMap(String k1, String k2) {
        ArrayList<Entry> result = new ArrayList<>();
        Node p = search(k1);
        Node curr;
        if(isStart(p)){
          curr = p.next;
        } else if(p.key.compareTo(k1) >= 0){
          curr = p;
        } else{
          curr = p.next;
        }


        while (!isEnd(curr) && curr.key.compareTo(k2) <= 0) {
          result.add(new Entry(curr.key, curr.value));
          curr = curr.next;
        }
        return result;
      }

      /*
        * Returns all entries from a start date onwards.
        * @param start - start date key
        * @return ArrayList of Entry objects
      */
      public ArrayList<Entry> fromStartDate(String start) {
        return subMap(start, "9999");
      }

      /*
        * Returns all entries up to a specified end date.
        * @param end - end date key
        * @return ArrayList of Entry objects
      */
      public ArrayList<Entry> toEndDate(String end) {
        return subMap("0000", end);
      }

      //Prints the entire skip list structure from top level down to level 0.
      public void printSkipList() {
        Node lvlHead = head;
        int currLevel = height;

        while (lvlHead != null) {
          System.out.print("(S" + currLevel + ")");
          Node walk = lvlHead.next;
          boolean isEmpty = true;
          StringBuilder sb = new StringBuilder();

          while (!isEnd(walk)) {
            isEmpty = false;
            sb.append(" ").append(walk.key).append(":").append(walk.value);
            walk = walk.next;
          }

          if (isEmpty) {
            System.out.println(" empty");
          } else {
            System.out.println(sb.toString());
          }

          lvlHead = lvlHead.down;
          currLevel--;
        }
      }
  }
}