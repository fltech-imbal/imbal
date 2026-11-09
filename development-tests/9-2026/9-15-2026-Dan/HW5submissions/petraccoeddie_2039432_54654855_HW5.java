/**
 * Eddie Petracco III
 * epetracco2025@my.fit.edu
 * CSE 2010
 * 9:30 am Lab Section
 * Using a skip list, creates a timeline of inputted events. This program can 
 * properly navigate the timeline with given input commands and eventually print 
 * the full skiplist as well. The Doubly Linked List Implementation was given along
 * with a randomheight code to help make testing easier.
 */

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class HW5 {
    
    /**
     * A basic doubly linked list implementation.
     *
     * @author Michael T. Goodrich
     * @author Roberto Tamassia
     * @author Michael H. Goldwasser
     */
    public static class DoublyLinkedList<E> {
    
      //---------------- nested Node class ----------------
      private static class Node<E> {
        private E element;               
        private Node<E> prev;            
        private Node<E> next;            
    
        public Node(E e, Node<E> p, Node<E> n) {
          element = e;
          prev = p;
          next = n;
        }
    
        public E getElement() { return element; }
        public Node<E> getPrev() { return prev; }
        public Node<E> getNext() { return next; }
        public void setPrev(Node<E> p) { prev = p; }
        public void setNext(Node<E> n) { next = n; }
      } 
      //----------- end of nested Node class -----------
    
      private Node<E> header;                    
      private Node<E> trailer;                   
      private int size = 0;                      
    
      public DoublyLinkedList() {
        header = new Node<>(null, null, null);      
        trailer = new Node<>(null, header, null);   
        header.setNext(trailer);                    
      }
    
      public int size() { return size; }
      public boolean isEmpty() { return size == 0; }
    
      public E first() {
        if (isEmpty()) return null;
        return header.getNext().getElement();   
      }
    
      public E last() {
        if (isEmpty()) return null;
        return trailer.getPrev().getElement();    
      }
    
      public void addFirst(E e) {
        addBetween(e, header, header.getNext());    
      }
    
      public void addLast(E e) {
        addBetween(e, trailer.getPrev(), trailer);  
      }
    
      public E removeFirst() {
        if (isEmpty()) return null;                  
        return remove(header.getNext());             
      }
    
      public E removeLast() {
        if (isEmpty()) return null;                  
        return remove(trailer.getPrev());            
      }
    
      private void addBetween(E e, Node<E> predecessor, Node<E> successor) {
        Node<E> newest = new Node<>(e, predecessor, successor);
        predecessor.setNext(newest);
        successor.setPrev(newest);
        size++;
      }
    
      private E remove(Node<E> node) {
        Node<E> predecessor = node.getPrev();
        Node<E> successor = node.getNext();
        predecessor.setNext(successor);
        successor.setPrev(predecessor);
        size--;
        return node.getElement();
      }
    
      public String toString() {
        StringBuilder sb = new StringBuilder("(");
        Node<E> walk = header.getNext();
        while (walk != trailer) {
          sb.append(walk.getElement());
          walk = walk.getNext();
          if (walk != trailer)
            sb.append(", ");
        }
        sb.append(")");
        return sb.toString();
      }
    } 

    /**
     * FakeRandHeight class provided for testing.
     * Simulates a random height distribution for Skip List levels.
     */
    public static class FakeRandHeight {
        static int[] height = {0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 
                               0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0}; 
        int count;  
    
        public FakeRandHeight() {
            count = 0;
        }
    
        // Returns the next number in the sequence
        public int get() {
            return height[count++ % 31];
        }
    }

    /**
     * SkipList implementation that manages the timeline of events.
     * Utilizes quad node approach for up, down, left, and right pointers
     */
    public static class SkipList {
        
        /**
         * SkipNode acts as an entry in the skip list.
         * Contains the key (date) and value (event), along with respective pointers.
         */
        private static class SkipNode {
            String key;
            String value;
            SkipNode up, down, prev, next;
            
            // Constructor to initialize key and value
            public SkipNode(String key, String value) {
                this.key = key;
                this.value = value;
            }
        }

        private SkipNode topLeft;
        private int h; // Height of the skip list
        private FakeRandHeight randHeight;

        /**
         * Initializes an empty Skip List.
         * Creates the bounding negative and positive infinity blocks for the base level.
         */
        public SkipList() {
            h = 0;
            // Negative and positive infinity bounds for the top level
            topLeft = new SkipNode("-oo", null);
            SkipNode topRight = new SkipNode("+oo", null);
            
            // Link them horizontally
            topLeft.next = topRight;
            topRight.prev = topLeft;
            
            randHeight = new FakeRandHeight();
        }

        /**
         * Searches for the node with the largest key less than or equal to target key.
         * Traverses top-down and left-right to locate the bottom-level floor node.
         */
        public SkipNode search(String key) {
            SkipNode p = topLeft;
            
            // Start at the top-left and move right and down through the list
            while (true) {
                // Move right as long as the next key is not +oo and is <= target key
                while (!p.next.key.equals("+oo") && p.next.key.compareTo(key) <= 0) {
                    p = p.next;
                }
                // Drop down a level if possible, else we are at S0
                if (p.down != null) {
                    p = p.down;
                } else {
                    break;
                }
            }
            return p;
        }

        /**
         * Returns the event associated with a specific date key.
         */
        public String get(String key) {
            // Find the floor node and check if the keys match exactly
            SkipNode p = search(key);
            if (p.key.equals(key)) return p.value;
            return null;
        }

        /**
         * Adds an event at the given date, or updates the event if date already exists.
         * Uses FakeRandHeight to determine the node's tower height.
         */
        public String put(String key, String value) {
            SkipNode p = search(key);
            
            // If the date exists, update the event value across all its levels
            if (p.key.equals(key)) {
                String old = p.value;
                SkipNode curr = p;
                while (curr != null) {
                    curr.value = value;
                    curr = curr.up;
                }
                return old;
            }

            int k = randHeight.get();
            
            // If the generated height exceeds the current list height, create new empty top levels
            while (h <= k) {
                SkipNode newTopLeft = new SkipNode("-oo", null);
                SkipNode newTopRight = new SkipNode("+oo", null);
                
                // Link new blocks horizontally
                newTopLeft.next = newTopRight;
                newTopRight.prev = newTopLeft;
                
                // Link them vertically to the old top level
                newTopLeft.down = topLeft;
                topLeft.up = newTopLeft;
                SkipNode tr = topLeft;
                while (tr.next != null) tr = tr.next; // Find old top right
                newTopRight.down = tr;
                tr.up = newTopRight;

                topLeft = newTopLeft;
                h++;
            }

            SkipNode q = null;
            int currentLevel = 0;
            
            // Build the nodes tower from bottom (S0) up to level k
            while (currentLevel <= k) {
                SkipNode newNode = new SkipNode(key, value);
                
                // Insert newNode horizontally after p
                newNode.next = p.next;
                newNode.prev = p;
                p.next.prev = newNode;
                p.next = newNode;

                // Link vertically to the node created on the level below
                if (q != null) {
                    newNode.down = q;
                    q.up = newNode;
                }
                q = newNode;

                // Stop if reached the randomized tower height
                if (currentLevel == k) break;

                // Move left to find a node that has an up connection to the next level
                while (p.up == null) {
                    p = p.prev;
                }
                p = p.up;
                currentLevel++;
            }
            return null;
        }

        /**
         * Removes a date and its event from the Skip List.
         * Also checks the list to ensure there is exactly one empty level at the top.
         */
        public String remove(String key) {
            SkipNode p = search(key);
            
            // Check if the key actually exists
            if (!p.key.equals(key)) return null;

            String oldVal = p.value;
            
            // Move up the tower, deleting the node across on every level
            while (p != null) {
                p.prev.next = p.next;
                p.next.prev = p.prev;
                p = p.up;
            }

            // Checks empty top levels. Keep at least one empty level unless height is 0
            while (h > 0 && topLeft.down.next.key.equals("+oo")) {
                topLeft = topLeft.down;
                topLeft.up = null;
                
                // Disconnect the removed right sentinel vertically
                SkipNode tr = topLeft;
                while (tr.next != null) tr = tr.next;
                tr.up = null;
                h--;
            }
            return oldVal;
        }

        /**
         * Returns the entry with the largest key less than or equal to key.
         */
        public SkipNode floorEntry(String key) {
            SkipNode p = search(key);
            if (p.key.equals("-oo")) return null;
            return p;
        }

        /**
         * Returns the entry with the smallest key greater than or equal to key.
         */
        public SkipNode ceilingEntry(String key) {
            SkipNode p = search(key);
            if (p.key.equals(key)) return p;
            
            // If floor key didn't match, ceiling is the immediate right block
            p = p.next;
            if (p.key.equals("+oo")) return null;
            return p;
        }

        /**
         * Prints the entire skip list horizontally by level, top to bottom.
         */
        public void printSkipList() {
            SkipNode levelStart = topLeft;
            int currentH = h;
            
            // Iterate down through each level
            while (levelStart != null) {
                System.out.print("(S" + currentH + ") ");
                SkipNode p = levelStart.next;
                
                // If level is empty (only has sentinels)
                if (p.key.equals("+oo")) {
                    System.out.println("empty");
                } else {
                    // Iterate horizontally and append all elements
                    StringBuilder sb = new StringBuilder();
                    while (!p.key.equals("+oo")) {
                        sb.append(p.key).append(":").append(p.value).append(" ");
                        p = p.next;
                    }
                    System.out.println(sb.toString().trim());
                }
                
                // Drop down a level
                levelStart = levelStart.down;
                currentH--;
            }
        }
        
        /**
         * Displays all events from the base level
         */
        public void displayAll() {
            SkipNode p = topLeft;
            // Drop to the very bottom left
            while (p.down != null) p = p.down;
            p = p.next;
            
            // If list is completely empty
            if (p.key.equals("+oo")) {
                System.out.print(" none");
                return;
            }
            
            // Print all elements
            while (!p.key.equals("+oo")) {
                System.out.print(" " + p.key + ":" + p.value);
                p = p.next;
            }
        }

        /**
         * Displays events bounded inclusively by start and end dates
         */
        public void displayEventsBetween(String d1, String d2) {
            SkipNode p = ceilingEntry(d1);
            boolean found = false;
            
            // Move from the start date ceiling until exceeding the end date
            while (p != null && p.key.compareTo(d2) <= 0) {
                System.out.print(" " + p.key + ":" + p.value);
                p = p.next;
                found = true;
            }
            if (!found) System.out.print(" none");
        }
        
        /**
         * Displays all events starting from a specific date
         */
        public void displayFrom(String d1) {
            SkipNode p = ceilingEntry(d1);
            boolean found = false;
            
            // Move from the start date ceiling until the end of the list
            while (p != null && !p.key.equals("+oo")) {
                System.out.print(" " + p.key + ":" + p.value);
                p = p.next;
                found = true;
            }
            if (!found) System.out.print(" none");
        }
        
        /**
         * Displays all events up to and including a specific end date
         */
        public void displayTo(String d2) {
            SkipNode p = topLeft;
            // Drop to base level
            while (p.down != null) p = p.down;
            p = p.next; 
            boolean found = false;
            
            // Move from the beginning of the list until exceeding the end date
            while (!p.key.equals("+oo") && p.key.compareTo(d2) <= 0) {
                System.out.print(" " + p.key + ":" + p.value);
                p = p.next;
                found = true;
            }
            if (!found) System.out.print(" none");
        }
    }
    
    /**
     * Main method parses commands from the input file and delegates to SkipList.
     */
    public static void main(String[] args) {
        // Enforce the requirement of passing the input filename
        if (args.length < 1) {
            System.err.println("Please provide an input file name as an argument.");
            return;
        }
        String fileIn = args[0];
        SkipList skipList = new SkipList();
        
        // File reader block
        try (Scanner fileScanner = new Scanner(new File(fileIn))) {
            while (fileScanner.hasNext()) {
                String keyword = fileScanner.next();
                
                // Add an event, print out if it was successful
                if (keyword.equals("AddEvent")) {
                    String date = fileScanner.next();
                    String event = fileScanner.next();
                    System.out.print("AddEvent " + date + " " + event);
                    skipList.put(date, event);
                    System.out.println(" success");
                }
                // Try fetching the event by date
                else if (keyword.equals("DisplayEvent")) {
                    String date = fileScanner.next();
                    System.out.print("DisplayEvent " + date);
                    String val = skipList.get(date);
                    if (val != null) {
                        System.out.println(" " + val);
                    } else {
                        System.out.println(" none");
                    }
                }
                // Remove the event, printing appropriate error if not found
                else if (keyword.equals("DeleteEvent")) {
                    String date = fileScanner.next();
                    System.out.print("DeleteEvent " + date);
                    String val = skipList.remove(date);
                    if (val != null) {
                        System.out.println(" success");
                    } else {
                        System.out.println(" noDateError");
                    }
                }
                // Prints whole list
                else if (keyword.equals("PrintSkipList")) {
                    System.out.println("PrintSkipList");
                    skipList.printSkipList();
                }
                // Display the entire timeline on one line
                else if (keyword.equals("DisplayAllEvents")) {
                    System.out.print("DisplayAllEvents");
                    skipList.displayAll();
                    System.out.println();
                }
                // Displays from one date to another using the skiplist and outputs
                else if (keyword.equals("DisplayEventsBetweenDates")) {
                    String date1 = fileScanner.next();
                    String date2 = fileScanner.next();
                    System.out.print("DisplayEventsBetweenDates " + date1 + " " + date2);
                    skipList.displayEventsBetween(date1, date2);
                    System.out.println();
                }
                // Display all future events including the start date
                else if (keyword.equals("DisplayEventsFromStartDate")) {
                    String date = fileScanner.next();
                    System.out.print("DisplayEventsFromStartDate " + date);
                    skipList.displayFrom(date);
                    System.out.println();
                }
                // Display all past events bounded by an end date
                else if (keyword.equals("DisplayEventsToEndDate")) {
                    String date = fileScanner.next();
                    System.out.print("DisplayEventsToEndDate " + date);
                    skipList.displayTo(date);
                    System.out.println();
                }
            }  
        } catch (FileNotFoundException e) {
            System.err.println("File name: " + fileIn + " was not found!"); 
        }
    }
}