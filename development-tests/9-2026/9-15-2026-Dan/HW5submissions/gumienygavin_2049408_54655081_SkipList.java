
/**
 * Author: Gavin Gumieny
   Email: ggumieny2025@my.fit.edu
   Course: CSE 2010
   Section: 4
   Description of this file:
   This file implements a skip list that is used to store dates (keys) and events (values).
   It keeps track of and manages the timeline of these events. This allows for the access of
   specific events within a given time range. The information is stored and linked in Quad-Nodes
   that contain pointers that point to nodes in front, behind, above, or below a specific node.
   This file goes with HW5.java which calls the methods of this file based on the appropriate
   action. This file also uses FakeRandHeight.java to generate predetermined heights in the skip
   list.
   
 * 
 */
import java.util.ArrayList;
import java.util.List;

public class SkipList
{
    /** A Quad-Node of a doubly linked list, which stores references to a date and an
     *  event, to previous and next nodes in the list, and to nodes below and above
     *  the node.
     */
    public static class Node {
        
        private String date;            // The date stored at this node.
        private String event;           // The event stored at this node.
        private Node prev;              // The previous node in the list.
        private Node next;              // The next node in the list.
        private Node below;             // The node below this node.
        private Node above;             // The node above this node.
        
        /** Creates a node with a given date and event.
         * 
         *  @param date  the date to be stored
         *  @param event  the event to be stored
         * 
         */
        public Node(String date, String event) {
            this.date = date;
            this.event = event;
        }
        
        // Public accessor methods
        /** Returns the date stored at this node.
         *  @return the date stored at this node
         */
        public String getDate() {
            return this.date;
        }
        
        /** Returns the event stored at this node.
         *  @return the event stored at this node
         */
        public String getEvent() {
            return this.event;
        }
        
        /** Returns the node that precedes this one (or null if none).
         *  @return the preceding node
         */
        public Node getPrev() {
            return this.prev;
        }
        
        /** Returns the node that follows this one (or null if none).
         *  @return the following node
         */
        public Node getNext() {
            return this.next;
        }
        
        /** Returns the node that exists below this one (or null if none).
         *  @return the node below
         */
        public Node getBelow() {
            return this.below;
        }
        
        /** Returns the node that exists above this one (or null if none).
         *  @return the node above
         */
        public Node getAbove() {
            return this.above;
        }
        
        // Update methods
        /** Sets the date stored at this node.
         *  @param date  the date to be set
         */
        public void setDate(String date) {
            this.date = date;
        }
        
        /** Sets the event stored at this node.
         *  @param event  the event to be set
         */
        public void setEvent(String event) {
            this.event = event;
        }
        
        /** Sets the node's previous reference to point to node n.
         *  @param prev  the node that should precede this one
         */
        public void setPrev(Node prev) {
            this.prev = prev;
        }
        
        /** Sets the node's next reference to point to node n.
         *  @param prev  the node that should follow this one
         */
        public void setNext(Node next) {
            this.next = next;
        }
        
        /** Sets the node's below reference to point to node n.
         *  @param prev  the node that should exist below this one
         */
        public void setBelow(Node below) {
            this.below = below;
        }
        
        /** Sets the node's above reference to point to node n.
         *  @param prev  the node that should exist above this one
         */
        public void setAbove(Node above) {
            this.above = above;
        }
    }
    
    // instance variables
    private Node head;          // stores the head
    private Node tail;          // stores the tail
    private int height;         // height of the skip list
    private static final int MAX_HEIGHT = 10;   // the maximum height the skip list can be
    
    // Fixed random height generator
    private final FakeRandHeight randHeight = new FakeRandHeight();
    
    // Positive and negative infinity
    public static final String NEG_INF = "0000";
    public static final String POS_INF = "9999";
    
    /** Constructs a new empty list. */
    public SkipList() {
        head = new Node(NEG_INF, null);
        tail = new Node(POS_INF, null);
        head.setNext(tail);
        tail.setPrev(head);
        height = 1;
    }
    
    // public accessor methods
    /**
     * Returns the height of skip list.
     * @return height of skip list
     */
    public int height() { return height; }


    
    /** Finds the entry node with a given key. If none is found then the entry less than k is returned.
     *  @param k  the key
     *  @return   the node with the key (or largest key less than k)
     */
    public Node findEntry(String k) {
        Node p = head;          // Start at top lists head
        
        while (true) {
            // Scan forward if the next node's key is less than or equal to k
            while (p.getNext().getDate() != POS_INF && p.getNext().getDate().compareTo(k) <= 0) {
                p = p.getNext();    // move right while the next node's key is not greater than k
            }
            
            // Drop down if a level exists
            if (p.getBelow() != null) {
                p = p.getBelow();
            }
            else {
                break;
            }
        }
        return p;
    }
    
    /** Returns the value associated with a key (if one exists).
     *  @param k  the key
     *  @return   the value associated with the key (or null if none)
     */
    public String get(String k) {
        Node p = findEntry(k);      // find the node
        
        // If the keys are equal (the dates), return the value (the event).
        if (p.getDate().equals(k)) {
            return p.getEvent();
        }
        
        return null;
    }
    
    /** Adds an entry to the list (or replaces an existing value).
     *  @param k  the key
     *  @param v  the value
     *  @return   null if a key doesn't already exist (or the old value if we are replacing)
     */
    public String put(String k, String v) {
        Node p = findEntry(k);      // find the node
        
        // Already exists, return old value
        if (p.getDate().equals(k)) {
            String old = p.getEvent();
            
            // Replace each value of p (same height is kept so we go until p is set to null)
            while (p != null) {
                p.setEvent(v);
                p = p.getAbove();
            }
            return old;
        }
        
        // New entry
        // Get our random height value and check that it's below the max assumed height
        int randomHeight = this.randHeight.get();
        if (randomHeight > MAX_HEIGHT) randomHeight = MAX_HEIGHT;
        
        // While the current height is less than the random height we add each level of the skip list
        while (height <= randomHeight + 1) {
            // Set up the new empty list.
            Node newHead = new Node(NEG_INF, null);
            Node newTail = new Node(POS_INF, null);
            
            // Link the negative and positive infinity.
            newHead.setNext(newTail);
            newTail.setPrev(newHead);
            
            // Attach head and newHead.
            newHead.setBelow(head);
            head.setAbove(newHead);
            
            // Attach tail and newTail.
            newTail.setBelow(tail);
            tail.setAbove(newTail);
            
            // Keep head and tail pointers at the top.
            head = newHead;
            tail = newTail;
            height++;       // increment height
        }
        
        
        // Inserting from bottom up
        Node back = p;
        Node down = null;
        
        // For each level i up until randomHeight (from bottom up) in the skip list, create and link the new node horizontally and vertically.
        for (int i = 0; i <= randomHeight; i++) {
            Node newEntry = new Node(k, v);
            
            // Horizontal insert on current level
            newEntry.setNext(back.getNext());
            newEntry.setPrev(back);
            back.getNext().setPrev(newEntry);
            back.setNext(newEntry);
            
            // Vertical link above and below the new node (down must not be null
            // in order to add above).
            if (down != null) {
                newEntry.setBelow(down);
                down.setAbove(newEntry);
            }
            down = newEntry;    // Down becomes the new entry (no longer null in case we need to add at next level)
            
            // While back's above pointer is null and back is pointing to a previous node, we back track to the previous node.
            // Once we have found a node that points to another node above it, we set our back tracking node to the node above.
            if (i < randomHeight) {
                while (back.getAbove() == null && back.getPrev() != null) {
                    back = back.getPrev();
                }
                back = back.getAbove();
            }
        }
        
        return null;
    }
    
    /** Removes a node from the list (if a key exists).
     *  @param k  the key
     *  @return   returns the value removed (or null if none)
     */
    public String remove(String k) {
        Node p = findEntry(k);      // find the node
        
        // If a key does not exist return null
        if (!p.getDate().equals(k)) {
            return null;
        }
        
        // Save the value of the node being removed.
        String old = p.getEvent();
        
        // While p is not null we disconnect from bottom up.
        while (p != null) {
            p.getPrev().setNext(p.getNext());
            p.getNext().setPrev(p.getPrev());
            p = p.getAbove();
        }
        
        // Don't forget to remove extra empty lists! We only remove if the current top layer has another empty layer below!
        while (height > 1 && head.getNext().getDate().equals(POS_INF) && head.getBelow() != null
                && head.getBelow().getNext().getDate().equals(POS_INF)) {
            Node oldHead = head;
            Node oldTail = tail;
            
            // Drop head and tail
            head = head.getBelow();
            tail = tail.getBelow();
            
            // Disconnect empty top layer.
            if (head != null) head.setAbove(null);
            if (tail != null) tail.setAbove(null);
            oldHead.setBelow(null);
            oldTail.setBelow(null);
            
            height--;
        }
        
        return old; // Return the old value being removed
    }
    
    /** Returns the entry with the largest key less than or equal to the key.
     *  @param k  the key
     *  @return   the entry with the largest key less than or equal to the key (or null if none)
     */
    public Node floorEntry(String k) {
        Node p = findEntry(k);      // find the node (this should already be the largest key less than k unless negative infinity is the next largest)
        if (p.getDate().equals(NEG_INF)) return null;
        return p;
    }
    
    /** Returns the entry with the smallest key greater than or equal to the key.
     *  @param k  the key
     *  @return   the entry with the smallest key greater than or equal to the key (or null if none)
     */
    public Node ceilingEntry(String k) {
        Node p = findEntry(k);      // find the node
        if (p.getDate().equals(k)) return p;        // Same keys
        if (p.getNext().getDate().equals(POS_INF)) return null;     // Cannot be positive infinity
        return p.getNext();         // return the next node since we searched for the largest key less than k
    }
    
    /** Returns all entries that are between key1 and key2 (inclusive).
     *  @param key1  the lower bound of the range
     *  @param key2  the upper bound of the range
     *  @return      the entries between both keys
     */
    public List<Node> subMap(String key1, String key2) {
        List<Node> results = new ArrayList<>();
        Node start = ceilingEntry(key1);        // Start at ceiling entry
        
        // Make sure we drop down to S0
        if (start != null) {
            while (start.getBelow() != null) {
                start = start.getBelow();
            }
        }
        
        // While start is not null and start key is less than or equal to key2
        while (start != null && start.getDate().compareTo(key2) <= 0) {
            // If start's key does not equal positive infinity, add to list
            if (!start.getDate().equals(POS_INF) && !start.getDate().equals(NEG_INF)) {
                results.add(start);
            }
            // Go to next
            start = start.getNext();
        }
        return results; // Return results
    }
    
    /** Prints the skip list from top to bottom.
     * 
     */
    public void printSkipList() {
        Node levelHead = head;          // start at top head
        int currHeight = height - 1;    // current height to print
        
        // While the levels head is not null (while a list exists), print the list
        while (levelHead != null) {
            System.out.print("(S" + currHeight + ")");
            Node curr = levelHead.getNext();
            // The list is empty if the next node is positive infinity
            if (curr.getDate().equals(POS_INF)) {
                System.out.print(" empty");
            }
            // Otherwise traverse through the list and print out the results.
            else {
                while (curr != null && !curr.getDate().equals(POS_INF)) {
                    System.out.print(" " + curr.getDate() + ":" + curr.getEvent());
                    curr = curr.getNext();
                }
            }
            System.out.println();                   // new line
            levelHead = levelHead.getBelow();       // move to next level
            currHeight--;                           // one lower
        }
    }
}