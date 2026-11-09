/*
 * Author: Jackson Young
 * Email: young2025@my.fit.edu
 * Course: CSE 2010
 * Section: 1
 * Description of this file: Used to represent the Skip List ADT.
 * A skip list has multiple "layers" of linked lists that help skip forward during the searching process.
 * The top layer will always be empty, while the bottom layer will always contain every entry.
 * Objects are added to a random layer via a pseudorandom number generator in the FakeRandHeight class.
*/
import java.util.ArrayList;
public class SkipList<T> {
    
    // Node class that has FOUR pointers:
    // next and prev point forward and backwards across a layer
    // up and down allow traversal to other layers of the skip list
    // key is used as an identifier for the node, and value is the actual value
    class Node<T> {
        // Pointer fields
        int key;
        T value;
        Node<T> next, prev, up, down;
        // Constructors
        public Node(int k, T v, Node<T> n, Node<T> p, Node<T> u, Node<T> d) {
            key = k; value = v; next = n; prev = p; up = u; down = d;
        }
        public Node(int k, T v) {
            key = k; value = v;
        }
        // toString
        @Override
        public String toString() {
            return String.format("%04d", key) + ":" + value;
        }
    }

    // Field variables
    private Node<T> start; // The upper left node of the skip list
    private int height = 0; // The number of layers
    private FakeRandHeight rng = new FakeRandHeight(); // PseudoRandom Number Generator

    // Constructor
    public SkipList() {
        Node<T> minNode = new Node(Integer.MIN_VALUE, null);
        Node<T> maxNode = new Node(Integer.MAX_VALUE, null);
        minNode.next = maxNode;
        maxNode.prev = minNode;
        start = minNode;
    }

    // public method to check if the skip list is empty
    public boolean isEmpty() {
        return (start.down == null) && (start.next.key == Integer.MAX_VALUE);
    }

    // public method used to search for and return a value from the skip list without removal
    // return null if the value is not found
    public T get(int key) {
        Node<T> found = floor(key);
        if (found != null && found.key == key) return found.value;
        else return null;
    }

    // public method to add key-value pairs to the skip list
    // The skip list will be searched for the key and
    // found values will be overwritten and the old value returned
    public T put(int key, T value) {
        // Get random height from FakeRandHeight class
        int randHeight = rng.get();

        // Check for key repeats
        Node<T> found = floor(key);
        if (found != null && found.key == key) {
            T oldValue = found.value;
            // Replace all the way down the skip list
            while (found != null) {
                found.value = value; // Replace old value
                found = found.down; // Move down
            }
            return oldValue; // Return old value

        } else {
            // Find location for new node:
            // Start with the randomized height and find the correct position by traversing each node in every applicable layer.
            Node<T> lastInserted = null; // Keep track of the last inserted node for up/down pointers
            Node<T> head = start.down; // Keep track of the layer to check
            int layer = height - 1;

            // Check if a layer must be added
            if (randHeight >= height) {
                // Add new key to top layer
                Node<T> newNode = new Node(key, value, start.next, start, null, null);
                lastInserted = newNode;
                // Fix pointers
                start.next.prev = newNode;
                start.next = newNode;
                // Add new layer on top
                newLayer();
            }

            while (head != null) {
                if (layer <= randHeight) {
                    // Find the correct position for the node
                    Node<T> current = head.next;
                    while (current != null) {
                        // Check if current node is greater than key
                        // If it is, insert the new node here
                        // Otherwise, move to the next node
                        if (current.key > key) {
                            // Insert new node before current
                            Node<T> newNode = new Node(key, value, current, current.prev, lastInserted, null);
                            if (lastInserted != null) lastInserted.down = newNode;
                            lastInserted = newNode;

                            // Fix pointers
                            current.prev.next = newNode;
                            current.prev = newNode;

                            break; // Exit loop

                        } else {
                            // Continue down the chain
                            current = current.next;
                        }
                    }
                }
                // Go down a layer
                head = head.down;
                layer--;
            }
            // Default return null (nothing replaced)
            return null;
        }
    }

    // public method for finding and removing a specified key
    // The removed value is returned or null if not found
    public T remove(int key) {
        // First find the key's node
        Node<T> found = floor(key);
        T foundValue = null;
        if (found != null && found.key == key) foundValue = found.value;

        // Move down the tower and fix pointers
        while (found != null) {
            // Fix pointers
            found.prev.next = found.next;
            found.next.prev = found.prev;
            // Move down
            found = found.down;
        }
        
        clean(); // Remove extra empty layers
        return foundValue;
    }

    // public method to search for the floor of a node with the specified key
    // return null if the node can't be found
    public String floorEntry(int key) {
        Node<T> floorNode = floor(key);
        if (floorNode != null) return floorNode.toString();
        else return null;
    }

    // public method that uses the floor function to return the closest entry >= the key
    // returns null if no value is greater than the key
    public String ceilingEntry(int key) {
        Node<T> ceilingNode = ceiling(key);
        if (ceilingNode != null) return ceilingNode.toString();
        else return null;
    }

    // public method that utilizes the floor function to return an arraylist of the nodes between k1 and k2
    public ArrayList<String> subMap(int k1, int k2) {
        if (k1 > k2) return null; // Default invalid boundaries check
        // Initialize empty list
        ArrayList<String> nodes = new ArrayList();
        // Get smallest value in interval
        Node<T> current = ceiling(k1);
        // Move to the bottom of the tower
        while (current.down != null) {
            current = current.down;
        }
        // Loop from smallest to largest and add to list
        while (current != null && current.key <= k2) {
            // Don't print sentinel values
            if (current.key != Integer.MIN_VALUE && current.key != Integer.MAX_VALUE) {
                nodes.add(current.toString());
            }
            current = current.next;
        }
        return nodes;
    }
    
    // public print method for visualizing the skip list's contents
    public void print() {
        Node<T> head = start;
        int layer = height;
        while (head != null) {
            Node<T> current = head.next;
            if (current.key == Integer.MAX_VALUE) {
                // Empty list
                System.out.println("(S" + layer +") empty");
            } else {
                System.out.print("(S" + layer + ")");
                while (current.key < Integer.MAX_VALUE) {
                    // Print every entry node in the layer
                    System.out.print(" " + current);
                    current = current.next;
                }
                System.out.println();
            }
            // Go down a layer
            head = head.down;
            layer--;
        }
    }

    //-- PRIVATE METHODS --//

    // private helper method that simplifies adding a new layer to the top of the skip list
    // Each new layer will start off with min and max value nodes
    private void newLayer() {
        // Create sentinel nodes
        Node<T> minNode = new Node(Integer.MIN_VALUE, null);
        Node<T> maxNode = new Node(Integer.MAX_VALUE, null);

        // Assign pointers
        minNode.next = maxNode;
        minNode.down = start;
        start.up = minNode;
        maxNode.prev = minNode;

        // Reassign start
        start = minNode;
        height++; // Increment height of skip list
    }

    // private method that removes empty lists from the top of the skip list
    private void clean() {
        while (start.down != null && start.down.next.key == Integer.MAX_VALUE) {
            if (start.down.down != null) {
                // If there is a node below the empty layer, cut out the empty list
                start.down.down.up = start;
                start.down = start.down.down;
            } else {
                // Otherwise, the skip list only has one layer
                start.down = null;
            }

            // Decrement height
            height--;
        }
    }

    // private helper method to search for the closest node to the specified key
    // return null if the node isn't found or all nodes are greater then key
    private Node<T> floor(int key) {
        // Start with the upper-left node of the skip list
        Node<T> current = start;
        while (current.key != key) {
            // If the next node is less than key, move to it
            // Otherwise, move down if possible
            // If moving down isn't possible, return null (not found)
            Node<T> nextNode = current.next;
            Node<T> downNode = current.down;
            if (nextNode.key <= key) { // Hop to the next node
                current = nextNode;
            } else if (downNode != null) { // Attempt to drop to the next layer
                current = downNode;
            } else {
                // Cannot go down any further
                if (current.key != Integer.MIN_VALUE) { // Don't return left sentinel
                    return current;
                } else {
                    return null;
                }
            }
        }
        return current; // Found node
    }

    // private method to search for the ceiling of a node with the specified key
    // return the actual node or null if the node can't be found
    private Node<T> ceiling(int key) {
        // Start with the upper-left node of the skip list
        Node<T> current = start;
        while (current.key != key) {
            // If the next node is less than key, move to it
            // Otherwise, move down if possible
            // If moving down isn't possible, return null (not found)
            Node<T> nextNode = current.next;
            Node<T> downNode = current.down;
            if (nextNode.key <= key) { // Hop to the next node
                current = nextNode;
            } else if (downNode != null) { // Attempt to drop to the next layer
                current = downNode;
            } else {
                // Cannot go down any further
                if (nextNode.key != Integer.MAX_VALUE) { // Don't return sentinels
                    return nextNode;
                } else {
                    return null;
                }
            }
        }
        return current; // Found node
    }
}