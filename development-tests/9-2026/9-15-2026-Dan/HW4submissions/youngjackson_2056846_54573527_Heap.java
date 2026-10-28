/*
 * Author: Jackson Young
 * Email: young2025@my.fit.edu
 * Course: CSE2010
 * Section: 1
 * Heap class is an implementation of the priority queue ADT that uses a binary tree
 * structure to keep integer keys sorted. Values are held as type T in an arraylist.
*/
import java.util.ArrayList;
public class Heap<T> {

    // Entry class for Heap
    public class Entry<T> {
        // Fields
        double key;
        T value;
        // Constructor
        public Entry(double key, T value) {
            this.key = key;
            this.value = value;
        }
    }

    // Field variables
    ArrayList<Entry<T>> heap; // Array to store the values of the heap

    // Constructor
    public Heap() {
        this.heap = new ArrayList();
    }

    // Methods \\
    // Public method that checks and returns whether the heap is empty
    public boolean isEmpty() {
        return heap.isEmpty();
    }

    // Public method that checks and returns whether the heap is full (== 100)
    public boolean isFull() {
        return heap.size() == 100;
    }

    // Public method that adds a value to the heap
    // Necessary operations are performed to ensure the heap maintains the correct struture
    // The first parameter is a value of type T that will be added to the heap via an Entry<T> object
    // THe second is a foating-point number of type double that will determine the priority of the object
    public void add(T value, double key) {
        // Check if heap is full
        // If it is full, nothing can be added
        if (!isFull()) {
            // Make a new entry and add it to the end of the heap
            Entry<T> newEntry = new Entry(key, value);
            heap.add(newEntry);

            // Check if the heap is of size 1
            // If it is, the heap is automatically sorted
            // If it isn't, upheap the entry with the greatest key to the front of the heap
            if (heap.size() != 1) {
                upheap(newEntry);
            }
        }
    }

    // Private helper method that recursively upheaps an entry
    // The parameter is the entry to be moved up
    private void upheap(Entry<T> newEntry) {
        // Get indicies of newEntry and its "parent" entry
        int newIndex = heap.indexOf(newEntry);
        int parentIndex = (newIndex - 1) / 2;

        // Upheap if the key of the parent entry is greater
        // because the smallest key is needed at the top (min heap)
        // The tiebreaker is based on alphabetical order of the values
        Entry<T> parentEntry = heap.get(parentIndex);
        if (parentEntry.key > newEntry.key || (parentEntry.key == newEntry.key &&
        parentEntry.value.toString().compareTo(newEntry.value.toString()) > 0)) {
            // Swap the values
            heap.set(newIndex, parentEntry);
            heap.set(parentIndex, newEntry);

            // Call upheap again now that the newEntry is at the proper index
            upheap(newEntry);
        }
    }

    // Public method that returns the minimum (root) value of the heap
    // Nothing is removed; this works the same way as a "peek" function
    public T getMin() {
        return heap.get(0).value;
    }

    // Public method that "removes" the minumum value (the root) from the heap and returns it
    // The end value is swapped to the root value and is then downheaped to its respective position
    public T removeMin() {
        // Swap the root and the end value
        Entry<T> root = heap.get(0);
        heap.set(0, heap.get(heap.size() - 1));
        heap.set(heap.size() - 1, root);

        // Remove root
        heap.remove(heap.size() - 1);
        // Downheap the new root to its respective position
        downheap(heap.get(0));
        // Return min
        return root.value;
    }

    // Private helper method that recursively moves the specified entry down the heap based on keys
    // Parameter is the entry that should be moved down
    private void downheap(Entry<T> entry) {
        // Get indicies of the entry and its "children" entries
        int entryIndex = heap.indexOf(entry);
        int leftChildIndex = 2 * entryIndex + 1;
        int rightChildIndex = 2 * entryIndex + 2;
        // Get the keys of the children if the index is within bounds
        if (leftChildIndex < heap.size() && rightChildIndex < heap.size()) { 
            Entry<T> leftChildEntry = heap.get(leftChildIndex);
            Entry<T> rightChildEntry = heap.get(rightChildIndex);

            // Determine the smaller child and swap with entry
            // Left side smaller
            if (leftChildEntry.key <= rightChildEntry.key && entry.key >= leftChildEntry.key) {
                // Swap
                heap.set(entryIndex, leftChildEntry);
                heap.set(leftChildIndex, entry);
                // Call downheap again
                downheap(entry);
            }
            // Right side smaller
            else if (rightChildEntry.key <= leftChildEntry.key && entry.key >= rightChildEntry.key) {
                // Swap
                heap.set(entryIndex, rightChildEntry);
                heap.set(rightChildIndex, entry);
                // Call downheap again
                downheap(entry);
            }
        } else if (leftChildIndex < heap.size() && rightChildIndex >= heap.size()) {
            Entry<T> leftChildEntry = heap.get(leftChildIndex);
            // Determine the smaller child and swap with entry
            if (entry.key >= leftChildEntry.key) {
                // Swap
                heap.set(entryIndex, leftChildEntry);
                heap.set(leftChildIndex, entry);
                // No need to call downheap again
            }
        }
    }

    // Public method that prints the heap in a readable way
    // Used for debugging purposes only
    public void print() {
        System.out.println("-----------------------------------");
        for (Entry<T> entry : heap) {
            System.out.println(entry.key + ": " + entry.value);
        }
    }
}