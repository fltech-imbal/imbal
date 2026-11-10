// Author: Adrian Santana
// Email: santanaa2025@my.fit.edu
// Section: 1
// Description: The program shown below implements and utilizes a SkipList to display a timeline of major events
// for a social media app.

public class SkipList {
    // Create a private class to design the nodes and give it reference pointers, a key, and a value
    private static class QuadNode {
        // Make reference pointers for each node to help in traversing the skip list
        private QuadNode up;
        private QuadNode down;
        private QuadNode left;
        private QuadNode right;
        // Give each node a key and value pair
        private int key;
        private String value;

        // Make a constructor to define a node with a key and a value
        QuadNode(int key, String value) {
            this.key = key;
            this.value = value;
        }
    }
    // Create a QuadNode reference that points to the start or top left of the skip list
    QuadNode start;
    FakeRandHeight heightGenerator = new FakeRandHeight();
    // Declare an int variable that keeps track of the height of the skip list
    int height = 0;

    // Make a constructor that starts the skip list with an empty layer
    public SkipList() {
        start = createLayer();
    }
    // Create a method that creates a level in the skip list with negative infinity and positive infinity at its ends
    private QuadNode createLayer() {
        QuadNode negInf = new QuadNode(Integer.MIN_VALUE, null);
        QuadNode posInf = new QuadNode(Integer.MAX_VALUE, null);
        // Connect the negative infinity and positive infinity nodes
        posInf.left = negInf;
        negInf.right = posInf;
        return negInf;
    }

    // Create a void method that takes an integer (key) and a String (value) and inserts a node in its proper position in the
    // skip list based on these parameters
    public void put(int key, String value) {
        int insertHeight = heightGenerator.get();
        // If the insertHeight is greater than or equal to the height of the skip list, add a new layer to the skip list
        if (insertHeight >= height) {
            insertHeight = height;
            QuadNode newLayerNegInf = createLayer();
            // Update each node's pointers after adding the new layer to the skip list
            newLayerNegInf.down = start;
            start.up = newLayerNegInf;
            newLayerNegInf.right.down = start.right;
            start.right.up = newLayerNegInf.right;
            start = newLayerNegInf;
            // Increase the height of the skip list
            height++;
        }

        QuadNode current = start;
        int currentHeight = height;
        QuadNode lastInsert = null;
        while (current != null) {
            // If the node on the right's key is less than or equal to the key, move to the right
            if (current.right.key <= key) {
                current = current.right;
            } else {
                // If the current height of the skip list is less than or equal to the insert height, insert the new node there
                if (currentHeight <= insertHeight) {
                    QuadNode newNode = new QuadNode(key, value);
                    // Update the pointers of each node after inserting newNode
                    current.right.left = newNode;
                    newNode.right = current.right;
                    newNode.left = current;
                    current.right = newNode;
                    // If lastInsert is not null, connect the pointers of newNode and lastInsert vertically
                    if (lastInsert != null) {
                        newNode.up = lastInsert;
                        lastInsert.down = newNode;
                    }
                    lastInsert = newNode;
                }
                // Move the current node down and decrement the height of the current node
                current = current.down;
                currentHeight--;
            }
        }
    }
    // Create a method that takes an integer (key) and searches for a node with the key
    // and returns a String (value of the node)
    public String get(int key) {
        QuadNode current = start;
        while (current != null) {
            // While the node to the right's key is less than the key, move to the right
            while (current.right.key < key) {
                current = current.right;
            }
            // If the node to the right's key is the desired key, return the right node's value
            if (current.right.key == key) {
                return current.right.value;
            }
            // Move current down to the next level in the skip list
            current = current.down;
        }
        return null;
    }
    // Create a method that takes in an integer (key) and removes the entry from the skip list and returns a String with
    // the removed entry's value
    public String remove(int key) {
        QuadNode current = start;
        QuadNode target = null;

        while (current != null) {
            // While the key of the node to the right of current is less than the key, move to the right
            while (current.right.key < key) {
                current = current.right;
            }
            // if the key of the node to the right of current is the key, set the target to be current's right node
            if (current.right.key == key) {
                target = current.right;
            }
            // Move the current node down
            current = current.down;
        }
        // If the target is null, return null
        if (target == null) {
            return null;
        }
        String oldValue = target.value;
        // While the node below the target is not null, move the target node down
        while (target.down != null) {
            target = target.down;
        }
        // While the target node is not null, update each node's pointers to remove the copies from the bottom-up
        while (target != null) {
            QuadNode upper = target.up;
            target.left.right = target.right;
            target.right.left = target.left;
            target = upper;
        }
        current = start.down;
        // While the node below current is not null and its right node is positive infinity, update the pointers of each node
        while (current.down != null && current.right.key == Integer.MAX_VALUE) {
            current.up.down = current.down;
            current.down.up = current.up;
            current.right.down.up = current.right.up;
            current.right.up.down = current.right.down;
            // Move current down and decrement the height
            current = current.down;
            height--;
        }
        // return the value belonging to the removed key
        return oldValue;
    }
    // Make a method that takes an integer (key) and returns a String (entry) with the largest key
    // that is less than or equal to the key
    public String floorEntry(int key) {
        QuadNode current = start;
        while (current != null) {
            // While the node on the right of current's key is less than or equal to the key, move current right
            while (current.right.key <= key) {
                current = current.right;
            }
            // if current's key is not negative infinity, use String.format() to return the entry
            if (current.key != Integer.MIN_VALUE) {
                return String.format("%04d:%s", current.key, current.value);
            }
            // Move current down
            current = current.down;
        }
        return null;
    }
    // Create a method that takes in an integer (key) and returns a String (entry) with the smallest key
    // greater than or equal to the key
    public String ceilingEntry(int key) {
        QuadNode current = start;
        while (current != null) {
            // while the node to the right of current's key is less than the key, move current right
            while (current.right.key < key) {
                current = current.right;
            }
            // If the key of the node to the right of current is not positive infinity and is greater than or equal to the key,
            // return the entry using String.format()
            if (current.right.key != Integer.MAX_VALUE) {
                if (current.right.key >= key) {
                    return String.format("%04d:%s", current.right.key, current.right.value);
                }
            }
            // move current down
            current = current.down;
        }
        return null;
    }
    // Create a method that takes in two integers (key1, key2) and returns a String (result) with all the entries
    // of which their keys are between or equal to key1 and key2
    public String subMap(int key1, int key2) {
        // if key1 is greater than key2, return "none"
        if (key1 > key2) {
            return "none";
        }
        QuadNode current = start;
        while (current.down != null) {
            // while the key of the node to the right of current is less than key1, move current to the right
            while (current.right.key < key1) {
                current = current.right;
            }
            // Move current down
            current = current.down;
        }
        while (current.right.key < key1) {
            current = current.right;
        }
        String result = "";
        // while the key of the node to the right of current is less than or equal to key2, move current to the right
        while (current.right.key <= key2) {
            current = current.right;
            if (!result.isEmpty()) {
                result += " ";
            }
            // Update result with the found entries using String.format()
            result += String.format("%04d:%s", current.key, current.value);
        }
        // if result is empty by the time traversal is complete, return "none"
        if (result.isEmpty()) {
            return "none";
        }
        // return the resulting String;
        return result;
    }
    // Create a toString() method that prints a representation of the skip list created
    public String toString() {
        String result = "";
        QuadNode currentLayerStart = start;
        int currentHeight = height;
        while (currentLayerStart != null) {
            // update result with the label of the layer
            result += String.format("(S%d)", currentHeight);
            QuadNode currentNode = currentLayerStart;
            while (currentNode.key != Integer.MAX_VALUE) {
                // If the current node is negative infinity and the node to the right is positive infinity, update result
                // with " empty"
                if (currentNode.key == Integer.MIN_VALUE && currentNode.right.key == Integer.MAX_VALUE) {
                    result += " empty";
                    // If the current node is not negative infinity, update result with the entry
                } else if (currentNode.key != Integer.MIN_VALUE){
                    result += String.format(" %04d:%s", currentNode.key, currentNode.value);
                }
                // Move currentNode right
                currentNode = currentNode.right;
            }
            // If currentHeight is greater than 0, update result with a new line
            if (currentHeight > 0) {
                result += "\n";
            }
            // Move down to the next level and decrement currentHeight
            currentLayerStart = currentLayerStart.down;
            currentHeight--;
        }
        // Return the resulting String from the method
        return result;
    }
}
