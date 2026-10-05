/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
  This file implements a general tree structure that is used to store strings containing names
  of supervisors and subordinates (from a company for example). The tree structure is made of
  nodes that are linked together between their parents and/or children (if any). This file also
  contains the query methods called from HW3.java when a query is read from an input file given
  HW3.java.

 */

import java.util.ArrayList;
import java.util.List;
import java.io.IOException;

public class Tree
{
    /** Node of a tree. Stores a reference to its element, a list of its
     *  children, and to its parent node (null if its the root).
     * 
     */
    private static class Node {
        /** The element stored at this node. */
        private String element;
        
        /** A reference to the list of its children */
        private List<Node> children;
        
        /** A reference to its parent (if any) */
        private Node parent;
        
        /** Creates a new node with the given element, a list to store its
         *  children, and a reference to store its parent.
         *  
         *  @param e  the element to be stored
         */
        public Node(String e) {
            this.element = e;
            this.children = new ArrayList<>();
            this.parent = null;
        }
        
        // Accessor methods
        /** Returns the element stored at the node.
         *  @return the element stored at the node.
         */
        public String getElement() { return this.element; }
        
        /** Returns the parent of the node (the node above).
         *  @return the parent of the node
         */
        public Node getParent() { return this.parent; }
        
        /** Returns the list that stores the children for this node.
         *  @return the list that stores the children.
         */
        public List<Node> getChildren() { return this.children; }
        
        // Update methods
        /** Sets the element of the node to the new element e.
         *  @param e  the new element
         */
        public void setElement(String e) {
            this.element = e;
        }
        
        /** Sets the parent of the node.
         *  @param parent  the new parent node
         */
        public void setParent(Node parent) {
            this.parent = parent;
        }
        
        /** Sets the list that stores the children of this node.
         *  @param children  the new list of children
         */
        public void setChildren(List<Node> children) {
            this.children = children;
        }
    }
    
    // Instance variables of the tree
    protected Node root = null;      // root of the tree
    private int size = 0;               // number of nodes in the tree
    
    /** Factory function to create a new node storing element e. */
    public Node createNode(String e) {
        return new Node(e);
    }
    
    // Constructor
    public Tree() { }
    
    // access methods
    /**
    * Returns the number of elements in the tree.
    * @return number of elements in the tree
    */
    public int size() { return size; }
    
    /** Returns whether the tree is empty or not.
     *  @return true if the tree is empty
     */
    public boolean isEmpty() { return size == 0; }
    
    /** Returns the root (top) of the tree.
     *  @return the root of the tree
     */
    public Node getRoot() {
        return root;
    }
    
    /** Adds the root (the first element) to the tree.
     *  @param e  the first element to be added to the tree
     */
    public void addRoot(String e) throws IllegalStateException {
        // If the tree is not empty then we cannot add a root!
        if (!isEmpty()) throw new IllegalStateException("Root already exists!");
        root = createNode(e);
        size = 1;
    }
    
    /** Creates a new child beneath a specified parent storing element e.
     *  @param parent  the parent of the new child
     *  @param child  the new child being added
     */
    public void addChild(String parent, String child) throws IllegalArgumentException {
        // Searches and stores the parent node (or null if not found).
        Node parentNode = findNode(root, parent);
        if (parentNode == null) {
            throw new IllegalArgumentException("Parent node not found!");
        }
        
        Node newNode = new Node(child); // Creates the new child node to be stored.
        newNode.setParent(parentNode);  // Sets the parent of the child node.
        List<Node> children = parentNode.getChildren(); // Stores the list of children the parent has.
        
        int i = 0;  // Index to keep track of where we are in the children list
        
        // For each child in the parent's children list, we check if the current child is alphabetically greater.
        while (i < children.size()) {
            String currentChildName = children.get(i).getElement().toString();
            String newChildName = child;
            
            // If the current child is alphabetically greater, we found our insertion spot
            if (currentChildName.compareTo(newChildName) > 0) {
                break;
            }
            i++;
        }
        children.add(i, newNode);   // Add the new child to the insertion spot found above
        size++;                     // Increase the size of the tree
    }
    
    /** Traverse through the tree (recursively) to find the node.
     *  @param current  the current node we check (starts with the root)
     *  @param element  the element we want to find
     */
    public Node findNode(Node current, String element) {
        if (current == null) return null;   // nothing to find
        
        // Base case:
        // If the current nodes element is equal to the element we want to find, then found!
        if (current.getElement().equals(element)) {
            return current;
        }
        
        // For each node child in the current node's children list
        for (Node child : current.getChildren()) {
            // Recursive call:
            // Call findNode with current as the child node.
            Node found = findNode(child, element);
            
            // Returns found if it is not null.
            if (found != null) {
                return found;
            }
        }
        return null;    // Otherwise returns null
    }
    
    // Query methods
    /** Finds the direct supervisor (parent) of a given node in the tree (if any).
     *  @param entity  the name of the subordinate of the unknown supervisor
     */
    public void directSupervisor(String entity) {
        Node node = findNode(root, entity);     // Finds the entity node
        String sup = "none";                    // Default "none" in case a direct supervisor is not found
        // If the node was found and they have a parent, we found a supervisor
        if (node != null && node.getParent() != null) {
            sup = node.getParent().getElement();
        }
        // Outputs the direct supervisor (parent) of the entity (child).
        System.out.println("DirectSupervisor " + entity + " " + sup);
    }
    
    /** Finds the direct subordinates (children) of a given node in the tree (if any).
     *  @param entity  the name of the supervisor of the unknown children
     */
    public void directSubordinates(String entity) {
        Node node = findNode(root, entity);     // Finds the entity node
        StringBuilder sb = new StringBuilder();
        
        // If the parent node was found and it has children we output its children
        if (node != null && !node.getChildren().isEmpty()) {
            // For each child in the node's children list, append to the string builder.
            for (Node child : node.getChildren()) {
                sb.append(child.getElement()).append(" ");
            }
        }
        
        // If string builder has nothing, then there are no children.
        // Otherwise the result to be printed is the list of children.
        String res = sb.length() == 0 ? "none" : sb.toString().trim();
        
        // Outputs the direct subordinates (children) of the entity (parent).
        System.out.println("DirectSubordinates " + entity + " " + res);
    }
    
    /** Finds all of the supervisors (parents, grandparents, ect.) of a given node in the tree (if any).
     *  @param entity  the name of the subordinate that has the unknown parents, grandparents, ect.
     */
    public void allSupervisors(String entity) {
        Node node = findNode(root, entity);     // Finds the entity node
        List<String> sups = new ArrayList<>();  // Declares a list to hold all supervisors
        
        // If a node was found we add all of the supervisors (parents) to the list
        if (node != null) {
            Node curr = node.getParent();
            
            // While the current node has a parent we add the parents name to the list
            while (curr != null) {
                sups.add(curr.getElement());
                curr = curr.getParent();
            }
        }
        
        StringBuilder sb = new StringBuilder();     // String builder to append each supervisor we found
        
        // For each supervisor we found in the supervisor list we append them to the string builder.
        for (String sup : sups) {
            sb.append(sup).append(" ");
        }
        
        // If the string builder has nothing, then there are no supervisors.
        // Otherwise the result to be printed is the list of all the node's supervisors.
        String res = sb.length() == 0 ? "none" : sb.toString().trim();
        
        // Outputs all supervisors (parents) of the entity (child).
        System.out.println("AllSupervisors " + entity + " " + res);
    }
    
    /** Finds all of the subordinates (children, grandchildren, ect.) of a given node in the tree (if any).
     *  @param entity  the name of the supervisor containing all of the subordinates (children, grandchildren, ect.)
     */
    public void allSubordinates(String entity) {
        Node node = findNode(root, entity);     // Finds the entity node
        List<String> subs = new ArrayList<>();  // Declares a list to hold all subordinates
        
        // If a node was found we search and collect each of its children, grandchildren, ect. using pre-order traversal
        if (node != null) {
            preOrderCollect(node, subs, true); // Skip the entity itself, collect descendants
        }
        
        StringBuilder sb = new StringBuilder(); // String builder to append each subordinate we found
        
        // For each subordinate we found in the subordinate list we append them to the string builder.
        for (String sub : subs) {
            sb.append(sub).append(" ");
        }
        
        // If the string builder has nothing, then there are no subordinates.
        // Otherwise the result to be printed is the list of all the node's subordinates.
        String res = sb.length() == 0 ? "none" : sb.toString().trim();
        
        // Outputs all subordinates (children) of the entity (parent).
        System.out.println("AllSubordinates " + entity + " " + res);
    }
    
    
    
    /** Traverses from an original node back up to the root to find the number of supervisors above the node.
     *  @param entity  the name of the subordinate with the unknown number of supervisors
     */
    public void numberOfAllSupervisors(String entity) {
        Node node = findNode(root, entity); // Finds the entity node
        int count = 0;                      // Count to keep track of the number of supervisors
        
        // If a node was found we get its parent
        if (node != null) {
            Node curr = node.getParent();
            
            // While current is not null we increment the count and get its parent.
            while (curr != null) {
                count++;
                curr = curr.getParent();
            }
        }
        
        // Outputs the total number of supervisors above the entity node in the tree.
        System.out.println("NumberOfAllSupervisors " + entity + " " + count);
    }
    
    /** Finds the number of subordinates below the node.
     *  @param entity  the name of the supervisor with the unknown number of subordinates
     */
    public void numberOfAllSubordinates(String entity) {
        Node node = findNode(root, entity); // Finds the entity node
        int count = 0;                      // Count to keep track of the number of subordinates
        
        // If a node was found we find the number of subordinates it has
        if (node != null) {
            count = countSubordinates(node) - 1; // exclude entity itself
        }
        
        // Outputs the total number of subordinates below the entity node in the tree.
        System.out.println("NumberOfAllSubordinates " + entity + " " + count);
    }
    
    /** Find the total number of subordinates (children, grandchildren, ect.) below a given current node.
     *  @param curr  the node we want to count children from
     */
    private int countSubordinates(Node curr) {
        if (curr == null) return 0; // No children
        int count = 1;              // Count of one to count each individual child
        
        // For each child node in the current node, we count its children and return that number.
        for (Node child : curr.getChildren()) {
            count += countSubordinates(child);
        }
        return count;   // Should be one for one child
    }
    
    /** Outputs whether a given supervisor is the entity's supervisor (does not need to be direct).
     *  @param entity  the entity we check the supervisor with
     *  @param supervisor  the supervisor in question
     */
    public void isSupervisor(String entity, String supervisor) {
        Node node = findNode(root, entity); // Finds the entity
        boolean found = false;              // Keep track of whether we found the supervisor
        
        // If a node was found we get its parent
        if (node != null) {
            Node curr = node.getParent();
            
            // While current is not null we check if the current nodes element is equal to the supervisor
            while (curr != null) {
                if (curr.getElement().equals(supervisor)) {
                    found = true;
                    break;
                }
                curr = curr.getParent();    // Gets the next supervisor (parent, if any) to check
            }
        }
        
        // Outputs whether the entity has the specified supervisor
        System.out.println("IsSupervisor " + entity + " " + supervisor + " " + (found ? "yes" : "no"));
    }
    
    /** Outputs whether a given subordinate is the entity's subordinate (does not need to be direct).
     *  @param entity  the entity we check the subordinate with
     *  @param subordinate  the subordinate in question
     */
    public void isSubordinate(String entity, String subordinate) {
        Node node = findNode(root, subordinate);    // Finds the subordinate
        boolean found = false;
        
        // If a node was found we get its parent
        if (node != null) {
            Node curr = node.getParent();
            
            // While current is not null we check if the current element is equal to the entity
            while (curr != null) {
                if (curr.getElement().equals(entity)) {
                    found = true;
                    break;
                }
                curr = curr.getParent();    // Gets the next supervisor (parent, if any) to check if we make it back to the entity
            }
        }
        
        // Outputs whether the entity has the specified subordinate
        System.out.println("IsSubordinate " + entity + " " + subordinate + " " + (found ? "yes" : "no"));
    }
    
    /** Compares the rank between of entities in the tree.
     *  @param entityOne  the entity being evaluated
     *  @param entityTwo  the entity for comparison
     */
    public void compareRank(String entityOne, String entityTwo) {
        int d1 = getDepth(entityOne);  // Gets the depth of entity one
        int d2 = getDepth(entityTwo);  // Gets the depth of entity two
        String rank = "same";          // Default rank of same if they are neither greater nor less than each other
        
        // If depth one is less than depth two, then depth two has a higher rank
        if (d1 < d2) {
            rank = "higher";
        }
        // If depth one is greater than depth two, then depth two has a lower rank
        else if (d1 > d2) {
            rank = "lower";
        }
        
        // Outputs whether entity one is greater than, less than, or equal to depth two.
        System.out.println("CompareRank " + entityOne + " " + entityTwo + " " + rank);
    }
    
    /** Finds the closest common supervisor between two given entities.
     *  @param entityOne  the first node to evaluate
     *  @param entityTwo  the second node to evaluate against the first
     */
    public void closestCommonSupervisor(String entityOne, String entityTwo) {
        String common = null;                           // Default to null
        
        // If entity one and entity two are equal then return the parent (since they are the same)
        if (entityOne.equals(entityTwo)) {
            Node node = findNode(root, entityOne);
            node = node.getParent();
            
            // If we found an existing node then we add the element
            if (node != null) {
                common = node.getElement();
            }
        }
        // Otherwise continue finding the closest common supervisor
        else {
            // Finds the path of both nodes from the node to the root
            List<String> p1 = getPathFromRoot(entityOne);
            List<String> p2 = getPathFromRoot(entityTwo);
        
            int min = Math.min(p1.size(), p2.size());       // Ensures we only loop up until the smallest's max index
        
            // For each index (up until the smallest's max index) we check if both paths have equal strings at the same index
            for (int i = 0; i < min; i++) {
                if (p1.get(i).equals(p2.get(i))) {
                    common = p1.get(i);     // If both paths find equal names at same index it becomes common
                }
                // Break out of look if we no longer find a common supervisor
                else {
                    break;
                }
            }
        }
        
        // If common is null then there are no common supervisors.
        // Otherwise the result to be printed is the closest common supervisor between two entities.
        String res = (common == null) ? "none" : common;
        System.out.println("ClosestCommonSupervisor " + entityOne + " " + entityTwo + " " + res);    
    }
    
    // Additional methods
    /** Visits each node below a given starting point (using recursive pre-order traversal) and adds them to the list.
     *  @param curr  the current node being checked and added
     *  @param list  the list containing the added nodes
     *  @param skipFirst  a boolean value to make sure we don't add the first element to the list
     */
    private void preOrderCollect(Node curr, List<String> list, boolean skipFirst) {
        if (curr == null) return;   // Do nothing
        
        // If current is the first element we skip adding it. Otherwise we add the element to the list.
        if (!skipFirst) {
            list.add(curr.getElement());
        }
        
        // For each node child in the current node's list of children, we call preOrderCollect to add it to the list (and then it will call for its children (if any).
        for (Node child : curr.getChildren()) {
            preOrderCollect(child, list, false);
        }
    }
    
    /** Finds the depth of a given entity in the tree (how far down they are).
     *  @param entity  the entity we want to know the depth from
     */
    private int getDepth(String entity) {
        Node node = findNode(root, entity); // Finds the entity node
        if (node == null) return -1;        // Node does not exist in the tree
        int depth = 0;                      // Depth counter to keep track of how deep we are in the tree
        Node curr = node.getParent();
        
        // While current is not null we traverse up to the root through each nodes parent. Each parent we find adds the depth by one.
        while (curr != null) {
            depth++;
            curr = curr.getParent();
        }
        return depth;
    }
    
    /** Returns a list of elements tracing the path from the root node to the target.
     *  @param entity  the entity node we want to trace the path from
     */
    private List<String> getPathFromRoot(String entity) {
        List<String> path = new ArrayList<>();  // List to store the path from the entity to the root
        Node node = findNode(root, entity);     // Finds the entity node
        
        // While node is not null add the node element (of each parent up to the root) to the beginning of the list.
        while (node != null) {
            path.add(0, node.getElement());
            node = node.getParent();
        }
        return path; // Return the list
    }
}