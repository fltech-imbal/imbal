// Name: Adrian Santana
// Email: santanaa2025@my.fit.edu
// Course: CSE 2010
// Description: The program shown below contains the methods used in HW3.java to implement the hierarchical structure
// of trees
import java.util.ArrayList;
import java.util.Collections;

public class Tree {
    private static class Node implements Comparable<Node> {
        String name;
        ArrayList<Node> children;
        Node parent;
        int depth;

        Node(Node parent, String data) {
            this.name = data;
            this.children = new ArrayList<>();
            this.parent = parent;
            if (parent != null) {
                parent.children.add(this);
                depth = parent.depth + 1;
            }
        }
        public ArrayList<String> getAllSubordinates() {
            ArrayList<String> subordinates = new ArrayList<>();
            for (Node child : children) {
                subordinates.add(child.name);
                subordinates.addAll(child.getAllSubordinates());
            }
            return subordinates;

    }

        public Node find(String data) {
            if (name.equals(data)) {
                return this;
            }
            else {
                for (Node child : children) {
                    Node foundNode = child.find(data);
                    if (foundNode != null) {
                        return foundNode;
                    }
                }
                return null;
            }

        }
        public void printNode() {
            System.out.println("\t".repeat(depth) + name);
            for (Node child : children) {
                child.printNode();
            }
        }

        // Use a compareTo method to sort each Node alphabetically by name
        @Override
        public int compareTo(Node o) {
            return this.name.compareTo(o.name);
        }
    }

    private Node root;

    // Make a constructor that takes a String (name) and creates the root Node of the tree
    public Tree(String name) {
        root = new Node (null, name);
    }

    // Create void method that takes two Strings (parentData and childData) and adds a new child Node under a parent
    public void addChild(String parentData, String childData) {
        Node foundParent = root.find(parentData);
        Node newChild = new Node(foundParent, childData);
        Collections.sort(foundParent.children);
    }

    // Create a method that takes in a String (childData) and returns a String with the name of the child Node's parent
    public String getParent(String childData) {
        Node foundChild = root.find(childData);

        if (foundChild.parent == null) {
            return null;
        } else {
            return foundChild.parent.name;
        }
    }
    // Create a method that takes in a String (parentData) and returns an ArrayList with the names of the parent Node's children
    public ArrayList<String> getChildren(String parentData) {
        Node foundParent = root.find(parentData);
        ArrayList<String> childrenData = new ArrayList<>();
        for (Node child : foundParent.children) {
            childrenData.add(child.name);
        }
        return childrenData;
    }

    // Create a method that takes in a String (data) and returns an ArrayList with the names of all supervisors
    public ArrayList<String> getAllSupervisors(String data) {
        Node current = root.find(data);
        ArrayList<String> supervisors = new ArrayList<>();
        while (current != null && current.parent != null) {
            current = current.parent;
            supervisors.add(current.name);
        }
        return supervisors;
    }

    // Create a method that takes in a String (data) that uses the getAllSubordinates() method from the Node class
    // and returns an ArrayList with the names of all subordinates
    public ArrayList<String> getAllSubordinates(String data) {
        Node current = root.find(data);
        return current.getAllSubordinates();
    }

    // Create a method that takes in a String (data) and returns an int representing the number of supervisors
    public int getNumberOfAllSupervisors(String data) {
        return getAllSupervisors(data).size();
    }

    // Create a method that takes in a String (data) and returns an int representing the number of subordinates
    public int getNumberOfAllSubordinates(String data) {
        return getAllSubordinates(data).size();
    }

    // Create a boolean method that takes two Strings (entity and supervisor) and determines whether
    // the supervisor appears in the entity's supervisor list, return true if so, and false otherwise
    public boolean isSupervisor(String entity, String supervisor) {
        ArrayList<String> supervisors = getAllSupervisors(entity);
        for (String s : supervisors) {
            if (s.equals(supervisor)) {
                return true;
            }
        }
        return false;
    }

    // Create a boolean method that takes two Strings (entity and subordinate) and determines whether
    // the subordinate appears in the entity's subordinate list, return true if so, and false otherwise
    public boolean isSubordinate(String entity, String subordinate) {
        ArrayList<String> subordinates = getAllSubordinates(entity);
        for (String s : subordinates) {
            if (s.equals(subordinate)) {
                return true;
            }
        }
        return false;
    }
    // Create a method that takes two Strings (entity1 and entity2) that returns the rank of entity1 compared to entity2
    // based on their ranks
    public String compareRank(String entity1, String entity2) {
        Node node1 = root.find(entity1);
        Node node2 = root.find(entity2);

        if (node1.depth < node2.depth) {
            return "higher";
        } else if (node1.depth > node2.depth) {
            return "lower";
        } else {
            return "same";
        }
    }
    // Create a method that takes two Strings (entity1 and entity2) and determines and returns their closest common
    // supervisor, or null otherwise
    public String closestCommonSupervisor(String entity1, String entity2) {
        ArrayList<String> entity1Supervisors = getAllSupervisors(entity1);
        ArrayList<String> entity2Supervisors = getAllSupervisors(entity2);
        for (String s : entity1Supervisors) {
            if (entity2Supervisors.contains(s)) {
                return s;
            }
        }
        return null;
    }

    public void printTree() {
        root.printNode();
    }
}
