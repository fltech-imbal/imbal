/**
 * Eddie Petracco III
 * epetracco2025@my.fit.edu
 * CSE 2010
 * 9:30 am Lab Section
 * Simulates a Tree data structure for an inputted data set that properly organizes a hiearchy of supervisors and subordinates starting with the root being the CEO. Different 
 * operations can be preformed on the strucutre, such as checking for subordinates or supervisors, and comparing ranks along with other queries from a given input file.
 */
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class HW3 {
    // Seperate Tree class that allows for multiple trees if neccessary but also keeps the code organized for convience.
    public static class Tree {
        // A nested node class similar to linked lists that allows for storing of data points in the Tree. Each node is a branch of the tree that all links back to a root
        private static class Node {
            private String element; 
            private Node parent;
            private ArrayList<Node> children;
            
            // A node the tree has an element which is a String of data, a parent and an arrayList for storing the children. The parent is null by default and the list is empty.
            public Node(String e) {
                element = e;
                parent = null; 
                children = new ArrayList<>();
            }
            
            // Getter functions
            // Returns the element of a given node
            public String getElement() { return element; }
            // Returns the parent of a given node 
            public Node getParent() { return parent; }
            // reutrns an arrayList of the children of a given node
            public ArrayList<Node> getChildren() { return children; }
            // Other Functions
            // Sets the parent to a given node
            public void setParent(Node p) { parent = p; }
            
            // adds a child to a given nodes nodeList
            public void addChild(Node c) {
                // This little complicated bit here is to ensure that the children list is in proper order according to the homework instructions
                int i = 0;
                while (i < children.size() && children.get(i).getElement().compareTo(c.getElement()) < 0) {
                    i++;
                }
                children.add(i, c);
            }
        }
        
        private Node root = null;
        // This arrayList keeps track of all the nodes in the tree which makes it possible to find any given node in the Tree by searching through this
        private ArrayList<Node> Tree = new ArrayList<>(); 
        
        // Specifcally adds the root to the tree which will have no parent
        public void addRoot(String element) {
            this.root = new Node(element);
            Tree.add(root);
        }
        
        // This method adds a new node to the tree. Takes the element and assigns its parent. Takes the parent and assings the element into its children list 
        public void addBranch(String element, String parent) {
            Node newBranch = new Node(element);
            Tree.add(newBranch);
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(parent)) {
                    current.addChild(newBranch);
                    newBranch.setParent(current);
                    return;
                }
            }
        }
        
        // Method for getting the supervisor of an entity. Searches through the Tree for the given Node and returns the parent using the getParent() method from the node class
        public String getSupervisor(String entity) {
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(entity)) {
                    if (current.getParent() == null) return "none";
                    return current.getParent().getElement();            
                }
            } 
            return "none";
        }
        
        // Method for getting the subordinates of an entity. Searches through the Tree, gets the childList of the entity, and returns a string of the names together.
        public String getSubordinates(String entity) {
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(entity)) {
                    if (current.getChildren().isEmpty()) return "none";
                    ArrayList<String> childNames = new ArrayList<>();
                    for (Node child : current.getChildren()) {
                        childNames.add(child.getElement());
                    }
                    return String.join(" ", childNames);
                }
            } 
            return "none"; 
        }
        
        // Method for checking if an entity has a given supervisor. Searches through the Tree for the entity, gets the parent (assuming its not the root), and returns yes or no
        public String isSupervisor(String entity, String supervisor) {
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(entity)) {
                    Node p = current.getParent();
                    while (p != null) {
                        if (p.getElement().equals(supervisor)) return "yes";
                        p = p.getParent();
                    }
                    return "no";
                }
            }   
            return "error";
        }
        
        // If the the given subordinate has the entity as a supervisor than the vice versa is true!
        public String isSubordinate(String entity, String subordinate) {
            return isSupervisor(subordinate, entity);
        }
        
        // Method for comparing the rank of two entities by finding their depth using the getDepth method and comparing the values
        public String compareRank(String entity1, String entity2) {
            int depth1 = 0;
            int depth2 = 0;
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(entity1)) depth1 = getDepth(current);
                if (current.getElement().equals(entity2)) depth2 = getDepth(current);
            }
            if (depth1 < depth2) return "higher";
            if (depth1 == depth2) return "same";
            return "lower";
        }
        
        // Method for getting the depth of an entity by going backwards through the tree and incrementing a value upwards until you reach the root
        public int getDepth(Node entity) {
            int depth = 0;
            while (entity.getParent() != null) {
                depth += 1;
                entity = entity.getParent();
            }
            return depth;
        }
        
        // This helps with the recursion happening in the allSubordinates method by adding all the children into a given list
        private void collectSubordinates(Node node, ArrayList<String> list) {
            for (Node child : node.getChildren()) {
                list.add(child.getElement());
                collectSubordinates(child, list);
            }
        }
        
        // Method for returning a string of all the subordinates of an entity. Finds it in the Tree. Calls the collectSubordinates method if the childList is not empty. 
        // the arrayList subs is to collect all the possible subordinates until you reach the bottom of the tree and then join them together with the help of collectSubordinates.
        public String allSubordinates(String entity) {
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(entity)) {
                    ArrayList<String> subs = new ArrayList<>();
                    collectSubordinates(current, subs);
                    if (subs.isEmpty()) return "none";
                    return String.join(" ", subs);
                }
            }
            return "none";
        }
        
        // Utilizes a similar method to allSubordinates but instead of concacting a string, simply gets the size of the arraylist created.
        public int numberOfAllSubordinates(String entity) {
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(entity)) {
                    ArrayList<String> subs = new ArrayList<>();
                    collectSubordinates(current, subs);
                    return subs.size();
                }
            }
            return 0;
        }
        
        // Gets a list of all the supervisors above an entity and returns it as a string by working back through the Tree, collecting the element, adding that into a list, and 
        // contining until you are at the root node.
        public String allSupervisors(String entity) {
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(entity)) {
                    if (current.getParent() == null) return "none";
                    ArrayList<String> sups = new ArrayList<>();
                    Node p = current.getParent();
                    while (p != null) {
                        sups.add(p.getElement());
                        p = p.getParent();
                    }
                    return String.join(" ", sups);
                }
            }
            return "none";
        }
        
        // Does the same thing as allSupervisors but instead of collecting the element at each node until the root, simply increments number by 1 each time.
        public int numberOfAllSupervisors(String entity) {
            int number = 0;
            for (int i = 0; i < Tree.size(); i++) {
                Node current = Tree.get(i);
                if (current.getElement().equals(entity)) {
                    Node p = current.getParent();
                    while (p != null) {
                        number += 1;
                        p = p.getParent();
                    }
                    return number;
                }
            }
            return 0;    
        }
        
        // Compares two entities and finds their common supervisor. Does this by creating a list of the parents for the first entity and compares it to a list of the parents
        // from the second entity. Then returns the common supervisor it finds first.
        public String closestCommonSupervisor(String entity1, String entity2) {
            Node node1 = null, node2 = null;
            for (Node current : Tree) {
                if (current.getElement().equals(entity1)) node1 = current;
                if (current.getElement().equals(entity2)) node2 = current;
            }
            
            if (node1 == null || node2 == null) return "none";

            ArrayList<Node> ancestors1 = new ArrayList<>();
            Node p1 = node1.getParent();
            while (p1 != null) {
                ancestors1.add(p1);
                p1 = p1.getParent();
            }

            Node p2 = node2.getParent();
            while (p2 != null) {
                if (ancestors1.contains(p2)) {
                    return p2.getElement();
                }
                p2 = p2.getParent();
            }
            return "none";
        }
    }
    
    public static void main(String[] args) {
        // Simple if statement to ensure two files are inputted
        if (args.length < 2) {
            System.err.println("Please provide both data and query file arguments.");
            return;
        }
        // the two files and a new Tree is created here 
        String fileData = args[0];
        String fileQueries = args[1];
        Tree Hiearchy = new Tree();
        
        // This adds the root to the Tree and also eveyr branch in the tree using the data file 
        try (Scanner fileScanner = new Scanner(new File(fileData))) {
            if (fileScanner.hasNext()) {
                String root = fileScanner.next();
                Hiearchy.addRoot(root);
            }
            while (fileScanner.hasNext()) {
                String parent = fileScanner.next();
                String child = fileScanner.next();
                Hiearchy.addBranch(child, parent);
            }    
        } catch (FileNotFoundException e) {
            System.err.println("File name: " + fileData + " was not found!");    
        }
        
        // This is the long chain of if statements that takes in the queries from the file and prints them out accordingly
        try (Scanner fileScanner = new Scanner(new File(fileQueries))) {
            while (fileScanner.hasNext()) {
                String keyword = fileScanner.next();
                if (keyword.equals("DirectSupervisor")) {
                    String entity = fileScanner.next();
                    String supervisor = Hiearchy.getSupervisor(entity);
                    System.out.println("DirectSupervisor " + entity + " " + supervisor);
                }
                else if (keyword.equals("DirectSubordinates")) {
                    String entity = fileScanner.next();  
                    String subordinates = Hiearchy.getSubordinates(entity);
                    System.out.println("DirectSubordinates " + entity + " " + subordinates);
                }
                else if (keyword.equals("AllSupervisors")) {
                    String entity = fileScanner.next();  
                    String supervisors = Hiearchy.allSupervisors(entity);
                    System.out.println("AllSupervisors " + entity + " " + supervisors);
                }
                else if (keyword.equals("AllSubordinates")) {
                    String entity = fileScanner.next();      
                    String subordinates = Hiearchy.allSubordinates(entity);
                    System.out.println("AllSubordinates " + entity + " " + subordinates);
                }
                else if (keyword.equals("NumberOfAllSupervisors")) {
                    String entity = fileScanner.next(); 
                    int number = Hiearchy.numberOfAllSupervisors(entity);
                    System.out.println("NumberOfAllSupervisors " + entity + " " + number);
                }
                else if (keyword.equals("NumberOfAllSubordinates")) {
                    String entity = fileScanner.next();  
                    int number = Hiearchy.numberOfAllSubordinates(entity);
                    System.out.println("NumberOfAllSubordinates " + entity + " " + number);
                }
                else if (keyword.equals("IsSupervisor")) {
                    String entity = fileScanner.next();  
                    String supervisor = fileScanner.next();
                    String check =  Hiearchy.isSupervisor(entity, supervisor);
                    System.out.println("IsSupervisor " + entity + " " + supervisor + " " + check);
                }
                else if (keyword.equals("IsSubordinate")) {
                    String entity = fileScanner.next();  
                    String subordinate = fileScanner.next();
                    String check = Hiearchy.isSubordinate(entity, subordinate);
                    System.out.println("IsSubordinate " + entity + " " + subordinate + " " + check);
                }
                else if (keyword.equals("CompareRank")) {
                    String entity1 = fileScanner.next();  
                    String entity2 = fileScanner.next();  
                    String check = Hiearchy.compareRank(entity1, entity2);
                    System.out.println("CompareRank " + entity1 + " " + entity2 + " " + check);
                }
                else if (keyword.equals("ClosestCommonSupervisor")) {
                    String entity1 = fileScanner.next();  
                    String entity2 = fileScanner.next();  
                    String supervisor = Hiearchy.closestCommonSupervisor(entity1, entity2);
                    System.out.println("ClosestCommonSupervisor " + entity1 + " " + entity2 + " " + supervisor);
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("File name: " + fileQueries + " was not found!");    
        }
    }
}