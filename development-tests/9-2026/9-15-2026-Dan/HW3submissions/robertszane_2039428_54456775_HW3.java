/*
Author: Zane Roberts
Email: zroberts2025@gmail.com
Course: CSE2002
Section: 3
Description of this file: Makes a hieharchy list of subordinates and supervisors and can list who is 
under who and also compare the ranks between two people 
*/

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class HW3 {

    // Tree Node class which stores the entity's name and a reference to its parent
    static class Node {
        String name;
        Node parent;
        List<Node> children;
        
        // Constructor initializes the node name, sets parent to null,
        // and creates an empty ArrayList for children.
        public Node(String name) {
            this.name = name;
            this.parent = null;
            this.children = new ArrayList<>();
        }
    }

    // Holds the root node and conatains helper methods for searching
    static class Tree {
        Node root;

        public Tree() {
            this.root = null;
        }

        // Helper method to recursively search for a node by name
        public Node findNode(Node current, String name) {
            if (current == null) {
                return null;
            }
            if (current.name.equals(name)) {
                return current;
            }
            for (Node child : current.children) {
                Node found = findNode(child, name);
                if (found != null) {
                    return found;
                }
            }
            return null;
        }

        // Adds childNode while maintaining alphabetical order among siblings
        public void addChild(Node parentNode, Node childNode) {
            childNode.parent = parentNode;
            
            // Insert child in alphabetical order
            int index = 0;
            while (index < parentNode.children.size() && 
                   parentNode.children.get(index).name.compareTo(childNode.name) < 0) {
                index++;
            }
            parentNode.children.add(index, childNode);
        }
        
        // returns the list of subordinates for a given node
        public List<Node> getChildren(Node node) {
            return node.children;
        }
        
        // returns the direct supervisor for a given node
        public Node getParent(Node node) {
            return node.parent;
        }

        // finds all direct and indirect subordinates and puts them under a node
        public void getPreOrderSubordinates(Node node, List<Node> result) {
            for (Node child : node.children) {
                result.add(child);
                getPreOrderSubordinates(child, result);
            }
        }

        // calculates depth for rank comparison
        public int getDepth(Node node) {
            int depth = 0;
            Node current = node;
            while (current.parent != null) {
                depth++;
                current = current.parent;
            }
            return depth;
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Error: no file found.");
            return;
        }

        String dataFileName = args[0];
        String queryFileName = args[1];

        Tree orgTree = new Tree();

        // 1. Read and build the tree from the organizational data file
        try {
            Scanner dataScanner = new Scanner(new File(dataFileName));

            // Top entity is on the first line
            if (dataScanner.hasNextLine()) {
                String topName = dataScanner.nextLine().trim();
                if (!topName.isEmpty()) {
                    orgTree.root = new Node(topName);
                }
            }

            // Read each supervisor-subordinate pair and build parent-child links
            while (dataScanner.hasNextLine()) {
                String line = dataScanner.nextLine().trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");
                if (parts.length >= 2) {
                    String supervisorName = parts[0];
                    String subordinateName = parts[1];

                    Node supervisorNode = orgTree.findNode(orgTree.root, supervisorName);
                    Node subordinateNode = orgTree.findNode(orgTree.root, subordinateName);

                    // creates a subordinate if there is none
                    if (subordinateNode == null) {
                        subordinateNode = new Node(subordinateName);
                    }

                    // Attach subordinate to supervisor
                    if (supervisorNode != null) {
                        orgTree.addChild(supervisorNode, subordinateNode);
                    }
                }
            }
            dataScanner.close();
        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + dataFileName);
            return;
        }

        // Read and process queries from the query file
        try {
            Scanner queryScanner = new Scanner(new File(queryFileName));

            while (queryScanner.hasNextLine()) {
                String line = queryScanner.nextLine().trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");
                String command = parts[0];
                
                // Prints the direct parent of the entity or "none" if root/not found
                if (command.equals("DirectSupervisor")) {
                    String entity = parts[1];
                    Node node = orgTree.findNode(orgTree.root, entity);
                    Node parent = (node != null) ? orgTree.getParent(node) : null;
                    if (parent != null) {
                        System.out.println("DirectSupervisor " + entity + " " + parent.name);
                    } else {
                        System.out.println("DirectSupervisor " + entity + " none");
                    }
                
                    // prints all immediate children or none if empty
                } else if (command.equals("DirectSubordinates")) {
                    String entity = parts[1];
                    Node node = orgTree.findNode(orgTree.root, entity);
                    List<Node> children = (node != null) ? orgTree.getChildren(node) : new ArrayList<>();

                    if (children.isEmpty()) {
                        System.out.println("DirectSubordinates " + entity + " none");
                    } else {
                        StringBuilder sb = new StringBuilder("DirectSubordinates ").append(entity);
                        for (Node child : children) {
                            sb.append(" ").append(child.name);
                        }
                        System.out.println(sb.toString());
                    }
                
                    // finds parent nodes to print direct supervisor
                } else if (command.equals("AllSupervisors")) {
                    String entity = parts[1];
                    Node node = orgTree.findNode(orgTree.root, entity);
                    List<String> supervisors = new ArrayList<>();

                    Node current = (node != null) ? orgTree.getParent(node) : null;
                    while (current != null) {
                        supervisors.add(current.name);
                        current = orgTree.getParent(current);
                    }

                    if (supervisors.isEmpty()) {
                        System.out.println("AllSupervisors " + entity + " none");
                    } else {
                        StringBuilder sb = new StringBuilder("AllSupervisors ").append(entity);
                        for (String sup : supervisors) {
                            sb.append(" ").append(sup);
                        }
                        System.out.println(sb.toString());
                    }
                 
                // collects and prints all decendents
                } else if (command.equals("AllSubordinates")) {
                    String entity = parts[1];
                    Node node = orgTree.findNode(orgTree.root, entity);
                    List<Node> subList = new ArrayList<>();
                    if (node != null) {
                        orgTree.getPreOrderSubordinates(node, subList);
                    }

                    if (subList.isEmpty()) {
                        System.out.println("AllSubordinates " + entity + " none");
                    } else {
                        StringBuilder sb = new StringBuilder("AllSubordinates ").append(entity);
                        for (Node sub : subList) {
                            sb.append(" ").append(sub.name);
                        }
                        System.out.println(sb.toString());
                    }
                
                // counts total steps from node to root
                } else if (command.equals("NumberOfAllSupervisors")) {
                    String entity = parts[1];
                    Node node = orgTree.findNode(orgTree.root, entity);
                    int count = 0;
                    Node current = (node != null) ? orgTree.getParent(node) : null;
                    while (current != null) {
                        count++;
                        current = orgTree.getParent(current);
                    }
                    System.out.println("NumberOfAllSupervisors " + entity + " " + count);
                
                // counts total number of items collected
                } else if (command.equals("NumberOfAllSubordinates")) {
                    String entity = parts[1];
                    Node node = orgTree.findNode(orgTree.root, entity);
                    List<Node> subList = new ArrayList<>();
                    if (node != null) {
                        orgTree.getPreOrderSubordinates(node, subList);
                    }
                    System.out.println("NumberOfAllSubordinates " + entity + " " + subList.size());
                
                } else if (command.equals("IsSupervisor")) {
                    String entity = parts[1];
                    String supervisor = parts[2];
                    Node entityNode = orgTree.findNode(orgTree.root, entity);
                    
                    boolean isSup = false;
                    Node current = (entityNode != null) ? orgTree.getParent(entityNode) : null;
                    while (current != null) {
                        if (current.name.equals(supervisor)) {
                            isSup = true;
                            break;
                        }
                        current = orgTree.getParent(current);
                    }
                    System.out.println("IsSupervisor " + entity + " " + supervisor + " " + (isSup ? "yes" : "no"));

                } else if (command.equals("IsSubordinate")) {
                    String entity = parts[1];
                    String subordinate = parts[2];
                    Node entityNode = orgTree.findNode(orgTree.root, entity);
                    
                    List<Node> subList = new ArrayList<>();
                    if (entityNode != null) {
                        orgTree.getPreOrderSubordinates(entityNode, subList);
                    }

                    boolean isSub = false;
                    for (Node sub : subList) {
                        if (sub.name.equals(subordinate)) {
                            isSub = true;
                            break;
                        }
                    }
                    System.out.println("IsSubordinate " + entity + " " + subordinate + " " + (isSub ? "yes" : "no"));

                // compares tree depth of node1 and node2 to check for higher or lower rank
                } else if (command.equals("CompareRank")) {
                    String entity1 = parts[1];
                    String entity2 = parts[2];
                    Node node1 = orgTree.findNode(orgTree.root, entity1);
                    Node node2 = orgTree.findNode(orgTree.root, entity2);

                    int depth1 = (node1 != null) ? orgTree.getDepth(node1) : -1;
                    int depth2 = (node2 != null) ? orgTree.getDepth(node2) : -1;

                    if (depth1 < depth2) {
                        System.out.println("CompareRank " + entity1 + " " + entity2 + " higher");
                    } else if (depth1 > depth2) {
                        System.out.println("CompareRank " + entity1 + " " + entity2 + " lower");
                    } else {
                        System.out.println("CompareRank " + entity1 + " " + entity2 + " same");
                    }

                } else if (command.equals("ClosestCommonSupervisor")) {
                    String entity1 = parts[1];
                    String entity2 = parts[2];
                    Node node1 = orgTree.findNode(orgTree.root, entity1);
                    Node node2 = orgTree.findNode(orgTree.root, entity2);

                    List<Node> supervisors1 = new ArrayList<>();
                    Node current = (node1 != null) ? orgTree.getParent(node1) : null;
                    while (current != null) {
                        supervisors1.add(current);
                        current = orgTree.getParent(current);
                    }

                    Node common = null;
                    Node current2 = (node2 != null) ? orgTree.getParent(node2) : null;
                    while (current2 != null) {
                        if (supervisors1.contains(current2)) {
                            common = current2;
                            break;
                        }
                        current2 = orgTree.getParent(current2);
                    }

                    if (common != null) {
                        System.out.println("ClosestCommonSupervisor " + entity1 + " " + entity2 + " " + common.name);
                    } else {
                        System.out.println("ClosestCommonSupervisor " + entity1 + " " + entity2 + " none");
                    }
                }
            }
            queryScanner.close();
        } catch (FileNotFoundException e) {
            System.out.println("File not found: " + queryFileName);
        }
    }
}
