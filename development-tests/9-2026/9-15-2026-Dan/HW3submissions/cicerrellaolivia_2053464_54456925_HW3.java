/*
Author: Olivia Cicerrella
Email: ocicerrella2025@fit.edu
Course: Algor and data Struc
Section: Section 1
Description of this file:
*/
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class HW3 {

    // Tree Node Structure
    // insures that the nodes are able to compaire them to other treenodes 
    // by using comparable 
    private static class TreeNode implements Comparable<TreeNode> {
        String name;  // the name 
        TreeNode parent; // links up to the sirect super
        List<TreeNode> children; // links down to the subor

        TreeNode(String name) {
            this.name = name;
            this.parent = null;
            this.children = new ArrayList<>();
        }
            // helps to sort the strings inthe alphabetical 
        @Override
        public int compareTo(TreeNode other) {
            return this.name.compareTo(other.name);
        }
    }

    // Main Tree Class to manage the hierarchy 
    //handles the nodes and performs the calculatiosn for the queries
    private static class OrgTree {
        TreeNode root; // the top parent

        OrgTree() {
            this.root = null;
        }
        // set the first top super
        void setRoot(String name) {
            root = new TreeNode(name);
        }
        // adds subor nodes to super's node
        // ensures that it stays in alphbetical order
        void addChild(TreeNode parentNode, TreeNode childNode) {
            childNode.parent = parentNode;
            parentNode.children.add(childNode);
            // Maintain alphabetical order of children
            Collections.sort(parentNode.children);
        }

        // Helper method to find a node by name using recursive depth-first search
        // searches down the tree from given node
        TreeNode findNode(TreeNode current, String name) {
            if (current == null) return null;
            if (current.name.equals(name)) return current;
            
            for (TreeNode child : current.children) {
                TreeNode found = findNode(child, name);
                if (found != null) return found;
            }
            return null;
        }
        
        // search for starting from the root of the tree
        TreeNode getNode(String name) {
            return findNode(root, name);
        }

        // queery inplementations

        // fins immediate manager 
        String getDirectSupervisor(String name) {
            TreeNode node = getNode(name);
            if (node == null || node.parent == null) return "none";
            return node.parent.name;
        }

        // gets the immediate children 
        String getDirectSubordinates(String name) {
            TreeNode node = getNode(name);
            if (node == null || node.children.isEmpty()) return "none";
            StringBuilder sb = new StringBuilder();
            for (TreeNode child : node.children) {
                sb.append(child.name).append(" ");
            }
            return sb.toString().trim();
        }
        
        // collects the ssupers , and lists them in the corect order
        List<String> getAllSupervisorsList(String name) {
            List<String> supervisors = new ArrayList<>();
            TreeNode current = getNode(name);
            if (current == null) return supervisors;
            
            current = current.parent;
            while (current != null) {
                supervisors.add(current.name);
                current = current.parent;
            }
            return supervisors;
        }
        // collects the paretns/ ancestors of them
        String getAllSupervisors(String name) {
            List<String> sups = getAllSupervisorsList(name);
            if (sups.isEmpty()) return "none";
            return String.join(" ", sups);
        }
        // uses recurstion to get all the subor's 
        void preOrderSubordinates(TreeNode node, List<String> result) {
            for (TreeNode child : node.children) {
                result.add(child.name);
                preOrderSubordinates(child, result);
            }
        }
        // // getss and convets all the subors into striings 
        String getAllSubordinates(String name) {
            TreeNode node = getNode(name);
            if (node == null || node.children.isEmpty()) return "none";
            List<String> subs = new ArrayList<>();
            preOrderSubordinates(node, subs);
            return String.join(" ", subs);
        }
        // get the size of the number of supers above 
        int getNumberOfAllSupervisors(String name) {
            return getAllSupervisorsList(name).size();
        }
        // gets teh size of the entire subords below 
        int getNumberOfAllSubordinates(String name) {
            TreeNode node = getNode(name);
            if (node == null) return 0;
            List<String> subs = new ArrayList<>();
            preOrderSubordinates(node, subs);
            return subs.size();
        }
        //  checks of is super: chesk to see if subordname is under makse sure its a super 
        String isSupervisor(String name, String supervisorName) {
            TreeNode current = getNode(name);
            if (current == null) return "no";
            current = current.parent;
            while (current != null) {
                if (current.name.equals(supervisorName)) return "yes";
                current = current.parent;
            }
            return "no";
        }
        // checks to see if it is subord to aother nodes 
        String isSubordinate(String name, String subordinateName) {
            return isSupervisor(subordinateName, name);
        }
        // calculats the deplt comapred to the root that has 1 depth
        int getDepth(TreeNode node) {
            int depth = 0;
            while (node != null) {
                depth++;
                node = node.parent;
            }
            return depth;
        }
        //compairs the depth/ level of two entres to see whictch is higher 
        String compareRank(String name1, String name2) {
            TreeNode node1 = getNode(name1);
            TreeNode node2 = getNode(name2);
            int depth1 = getDepth(node1);
            int depth2 = getDepth(node2);

            if (depth1 < depth2) return "higher";
            if (depth1 > depth2) return "lower";
            return "same";
        }
        // finds the closes common super find the lowest rank boss between the two compared
        //tracs the super line up to a common point 
        String getClosestCommonSupervisor(String name1, String name2) {
            TreeNode node1 = getNode(name1);
            TreeNode node2 = getNode(name2);
            if (node1 == null || node2 == null) return "none";

            List<String> sups1 = getAllSupervisorsList(name1);
            TreeNode current = node2.parent;
            
            while (current != null) {
                if (sups1.contains(current.name)) {
                    return current.name;
                }
                current = current.parent;
            }
            return "none";
        }
    }
    // Main function 
    // gets the filds in from args
    // reads the and prints results 
    public static void main(String[] args) throws Exception {
        // ensures that the args are there so the program can run correctly 
        if (args.length < 2) {
            System.out.println("Error: Please provide organizational data and query files.");
            return;
        }

        String orgFile = args[0]; // takes the first arg
        String queryFile = args[1]; // takes the secomd arg
        OrgTree tree = new OrgTree();

        // get the scanner to read the files
        Scanner orgScanner = new Scanner(new File(orgFile));
        // gets the top enitiy on the first line
        if (orgScanner.hasNextLine()) {
            String rootName = orgScanner.nextLine().trim();
            if (!rootName.isEmpty()) {
                tree.setRoot(rootName);
            }
        }
        // reads line 2 - to the end of the file
        // the lines contain a pair of super and subord
        while (orgScanner.hasNextLine()) {
            String line = orgScanner.nextLine().trim();
            if (line.isEmpty()) continue; // skips life if empty
            String[] parts = line.split("\\s+");// splits line by blank spaces
            if (parts.length >= 2) {
                String supervisor = parts[0];
                String subordinate = parts[1];
                // see if the nodes already exits in the structure 
                TreeNode parentNode = tree.getNode(supervisor);
                TreeNode childNode = tree.getNode(subordinate);
                
                // If child node isn't built yet build it
                if (childNode == null) {
                    childNode = new TreeNode(subordinate);
                }
                
                // Attach the newly constructed node to its designated supervisor
                if (parentNode != null) {
                    tree.addChild(parentNode, childNode);
                }
            }
        }

        // gets a diff scanner to make it clean
        Scanner queryScanner = new Scanner(new File(queryFile));
        //process the queries file using the new scanner
        while (queryScanner.hasNextLine()) {
            String line = queryScanner.nextLine().trim();
            if (line.isEmpty()) continue;
            String[] parts = line.split("\\s+");
            String queryType = parts[0];
            // splits the query line use ing spaces
            
            // get the query to the corect black based on its name
            switch (queryType) {
                case "DirectSupervisor":
                    System.out.println("DirectSupervisor " + parts[1] + " " + tree.getDirectSupervisor(parts[1]));
                    break;
                case "DirectSubordinates":
                    System.out.println("DirectSubordinates " + parts[1] + " " + tree.getDirectSubordinates(parts[1]));
                    break;
                case "AllSupervisors":
                    System.out.println("AllSupervisors " + parts[1] + " " + tree.getAllSupervisors(parts[1]));
                    break;
                case "AllSubordinates":
                    System.out.println("AllSubordinates " + parts[1] + " " + tree.getAllSubordinates(parts[1]));
                    break;
                case "NumberOfAllSupervisors":
                    System.out.println("NumberOfAllSupervisors " + parts[1] + " " + tree.getNumberOfAllSupervisors(parts[1]));
                    break;
                case "NumberOfAllSubordinates":
                    System.out.println("NumberOfAllSubordinates " + parts[1] + " " + tree.getNumberOfAllSubordinates(parts[1]));
                    break;
                case "IsSupervisor":
                    System.out.println("IsSupervisor " + parts[1] + " " + parts[2] + " " + tree.isSupervisor(parts[1], parts[2]));
                    break;
                case "IsSubordinate":
                    System.out.println("IsSubordinate " + parts[1] + " " + parts[2] + " " + tree.isSubordinate(parts[1], parts[2]));
                    break;
                case "CompareRank":
                    System.out.println("CompareRank " + parts[1] + " " + parts[2] + " " + tree.compareRank(parts[1], parts[2]));
                    break;
                case "ClosestCommonSupervisor":
                    System.out.println("ClosestCommonSupervisor " + parts[1] + " " + parts[2] + " " + tree.getClosestCommonSupervisor(parts[1], parts[2]));
                    break;
                default:
                    break;
            }
        }
    }
}