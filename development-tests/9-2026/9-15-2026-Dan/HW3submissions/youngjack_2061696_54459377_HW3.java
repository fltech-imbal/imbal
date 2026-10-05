/*
* Author: Jack Young
* Email: jack2025@fit.edu
* Course: CSE2010: Algorithms & Data Struct
* Section: 1
*
* Description of this file:
 * Reads an organizational hierarchy and answers questions about supervisors, subordinates, and employee rank.
*/
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class HW3 {
    static class Node {
        String name;
        Node parent;
        Node firstChild;
        Node nextSibling;

        Node(String name) {
            this.name = name;
        }
    }

    static class Tree {
        Node root;
        HashMap<String, Node> nodes = new HashMap<>();

        Tree(String rootName) {
            root = new Node(rootName);
            nodes.put(rootName, root);
        }

        Node getNode(String name) {
            return nodes.get(name);
        }

        void addChild(Node parent, Node child) {
            child.parent = parent;

            if (parent.firstChild == null) {
                parent.firstChild = child;
                return;
            }

            if (child.name.compareTo(parent.firstChild.name) < 0) {
                child.nextSibling = parent.firstChild;
                parent.firstChild = child;
                return;
            }

            Node current = parent.firstChild;

            while (current.nextSibling != null && current.nextSibling.name.compareTo(child.name) < 0) {
                current = current.nextSibling;
            }

            child.nextSibling = current.nextSibling;
            current.nextSibling = child;
        }

        List<Node> getChildren(Node node) {
            ArrayList<Node> children = new ArrayList<>();

            Node current = node.firstChild;

            while (current != null) {
                children.add(current);
                current = current.nextSibling;
            }

            return children;
        }

        Node getParent(Node node) {
            return node.parent;
        }

        List<Node> getSupervisors(Node node) {
            ArrayList<Node> supervisors = new ArrayList<>();

            Node current = node.parent;

            while (current != null) {
                supervisors.add(current);
                current = current.parent;
            }

            return supervisors;
        }

        void getSubordinates(Node node, List<Node> result) {
            Node child = node.firstChild;

            while (child != null) {
                result.add(child);
                getSubordinates(child, result);
                child = child.nextSibling;
            }
        }

        int getDepth(Node node) {
            int depth = 0;

            while (node.parent != null) {
                depth++;
                node = node.parent;
            }

            return depth;
        }

        boolean isSupervisor(Node supervisor, Node employee) {
            Node current = employee.parent;

            while (current != null) {
                if (current == supervisor) {
                    return true;
                }

                current = current.parent;
            }

            return false;
        }

        boolean isSubordinate(Node supervisor, Node employee) {
            return isSupervisor(supervisor, employee);
        }

        Node closestCommonSupervisor(Node first, Node second) {
            ArrayList<Node> firstSupervisors = new ArrayList<>();

            Node current = first.parent;

            while (current != null) {
                firstSupervisors.add(current);
                current = current.parent;
            }

            current = second.parent;

            while (current != null) {
                for (int i = 0; i < firstSupervisors.size(); i++) {
                    Node supervisor = firstSupervisors.get(i);

                    if (supervisor == current) {
                        return current;
                    }
                }

                current = current.parent;
            }

            return null;
        }
    }

    public static void main(String[] args) throws IOException {
        Tree tree = readData(args[0]);
        answerQueries(tree, args[1]);
    }

    static Tree readData(String filename) throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader(filename));

        String rootName = reader.readLine();

        Tree tree = new Tree(rootName);

        String line;

        while ((line = reader.readLine()) != null) {

            String[] parts = line.trim().split("\\s+");

            String supervisorName = parts[0];
            String subordinateName = parts[1];

            Node supervisor = tree.getNode(supervisorName);

            if (supervisor == null) {
                supervisor = new Node(supervisorName);
                tree.nodes.put(supervisorName, supervisor);
            }

            Node subordinate = tree.getNode(subordinateName);

            if (subordinate == null) {
                subordinate = new Node(subordinateName);
                tree.nodes.put(subordinateName, subordinate);
            }

            tree.addChild(supervisor, subordinate);
        }


        return tree;
    }

    static void answerQueries(Tree tree, String filename) throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader(filename));

        String line;

        while ((line = reader.readLine()) != null) {

            String[] parts = line.trim().split("\\s+");

            String query = parts[0];

            Node first = tree.getNode(parts[1]);

            if (query.equals("DirectSupervisor")) {

                Node parent = tree.getParent(first);

                if (parent == null) {
                    System.out.println("DirectSupervisor " + first.name + " none");
                }
                else {
                    System.out.println("DirectSupervisor " + first.name + " " + parent.name);
                }
            }

            else if (query.equals("DirectSubordinates")) {

                System.out.print("DirectSubordinates " + first.name);

                List<Node> children = tree.getChildren(first);

                if (children.isEmpty()) {
                    System.out.print(" none");
                }
                else {
                    for (int i = 0; i < children.size(); i++) {
                        Node child = children.get(i);
                        System.out.print(" " + child.name);
                    }
                }

                System.out.println();
            }

            else if (query.equals("AllSupervisors")) {

                System.out.print("AllSupervisors " + first.name);

                List<Node> supervisors = tree.getSupervisors(first);

                if (supervisors.isEmpty()) {
                    System.out.print(" none");
                }
                else {
                    for (int i = 0; i < supervisors.size(); i++) {
                        Node supervisor = supervisors.get(i);
                        System.out.print(" " + supervisor.name);
                    }
                }

                System.out.println();
            }

            else if (query.equals("AllSubordinates")) {

                System.out.print("AllSubordinates " + first.name);

                ArrayList<Node> subordinates = new ArrayList<>();

                tree.getSubordinates(first, subordinates);

                if (subordinates.isEmpty()) {
                    System.out.print(" none");
                }
                else {
                    for (int i = 0; i < subordinates.size(); i++) {
                        Node subordinate = subordinates.get(i);
                        System.out.print(" " + subordinate.name);
                    }
                }

                System.out.println();
            }

            else if (query.equals("NumberOfAllSupervisors")) {

                int count = tree.getSupervisors(first).size();

                System.out.println("NumberOfAllSupervisors " + first.name + " " + count);
            }

            else if (query.equals("NumberOfAllSubordinates")) {

                ArrayList<Node> subordinates = new ArrayList<>();

                tree.getSubordinates(first, subordinates);

                System.out.println("NumberOfAllSubordinates " + first.name + " " + subordinates.size());
            }

            else if (query.equals("IsSupervisor")) {

                Node second = tree.getNode(parts[2]);

                boolean answer = tree.isSupervisor(second, first);

                if (answer) {
                    System.out.println("IsSupervisor " + first.name + " " + second.name + " yes");
                }
                else {
                    System.out.println("IsSupervisor " + first.name + " " + second.name + " no");
                }
            }

            else if (query.equals("IsSubordinate")) {

                Node second = tree.getNode(parts[2]);

                boolean answer = tree.isSubordinate(first, second);

                if (answer) {
                    System.out.println("IsSubordinate " + first.name + " " + second.name + " yes");
                }
                else {
                    System.out.println("IsSubordinate " + first.name + " " + second.name + " no");
                }
            }

            else if (query.equals("CompareRank")) {

                Node second = tree.getNode(parts[2]);

                int firstDepth = tree.getDepth(first);
                int secondDepth = tree.getDepth(second);

                String result;

                if (firstDepth < secondDepth) {
                    result = "higher";
                }
                else if (firstDepth > secondDepth) {
                    result = "lower";
                }
                else {
                    result = "same";
                }

                System.out.println("CompareRank " + first.name + " " + second.name + " " + result);
            }

            else if (query.equals("ClosestCommonSupervisor")) {

                Node second = tree.getNode(parts[2]);

                Node common = tree.closestCommonSupervisor(first, second);

                if (common == null) {
                    System.out.println("ClosestCommonSupervisor " + first.name + " " + second.name + " none");
                }
                else {
                    System.out.println("ClosestCommonSupervisor " + first.name + " " + second.name + " " + common.name);
                }
            }
        }

    }
}