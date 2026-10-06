import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

/*
  Author: Tristan Hila
  Email: thila2025@my.fit.edu
  CSE 2010 HW3: answers organization hierarchy queries.
*/
public class HW3 {

    /* A linked tree node. */
    private static class Node {
        String name;
        Node parent;
        Node firstChild;
        Node nextSibling;

        Node(String name) {
            this.name = name;
        }
    }

    /* A tree whose children stay in lexicographical order. */
    private static class Tree {
        private Node root;
        private final Map<String, Node> nodes = new HashMap<String, Node>();

        Node getNode(String name) {
            Node node = nodes.get(name);
            if (node == null) {
                node = new Node(name);
                nodes.put(name, node);
            }
            return node;
        }

        void setRoot(String name) {
            root = getNode(name);
        }

        void addChild(Node parent, Node child) {
            child.parent = parent;
            if (parent.firstChild == null
                    || child.name.compareTo(parent.firstChild.name) < 0) {
                child.nextSibling = parent.firstChild;
                parent.firstChild = child;
                return;
            }

            Node current = parent.firstChild;
            while (current.nextSibling != null
                    && current.nextSibling.name.compareTo(child.name) < 0) {
                current = current.nextSibling;
            }
            child.nextSibling = current.nextSibling;
            current.nextSibling = child;
        }

        List<Node> getChildren(Node node) {
            List<Node> children = new ArrayList<Node>();
            for (Node child = node.firstChild; child != null; child = child.nextSibling) {
                children.add(child);
            }
            return children;
        }

        Node getParent(Node node) {
            return node.parent;
        }

        int depth(Node node) {
            int depth = 0;
            for (Node current = node.parent; current != null; current = current.parent) {
                depth++;
            }
            return depth;
        }
    }

    /* Builds the tree from the organizational data file. */
    private static Tree readTree(String filename) throws FileNotFoundException {
        Tree tree = new Tree();
        Scanner input = new Scanner(new File(filename));
        if (!input.hasNextLine()) {
            input.close();
            return tree;
        }

        tree.setRoot(input.nextLine().trim());
        while (input.hasNext()) {
            String supervisor = input.next();
            if (!input.hasNext()) {
                break;
            }
            String subordinate = input.next();
            tree.addChild(tree.getNode(supervisor), tree.getNode(subordinate));
        }
        input.close();
        return tree;
    }

    /* Adds descendants in pre-order. */
    private static void addSubordinates(Node node, List<Node> result) {
        for (Node child = node.firstChild; child != null; child = child.nextSibling) {
            result.add(child);
            addSubordinates(child, result);
        }
    }

    /* Checks whether possibleSupervisor is above entity. */
    private static boolean isSupervisor(Node possibleSupervisor, Node entity) {
        for (Node current = entity.parent; current != null; current = current.parent) {
            if (current == possibleSupervisor) {
                return true;
            }
        }
        return false;
    }

    /* Finds the lowest shared supervisor of two nodes. */
    private static Node closestCommonSupervisor(Node first, Node second) {
        Map<Node, Boolean> ancestors = new HashMap<Node, Boolean>();
        for (Node current = first.parent; current != null; current = current.parent) {
            ancestors.put(current, true);
        }
        for (Node current = second.parent; current != null; current = current.parent) {
            if (ancestors.containsKey(current)) {
                return current;
            }
        }
        return null;
    }

    /* Formats a list of nodes for output. */
    private static String names(List<Node> nodes) {
        if (nodes.isEmpty()) {
            return "none";
        }
        StringBuilder result = new StringBuilder();
        for (Node node : nodes) {
            if (result.length() > 0) {
                result.append(' ');
            }
            result.append(node.name);
        }
        return result.toString();
    }

    /* Answers every query in the queries file. */
    private static void answerQueries(Tree tree, String filename) throws FileNotFoundException {
        Scanner input = new Scanner(new File(filename));
        while (input.hasNextLine()) {
            String line = input.nextLine().trim();
            if (line.length() == 0) {
                continue;
            }
            String[] parts = line.split("\\s+");
            String query = parts[0];
            Node first = tree.getNode(parts[1]);
            String answer;

            if (query.equals("DirectSupervisor")) {
                Node parent = tree.getParent(first);
                answer = parent == null ? "none" : parent.name;
            } else if (query.equals("DirectSubordinates")) {
                answer = names(tree.getChildren(first));
            } else if (query.equals("AllSupervisors")) {
                List<Node> supervisors = new ArrayList<Node>();
                for (Node current = first.parent; current != null; current = current.parent) {
                    supervisors.add(current);
                }
                answer = names(supervisors);
            } else if (query.equals("AllSubordinates")) {
                List<Node> subordinates = new ArrayList<Node>();
                addSubordinates(first, subordinates);
                answer = names(subordinates);
            } else if (query.equals("NumberOfAllSupervisors")) {
                answer = Integer.toString(tree.depth(first));
            } else if (query.equals("NumberOfAllSubordinates")) {
                List<Node> subordinates = new ArrayList<Node>();
                addSubordinates(first, subordinates);
                answer = Integer.toString(subordinates.size());
            } else if (query.equals("IsSupervisor")) {
                Node supervisor = tree.getNode(parts[2]);
                answer = isSupervisor(supervisor, first) ? "yes" : "no";
            } else if (query.equals("IsSubordinate")) {
                Node subordinate = tree.getNode(parts[2]);
                answer = isSupervisor(first, subordinate) ? "yes" : "no";
            } else if (query.equals("CompareRank")) {
                Node second = tree.getNode(parts[2]);
                int firstDepth = tree.depth(first);
                int secondDepth = tree.depth(second);
                answer = firstDepth < secondDepth ? "higher"
                        : (firstDepth > secondDepth ? "lower" : "same");
            } else if (query.equals("ClosestCommonSupervisor")) {
                Node second = tree.getNode(parts[2]);
                Node supervisor = closestCommonSupervisor(first, second);
                answer = supervisor == null ? "none" : supervisor.name;
            } else {
                continue;
            }
            System.out.println(line + " " + answer);
        }
        input.close();
    }

    /* Reads the two input files and prints answers. */
    public static void main(String[] args) {
        if (args.length != 2) {
            return;
        }
        try {
            answerQueries(readTree(args[0]), args[1]);
        } catch (FileNotFoundException exception) {
            System.err.println("Input file not found.");
        }
    }
}
