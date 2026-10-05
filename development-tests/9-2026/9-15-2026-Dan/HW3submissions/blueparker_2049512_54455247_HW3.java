/*
 Author: Parker Blue
 Email: pblue2025@my.fit.edu
 Course: CSE2010
 Section:
 Description: This program builds a tree that represents an organization's hierarcy and then
 answers queries about said hierarchy. This will be done using two files, one with the
 organizational data, and one with the queries (one per line). 
*/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
 
public class HW3 {
 
    /*
     * Variables:
     *   tree  - the organization's hierarchy with its root  being the top entity
     *   nodes - maps each entity's name to its node in the tree so a name can be read faster
     */
    private static final Tree tree = new Tree();
    private static final Map<String, Tree.Node> nodes = new HashMap<>();
 
    /**
     * Entry point. Builds the tree from the organizational data file, then
     * reads the query file line by line and prints each query's answer.
     *
     * parameter args - args[0] = filename of the organizational data,
     *                  args[1] = filename of the queries
     * throws IOException if either file cannot be opened or read
     */
    public static void main(String[] args) throws IOException {
        // Block 1: make sure both filenames were provided, print usage and stop if not
        if (args.length < 2) {
            System.err.println("Usage: java HW3 <orgDataFile> <queryFile>");
            return;
        }
 
        // Block 2: build the organization tree from the first file
        buildTree(args[0]);
 
        // Block 3: answer the queries
        //   out  - collects every answer line so the output can be printed at the end at one time
        //   line - the current raw line from the query file
        //   tok  - line split into words
        // Blank lines are skipped and unknown queries return null and are also skipped
        StringBuilder out = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(args[1]))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tok = tokenize(line);
                if (tok.length == 0) continue;
                String answer = answer(tok);
                if (answer != null) out.append(answer).append('\n');
            }
        }
 
        // Block 4: print all answers
        System.out.print(out);
    }
 
    /*
     * Splits one line of input into its words and removes invisible characters.
     *
     * parameter line - one raw line read from an input file
     * return the words on the line or an empty array if the line is blank
     */
    private static String[] tokenize(String line) {
        // Block 1: clean the line by reomving byte-order marks, turn non-breaking spaces into
        // normal spaces, and remove leading and trailing whitespace.
        // A line with nothing left is blank.
        line = line.replace("\uFEFF", "").replace('\u00A0', ' ').trim();
        if (line.isEmpty()) return new String[0];
 
        // Block 2: split the line into words
        //   raw - the pieces after splitting
        //   tok - the final list of words, with each piece trimmed
        String[] raw = line.contains("\t") ? line.split("\t+") : line.split("\\s+");
        List<String> tok = new ArrayList<>();
        for (String s : raw) {
            s = s.trim();
            if (!s.isEmpty()) tok.add(s);
        }
        return tok.toArray(new String[0]);
    }
 
    /**
     * Returns the node for an entity name and creating it the first time the name is seen.
     * This lets a pair line refer to an entity before or after that entity has been linked into the
     * tree.
     * parameter name - the entity's name
     * return the one node that represents this entity
     */
    private static Tree.Node getOrCreate(String name) {
        return nodes.computeIfAbsent(name, Tree.Node::new);
    }
 
    /**
     * Reads the organizational data file and builds the tree. The first
     * non-blank line is the top entity (the root). Every later line is a
     * "supervisor subordinate" pair, which links the subordinate as a child
     * of the supervisor.
     *
     * parameter filename - name of the organizational data file
     * throws IOException if the file cannot be opened or read
     */
    private static void buildTree(String filename) throws IOException {
        // Variables:
        //   line  - the current raw line from the file.
        //   first - true until the first non-blank line is read.
        //   tok   - the words on the current line.
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                String[] tok = tokenize(line);
                if (tok.length == 0) continue;
 
                // Block: the first line sets the root. Every other line
                // with at least two words adds a subordinate link.
                if (first) {
                    tree.setRoot(getOrCreate(tok[0]));
                    first = false;
                } else if (tok.length >= 2) {
                    tree.addChild(getOrCreate(tok[0]), getOrCreate(tok[1]));
                }
            }
        }
    }
  
    /*
     * Answers one query and builds its output line, which is the query repeated exactly, then a space, then the answer.
     *
     * parameter tok - the words of the query line
     * return the full output line, or null if the query name is unknown
     */
    private static String answer(String[] tok) {
        // Block 1: set up the query.
        //   q      - the query name.
        //   a      - node of the first entity (null if none was given).
        //   b      - node of the second entity (null if none was given).
        //   prefix - the query words joined back together; every answer
        //            line starts with it.
        String q = tok[0];
        Tree.Node a = tok.length > 1 ? nodes.get(tok[1]) : null;
        Tree.Node b = tok.length > 2 ? nodes.get(tok[2]) : null;
        String prefix = String.join(" ", tok);
 
        // Block 2: choose the answer based on the query name
        switch (q) {
            // The parent of a, or "none" if a is the top entity.
            case "DirectSupervisor": {
                Tree.Node p = tree.getParent(a);
                return prefix + " " + (p == null ? "none" : p.name);
            }
            // a's children
            case "DirectSubordinates":
                return prefix + " " + joinOrNone(tree.getChildren(a));
            // Every ancestor of a
            case "AllSupervisors":
                return prefix + " " + joinOrNone(allSupervisors(a));
            // Every descendant of a
            case "AllSubordinates":
                return prefix + " " + joinOrNone(allSubordinates(a));
            // Counts of the two lists above
            case "NumberOfAllSupervisors":
                return prefix + " " + allSupervisors(a).size();
            case "NumberOfAllSubordinates":
                return prefix + " " + allSubordinates(a).size();
            // Is b a supervisor of a?
            case "IsSupervisor":
                return prefix + " " + (isAncestor(b, a) ? "yes" : "no");
            // Is b a subordinate (at any level) of a?
            case "IsSubordinate":
                return prefix + " " + (isAncestor(a, b) ? "yes" : "no");
            // Smaller depth means closer to the top, which is a higher rank.
            //   da, db - depths of a and b
            case "CompareRank": {
                int da = depth(a), db = depth(b);
                String r = da < db ? "higher" : (da > db ? "lower" : "same");
                return prefix + " " + r;
            }
            // The lowest entity that supervises both a and b.
            case "ClosestCommonSupervisor": {
                Tree.Node c = closestCommonSupervisor(a, b);
                return prefix + " " + (c == null ? "none" : c.name);
            }
            // Unknown query name
            default:
                return null;
        }
    }
 
    /**
     * Lists every subordinate of an entity, from its direct subordinate up to
     * the top entity.
     *
     * parameter n - the entity's node
     * return the subordinates in order
     */
    private static List<Tree.Node> allSupervisors(Tree.Node n) {
        // Block: start at n's parent and follow parent links until reaching null, adding each node on the way.
        //   list - the subordinates found so far.
        //   p    - the node currently being visited.
        List<Tree.Node> list = new ArrayList<>();
        for (Tree.Node p = tree.getParent(n); p != null; p = tree.getParent(p)) list.add(p);
        return list;
    }
 
    /**
     * Lists every subordinate of an entity in pre-order: each node comes
     * before its own subordinates, and siblings are listed alphabetically.
     *
     * parameter n - the entity's node
     * return the subordinates in pre-order; an empty list if n has none
     */
    private static List<Tree.Node> allSubordinates(Tree.Node n) {
        // Block 1: set up the traversal
        //   list  - the subordinates in the order they are visited.
        //   stack - the nodes still waiting to be visited
        List<Tree.Node> list = new ArrayList<>();
        Deque<Tree.Node> stack = new ArrayDeque<>();
        List<Tree.Node> kids = tree.getChildren(n);
        for (int i = kids.size() - 1; i >= 0; i--) stack.push(kids.get(i));
 
        // Block 2: pre-order traversal
        // Pop a node and record it, then push its children in reverse
        // order.
        //   cur - the node being visited.
        //   ck  - cur's children.
        while (!stack.isEmpty()) {
            Tree.Node cur = stack.pop();
            list.add(cur);
            List<Tree.Node> ck = tree.getChildren(cur);
            for (int i = ck.size() - 1; i >= 0; i--) stack.push(ck.get(i));
        }
        return list;
    }
 
    /**
     * Checks whether one entity is a subordinate of another at any level
     * An entity is never considered its own supervisor
     *
     * parameter anc - the possible subordinate
     * parameter n - the entity whose supervisors are checked
     * return true if anc appears somewhere above n in the tree
     */
    private static boolean isAncestor(Tree.Node anc, Tree.Node n) {
        // Block: walk upward from n's parent toward the root. If anc is
        // met along the way it is a subordinate of n.
        //   p - the node currently being checked.
        for (Tree.Node p = tree.getParent(n); p != null; p = tree.getParent(p)) {
            if (p == anc) return true;
        }
        return false;
    }
 
    /**
     * Computes an entity's depth, meaning how many levels it sits below the
     * top entity.
     *
     * parameter n - the entity's node
     * return 0 for the top entity, 1 for its direct subordinates, etc.
     */
    private static int depth(Tree.Node n) {
        // Block: count the parent links from n to the root.
        //   d - the number of links counted so far
        //   p - the currently viewed node
        int d = 0;
        for (Tree.Node p = tree.getParent(n); p != null; p = tree.getParent(p)) d++;
        return d;
    }
 
    /**
     * Finds the closest common subordinate of two entities.
     *
     * parameter a - the first entity's node
     * parameter b - the second entity's node
     * return the closest common subordinate, or null if there is none
     */
    private static Tree.Node closestCommonSupervisor(Tree.Node a, Tree.Node b) {
        // Block: store all of a's subordinates in a set, then move up from b's parent
        //   supsOfA - every supervisor of a
        //   p - the supervisor of b currently being checked
        Set<Tree.Node> supsOfA = new HashSet<>(allSupervisors(a));
        for (Tree.Node p = tree.getParent(b); p != null; p = tree.getParent(p)) {
            if (supsOfA.contains(p)) return p;
        }
        return null;
    }
 
    /**
     * Joins the names of a list of nodes with single spaces.
     *
     * parameter list-  the nodes to print
     * return the names separated by spaces, or "none" if the list is empty
     */
    private static String joinOrNone(List<Tree.Node> list) {
        // Block: an empty list prints "none". Otherwise append each name,putting a space before every name except the
        // first.
        // sb - builds the output string.
        if (list.isEmpty()) return "none";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(' ');
            sb.append(list.get(i).name);
        }
        return sb.toString();
    }
}
 
/*
 * Tree: a general tree made of linked nodes. Each node knows its parent and
 * keeps its children in alphabetical order.
 */
class Tree {
 
    /*
     * Node: one entity in the organization
     * Variables:
     *   name - the entity's name
     *   parent - the entity's direct subordinate; null for the root
     *   children - the entity's direct subordinates, kept in alphabetical order by addChild.
     */
    static class Node {
        final String name;
        Node parent;
        final List<Node> children = new ArrayList<>();
 
        /**
         * Creates a node with no parent and no children
         *
         * parameter name - the entity's name
         */
        Node(String name) {
            this.name = name;
        }
    }
 
    // Variable: root - the top entity of the organization
    private Node root;
 

    //return the root of the tree
    Node getRoot() { return root; }
 
    // Sets the root  of the tree.
    // parameter root - the node to become the root
    void setRoot(Node root) { this.root = root; }
 
    /*
     * Makes childNode a child of node, inserting it at the position that
     * keeps node's children in alphabetical (lexicographical) order.
     *
     * parameter node - the parent
     * parameter childNode - the new child
     */
    void addChild(Node node, Node childNode) {
        // Block: link the child to its parent, then find the insertion
        // point. Inserting there keeps the list sorted, so it never needs to be sorted later.
        // kids - node's current list of children.
        // i - the index where childNode will be inserted.
        childNode.parent = node;
        List<Node> kids = node.children;
        int i = 0;
        while (i < kids.size() && kids.get(i).name.compareTo(childNode.name) < 0) i++;
        kids.add(i, childNode);
    }
 
    /**
     * parameter node - the node whose children are wanted return node's children in alphabetical 
     * order
     */
    List<Node> getChildren(Node node) { return node.children; }
 
    /**
     * parameter node - the node whose parent is wanted return node's parent, or null if node is the
     * root
     */
    Node getParent(Node node) { return node.parent; }
}
 














