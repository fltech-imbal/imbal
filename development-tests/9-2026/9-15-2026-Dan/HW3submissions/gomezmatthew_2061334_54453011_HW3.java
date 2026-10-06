/*
    Author: Matthew Gomez
    Email: matthewgomez2025@fit.edu
    Course: CSE 2010
    Section: 4
    Description of this file:
    Reads organizational data from an input file (the top entity on the
    first line, then one supervisor and subordinate pair per line) and
    builds a tree of linked nodes, keeping each entity's subordinates in
    alphabetical order. Then reads a second file of queries (such as
    DirectSupervisor, AllSubordinates, IsSupervisor, CompareRank, and
    ClosestCommonSupervisor) and prints the answer to each query, one
    line per query, to the standard output.
*/
import java.io.*;
import java.util.*;

public class HW3 {

    // One node of the organization tree
    static class TreeNode {
        String name;
        TreeNode parent;
        ArrayList<TreeNode> children = new ArrayList<>();
        TreeNode(String name) { this.name = name; }
    }

    // Tree made of linked TreeNodes
    static class Tree {
        TreeNode root;
        HashMap<String, TreeNode> nodes = new HashMap<>();

        TreeNode getNode(String name) { return nodes.get(name); }

        TreeNode addRoot(String name) {
            root = new TreeNode(name);
            nodes.put(name, root);
            return root;
        }

        // adds childNode under node, keeping children in alphabetical order
        void addChild(TreeNode node, TreeNode childNode) {
            childNode.parent = node;
            int i = 0;
            while (i < node.children.size()
                    && node.children.get(i).name.compareTo(childNode.name) < 0) i++;
            node.children.add(i, childNode);
            nodes.put(childNode.name, childNode);
        }

        ArrayList<TreeNode> getChildren(TreeNode node) { return node.children; }

        TreeNode getParent(TreeNode node) { return node.parent; }

        int depth(TreeNode node) {
            int d = 0;
            while (node.parent != null) { node = node.parent; d++; }
            return d;
        }

        // is anc a proper ancestor of node?
        boolean isAncestor(TreeNode anc, TreeNode node) {
            TreeNode p = node.parent;
            while (p != null) {
                if (p == anc) return true;
                p = p.parent;
            }
            return false;
        }

        // pre-order list of all descendants (node itself not included)
        void preOrder(TreeNode node, ArrayList<String> out) {
            for (TreeNode c : node.children) {
                out.add(c.name);
                preOrder(c, out);
            }
        }
    }

    static String join(ArrayList<String> list) {
        if (list.isEmpty()) return "none";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            if (i > 0) sb.append(" ");
            sb.append(list.get(i));
        }
        return sb.toString();
    }

    // remove BOM / invisible characters and extra spaces
    static String clean(String s) {
        return s.replace("\uFEFF", "").replace("\u200B", "").trim();
    }

    public static void main(String[] args) throws IOException {
        Tree tree = new Tree();

        // ---- read organizational data ----
        BufferedReader in = new BufferedReader(new FileReader(args[0]));
        String line = clean(in.readLine());
        tree.addRoot(line);
        while ((line = in.readLine()) != null) {
            line = clean(line);
            if (line.isEmpty()) continue;
            String[] p = line.split("\\s+");
            TreeNode sup = tree.getNode(p[0]);
            if (sup == null) sup = tree.addRoot(p[0]); // safety only
            TreeNode sub = new TreeNode(p[1]);
            tree.addChild(sup, sub);
        }
        in.close();

        // ---- answer queries ----
        BufferedReader q = new BufferedReader(new FileReader(args[1]));
        while ((line = q.readLine()) != null) {
            line = clean(line);
            if (line.isEmpty()) continue;
            String[] p = line.split("\\s+");
            String cmd = p[0];
            TreeNode a = tree.getNode(p[1]);
            String res;

            switch (cmd) {
                case "DirectSupervisor": {
                    TreeNode par = tree.getParent(a);
                    res = (par == null) ? "none" : par.name;
                    break;
                }
                case "DirectSubordinates": {
                    ArrayList<String> l = new ArrayList<>();
                    for (TreeNode c : tree.getChildren(a)) l.add(c.name);
                    res = join(l);
                    break;
                }
                case "AllSupervisors": {
                    ArrayList<String> l = new ArrayList<>();
                    TreeNode par = tree.getParent(a);
                    while (par != null) { l.add(par.name); par = tree.getParent(par); }
                    res = join(l);
                    break;
                }
                case "AllSubordinates": {
                    ArrayList<String> l = new ArrayList<>();
                    tree.preOrder(a, l);
                    res = join(l);
                    break;
                }
                case "NumberOfAllSupervisors":
                    res = "" + tree.depth(a);
                    break;
                case "NumberOfAllSubordinates": {
                    ArrayList<String> l = new ArrayList<>();
                    tree.preOrder(a, l);
                    res = "" + l.size();
                    break;
                }
                case "IsSupervisor": {
                    TreeNode b = tree.getNode(p[2]);
                    res = tree.isAncestor(b, a) ? "yes" : "no";
                    break;
                }
                case "IsSubordinate": {
                    TreeNode b = tree.getNode(p[2]);
                    res = tree.isAncestor(a, b) ? "yes" : "no";
                    break;
                }
                case "CompareRank": {
                    TreeNode b = tree.getNode(p[2]);
                    int da = tree.depth(a), db = tree.depth(b);
                    res = (da < db) ? "higher" : (da > db) ? "lower" : "same";
                    break;
                }
                case "ClosestCommonSupervisor": {
                    TreeNode b = tree.getNode(p[2]);
                    HashSet<TreeNode> sups = new HashSet<>();
                    TreeNode x = tree.getParent(a);
                    while (x != null) { sups.add(x); x = tree.getParent(x); }
                    res = "none";
                    TreeNode y = tree.getParent(b);
                    while (y != null) {
                        if (sups.contains(y)) { res = y.name; break; }
                        y = tree.getParent(y);
                    }
                    break;
                }
                default:
                    continue;
            }
            // echo the query, then the answer
            System.out.println(line + " " + res);
        }
        q.close();
    }
}