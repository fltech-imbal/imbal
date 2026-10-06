/*

  Author: Chris B.
  Email: cblackwell2025@my.fit.edu
  Course: cse2010
  Section: 01
  Description of this file: builds a tree and answers queries on said tree.

 */
import java.io.*;
import java.util.*;
public class HW3
{
    static class TreeNode {
        String name;
        TreeNode parent;
        TreeNode firstChild;
        TreeNode nextSibling;

        TreeNode(String name) {
            this.name = name;
        }
    }

    static class Tree {
        private TreeNode root;
        private HashMap<String, TreeNode> nodes;

        Tree() {
            nodes = new HashMap<String, TreeNode>();
        }

        TreeNode getOrCreate(String name) {
            TreeNode node = nodes.get(name);

            if (node == null) {
                node = new TreeNode(name);
                nodes.put(name, node);
            }

            return node;
        }

        void setRoot(String name) {
            root = getOrCreate(name);
            root.parent = null;
        }

        TreeNode getNode(String name) {
            return nodes.get(name);
        }
        void addChild(TreeNode node, TreeNode childNode) {
            childNode.parent = node;

            // first child
            if (node.firstChild == null) {
                node.firstChild = childNode;
                childNode.nextSibling = null;
                return;
            }

            // inserting before the current first child
            if (childNode.name.compareTo(node.firstChild.name) < 0) {
                childNode.nextSibling = node.firstChild;
                node.firstChild = childNode;
                return;
            }

            // find position
            TreeNode current = node.firstChild;

            while (current.nextSibling != null
                    && current.nextSibling.name.compareTo(childNode.name) < 0) {
                current = current.nextSibling;
            }

            childNode.nextSibling = current.nextSibling;
            current.nextSibling = childNode;
        }

        ArrayList<TreeNode> getChildren(TreeNode node) {
            ArrayList<TreeNode> children = new ArrayList<TreeNode>();

            TreeNode current = node.firstChild;

            while (current != null) {
                children.add(current);
                current = current.nextSibling;
            }

            return children;
        }

        TreeNode getParent(TreeNode node) {
            if (node == null) {
                return null;
            }

            return node.parent;
        }

        ArrayList<TreeNode> getAllSupervisors(TreeNode node) {
            ArrayList<TreeNode> result = new ArrayList<TreeNode>();

            TreeNode current = node.parent;

            while (current != null) {
                result.add(current);
                current = current.parent;
            }

            return result;
        }

        //add descendants in preorder
        void getAllSubordinates(TreeNode node, ArrayList<TreeNode> result) {
            TreeNode child = node.firstChild;

            while (child != null) {
                result.add(child);
                getAllSubordinates(child, result);
                child = child.nextSibling;
            }
        }

        int getDepth(TreeNode node) {
            int depth = 0;
            TreeNode current = node;

            while (current != null && current.parent != null) {
                depth++;
                current = current.parent;
            }

            return depth;
        }

        //return if possibleSupervisor is a supervisor

        boolean isSupervisor(TreeNode entity, TreeNode possibleSupervisor) {
            if (entity == null || possibleSupervisor == null) {
                return false;
            }

            TreeNode current = entity.parent;

            while (current != null) {
                if (current == possibleSupervisor) {
                    return true;
                }

                current = current.parent;
            }

            return false;
        }

        //return if possibleSubordinate is a descendant
        boolean isSubordinate(TreeNode entity, TreeNode possibleSubordinate) {
            return isSupervisor(possibleSubordinate, entity);
        }

        //lowest common ancestor of two nodes
        TreeNode closestCommonSupervisor(TreeNode first, TreeNode second) {
            if (first == null || second == null) {
                return null;
            }

            HashSet<TreeNode> ancestors = new HashSet<TreeNode>();

            TreeNode current = first.parent;

            while (current != null) {
                ancestors.add(current);
                current = current.parent;
            }

            current = second.parent;

            while (current != null) {
                if (ancestors.contains(current)) {
                    return current;
                }

                current = current.parent;
            }

            return null;
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            return;
        }

        String organizationFile = args[0];
        String queryFile = args[1];

        Tree tree = new Tree();

        try {
            readOrganizationFile(organizationFile, tree);
            processQueries(queryFile, tree);
        } catch (IOException e) {
            System.err.println("Error reading file: " + e.getMessage());
        }
    }

    static void readOrganizationFile(String filename, Tree tree)
            throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader(filename));

        String line = reader.readLine();

        if (line == null) {
            reader.close();
            return;
        }

        line = line.trim();

        if (line.length() > 0) {
            tree.setRoot(line);
        }

        while ((line = reader.readLine()) != null) {
            line = line.trim();

            if (line.length() == 0) {
                continue;
            }

            String[] parts = line.split("\\s+");

            if (parts.length < 2) {
                continue;
            }

            String supervisorName = parts[0];
            String subordinateName = parts[1];

            TreeNode supervisor = tree.getOrCreate(supervisorName);
            TreeNode subordinate = tree.getOrCreate(subordinateName);

            if (subordinate.parent == null) {
                tree.addChild(supervisor, subordinate);
            }
        }

        reader.close();
    }

    static void processQueries(String filename, Tree tree)
            throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader(filename));

        String line;

        while ((line = reader.readLine()) != null) {
            line = line.trim();

            if (line.length() == 0) {
                continue;
            }

            String[] parts = line.split("\\s+");

            if (parts.length == 0) {
                continue;
            }

            String query = parts[0];

            if (query.equals("DirectSupervisor")) {
                directSupervisor(parts, tree);
            }
            else if (query.equals("DirectSubordinates")) {
                directSubordinates(parts, tree);
            }
            else if (query.equals("AllSupervisors")) {
                allSupervisors(parts, tree);
            }
            else if (query.equals("AllSubordinates")) {
                allSubordinates(parts, tree);
            }
            else if (query.equals("NumberOfAllSupervisors")) {
                numberOfAllSupervisors(parts, tree);
            }
            else if (query.equals("NumberOfAllSubordinates")) {
                numberOfAllSubordinates(parts, tree);
            }
            else if (query.equals("IsSupervisor")) {
                isSupervisor(parts, tree);
            }
            else if (query.equals("IsSubordinate")) {
                isSubordinate(parts, tree);
            }
            else if (query.equals("CompareRank")) {
                compareRank(parts, tree);
            }
            else if (query.equals("ClosestCommonSupervisor")) {
                closestCommonSupervisor(parts, tree);
            }
        }

        reader.close();
    }

    static void directSupervisor(String[] parts, Tree tree) {
        String entityName = parts[1];
        TreeNode entity = tree.getNode(entityName);

        System.out.print("DirectSupervisor " + entityName);

        TreeNode supervisor = tree.getParent(entity);

        if (supervisor == null) {
            System.out.println(" none");
        } else {
            System.out.println(" " + supervisor.name);
        }
    }

    static void directSubordinates(String[] parts, Tree tree) {
        String entityName = parts[1];
        TreeNode entity = tree.getNode(entityName);

        System.out.print("DirectSubordinates " + entityName);

        ArrayList<TreeNode> children = tree.getChildren(entity);

        if (children.size() == 0) {
            System.out.println(" none");
            return;
        }

        for (TreeNode child : children) {
            System.out.print(" " + child.name);
        }

        System.out.println();
    }

    static void allSupervisors(String[] parts, Tree tree) {
        String entityName = parts[1];
        TreeNode entity = tree.getNode(entityName);

        System.out.print("AllSupervisors " + entityName);

        ArrayList<TreeNode> supervisors = tree.getAllSupervisors(entity);

        if (supervisors.size() == 0) {
            System.out.println(" none");
            return;
        }

        for (TreeNode supervisor : supervisors) {
            System.out.print(" " + supervisor.name);
        }

        System.out.println();
    }

    static void allSubordinates(String[] parts, Tree tree) {
        String entityName = parts[1];
        TreeNode entity = tree.getNode(entityName);

        System.out.print("AllSubordinates " + entityName);

        ArrayList<TreeNode> subordinates = new ArrayList<TreeNode>();
        tree.getAllSubordinates(entity, subordinates);

        if (subordinates.size() == 0) {
            System.out.println(" none");
            return;
        }

        for (TreeNode subordinate : subordinates) {
            System.out.print(" " + subordinate.name);
        }

        System.out.println();
    }

    static void numberOfAllSupervisors(String[] parts, Tree tree) {
        String entityName = parts[1];
        TreeNode entity = tree.getNode(entityName);

        ArrayList<TreeNode> supervisors = tree.getAllSupervisors(entity);

        System.out.println("NumberOfAllSupervisors "
                + entityName + " " + supervisors.size());
    }

    static void numberOfAllSubordinates(String[] parts, Tree tree) {
        String entityName = parts[1];
        TreeNode entity = tree.getNode(entityName);

        ArrayList<TreeNode> subordinates = new ArrayList<TreeNode>();
        tree.getAllSubordinates(entity, subordinates);

        System.out.println("NumberOfAllSubordinates "
                + entityName + " " + subordinates.size());
    }

    static void isSupervisor(String[] parts, Tree tree) {
        String entityName = parts[1];
        String supervisorName = parts[2];

        TreeNode entity = tree.getNode(entityName);
        TreeNode supervisor = tree.getNode(supervisorName);

        boolean result = tree.isSupervisor(entity, supervisor);

        System.out.println("IsSupervisor " + entityName + " "
                + supervisorName + " " + (result ? "yes" : "no"));
    }

    static void isSubordinate(String[] parts, Tree tree) {
        String entityName = parts[1];
        String subordinateName = parts[2];

        TreeNode entity = tree.getNode(entityName);
        TreeNode subordinate = tree.getNode(subordinateName);

        boolean result = tree.isSubordinate(entity, subordinate);

        System.out.println("IsSubordinate " + entityName + " "
                + subordinateName + " " + (result ? "yes" : "no"));
    }

    static void compareRank(String[] parts, Tree tree) {
        String firstName = parts[1];
        String secondName = parts[2];

        TreeNode first = tree.getNode(firstName);
        TreeNode second = tree.getNode(secondName);

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

        System.out.println("CompareRank " + firstName + " "
                + secondName + " " + result);
    }

    static void closestCommonSupervisor(String[] parts, Tree tree) {
        String firstName = parts[1];
        String secondName = parts[2];

        TreeNode first = tree.getNode(firstName);
        TreeNode second = tree.getNode(secondName);

        TreeNode common = tree.closestCommonSupervisor(first, second);

        System.out.print("ClosestCommonSupervisor " + firstName
                + " " + secondName);

        if (common == null) {
            System.out.println(" none");
        } else {
            System.out.println(" " + common.name);
        }
    }
}
