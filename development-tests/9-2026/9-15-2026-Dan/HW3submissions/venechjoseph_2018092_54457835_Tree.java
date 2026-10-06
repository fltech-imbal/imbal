/*

  Author: Joseph Venech
  Email: jvenech2025@my.fit.edu
  Course: CSE 2010
  Section: 1
  Description of this file: Stores an organization as a tree and has methods for finding 
    supervisors, subordinates, ranks, and common supervisors

 */

import java.util.ArrayList;

public class Tree {
    private TreeNode root;
    private ArrayList<TreeNode> nodes;
    
    //Creates the root node and add to list of nodes
    public Tree(String rootName) {
        root = new TreeNode(rootName);
        nodes = new ArrayList<>();
        nodes.add(root);
    }

    //Search list of nodes for node with given name
    private TreeNode findNode(String name) {
        for (int i = 0; i < nodes.size(); i++) {
            if (nodes.get(i).data.equals(name)) {
                return nodes.get(i);
            }
        }

        return null;
    }

    //Adds a new child under a given parent
    public void addChild(String parentName, String childName) {
        TreeNode parent = findNode(parentName);

        if (parent == null) {
            return;
        }

        TreeNode child = new TreeNode(childName);
        child.parent = parent;

        int position = 0;

        //Find the correct alphabetical position for the child
        while (position < parent.children.size() && parent.children.get(position).data.compareTo(childName) < 0) {
            position++;
        }

        parent.children.add(position, child);
        nodes.add(child);
    }

    //Returns the children of a given node
    public ArrayList<TreeNode> getChildren(String name) {
        TreeNode node = findNode(name);

        if (node == null) {
            return new ArrayList<TreeNode>();
        }

        return node.children;
    }

    //Returns the parent of the given node
    public TreeNode getParent(String name) {
        TreeNode node = findNode(name);

        if (node == null) {
            return null;
        }

        return node.parent;
    }

    //Returns the direct parent of an entity
    public String directSupervisor(String name) {
        TreeNode parent = getParent(name);

        if (parent == null) {
            return "none";
        }

        return parent.data;
    }

    //Returns the direct children of an entity
    public String directSubordinates(String name) {
        TreeNode node = findNode(name);

        if (node == null || node.children.size() == 0) {
            return "none";
        }

        String result = "";

        for (int i = 0; i < node.children.size(); i++) {
            if (i > 0) {
                result += " ";
            }

            result += node.children.get(i).data;
        }

        return result;
    }

    //Returns all parents above an entity
    public String allSupervisors(String name) {
        TreeNode node = findNode(name);

        if (node == null || node.parent == null) {
            return "none";
        }

        String result = "";
        TreeNode current = node.parent;

        while (current != null) {
            if (result.length() > 0) {
                result += " ";
            }

            result += current.data;
            current = current.parent;
        }

        return result;
    }

    //Returns all children below an entity
    //pre-order traversal
    public String allSubordinates(String name) {
        TreeNode node = findNode(name);

        if (node == null || node.children.size() == 0) {
            return "none";
        }

        ArrayList<String> result = new ArrayList<String>();

        preOrder(node, result, false);

        String output = "";

        for (int i = 0; i < result.size(); i++) {
            if (i > 0) {
                output += " ";
            }

            output += result.get(i);
        }

        return output;
    }

    //Visits the nodes in pre-order and adds their names to result list
    private void preOrder(TreeNode node, ArrayList<String> result, boolean includeNode) {

        if (includeNode) {
            result.add(node.data);
        }

        for (int i = 0; i < node.children.size(); i++) {
            TreeNode child = node.children.get(i);

            result.add(child.data);

            preOrder(child, result, false);
        }
    }

    //Counts number of parents above an entity
    public int numberOfAllSupervisors(String name) {
        TreeNode node = findNode(name);

        if (node == null) {
            return 0;
        }

        int count = 0;
        TreeNode current = node.parent;

        while (current != null) {
            count++;
            current = current.parent;
        }

        return count;
    }

    //Counts all children below an entity
    public int numberOfAllSubordinates(String name) {
        TreeNode node = findNode(name);

        if (node == null) {
            return 0;
        }

        return countSubordinates(node);
    }

    //Counts every child and descendent of given node
    private int countSubordinates(TreeNode node) {
        int count = 0;

        for (int i = 0; i < node.children.size(); i++) {
            count++;
            count += countSubordinates(node.children.get(i));
        }

        return count;
    }
    
    //Checks if given supervisor is above entity
    public boolean isSupervisor(String entity, String supervisor) {
        TreeNode node = findNode(entity);

        if (node == null) {
            return false;
        }

        TreeNode current = node.parent;

        while (current != null) {
            if (current.data.equals(supervisor)) {
                return true;
            }

            current = current.parent;
        }

        return false;
    }

    //Checks if given subordinate is below entity
    public boolean isSubordinate(String entity, String subordinate) {
        TreeNode node = findNode(entity);

        if (node == null) {
            return false;
        }

        return isSubordinateHelper(node, subordinate);
    }

    //Searches through children of a node to find specific child
    private boolean isSubordinateHelper(TreeNode node, String subordinate) {
        for (int i = 0; i < node.children.size(); i++) {
            TreeNode child = node.children.get(i);

            if (child.data.equals(subordinate)) {
                return true;
            }

            if (isSubordinateHelper(child, subordinate)) {
                return true;
            }
        }

        return false;
    }

    //Compares entities based on number of supervisors
    public String compareRank(String entity1, String entity2) {
        int rank1 = numberOfAllSupervisors(entity1);
        int rank2 = numberOfAllSupervisors(entity2);

        //Fewer supervisors means entity is higher in the tree
        if (rank1 < rank2) {
            return "higher";
        } else if (rank1 > rank2) {
            return "lower";
        } else {
            return "same";
        }
    }

    //Finds closest supervisor shared by two entities
    public String closestCommonSupervisor(String entity1, String entity2) {
        TreeNode node1 = findNode(entity1);
        TreeNode node2 = findNode(entity2);

        if (node1 == null || node2 == null) {
            return "none";
        }

        //All supervisors of the first entity
        ArrayList<String> supervisors = new ArrayList<String>();

        TreeNode current = node1.parent;

        while (current != null) {
            supervisors.add(current.data);
            current = current.parent;
        }
        
        //Check the second entity's supervisor aganist the first
        current = node2.parent;

        while (current != null) {
            for (int i = 0; i < supervisors.size(); i++) {
                if (supervisors.get(i).equals(current.data)) {
                    return current.data;
                }
            }

            current = current.parent;
        }

        return "none";
    }
}