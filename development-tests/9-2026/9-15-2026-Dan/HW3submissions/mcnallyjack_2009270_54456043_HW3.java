import java.io.*;
import java.util.*;

public class HW3 {

    // Tree node
    static class TreeNode {
        String name;
        TreeNode parent;
        ArrayList<TreeNode> children;

        TreeNode(String name) {
            this.name = name;
            this.parent = null;
            this.children = new ArrayList<>();
        }
    }

    // Tree class
    static class Tree {
        private TreeNode root;
        private HashMap<String, TreeNode> nodes;

        Tree(String rootName) {
            root = new TreeNode(rootName);
            nodes = new HashMap<>();
            nodes.put(rootName, root);
        }

        public TreeNode getNode(String name) {
            return nodes.get(name);
        }

        // Adds child while maintaining alphabetical order
        public void addChild(TreeNode node, TreeNode childNode) {
            childNode.parent = node;

            int index = 0;
            while (index < node.children.size()
                    && node.children.get(index).name.compareTo(childNode.name) < 0) {
                index++;
            }

            node.children.add(index, childNode);
            nodes.put(childNode.name, childNode);
        }

        public ArrayList<TreeNode> getChildren(TreeNode node) {
            return node.children;
        }

        public TreeNode getParent(TreeNode node) {
            return node.parent;
        }

        // Used when constructing the tree
        public void addChild(String parentName, String childName) {
            TreeNode parent = nodes.get(parentName);

            if (parent == null) {
                return;
            }

            TreeNode child = new TreeNode(childName);
            addChild(parent, child);
        }

        // DirectSupervisor
        public String directSupervisor(String entity) {
            TreeNode node = nodes.get(entity);

            if (node == null || node.parent == null) {
                return "none";
            }

            return node.parent.name;
        }

        // DirectSubordinates
        public String directSubordinates(String entity) {
            TreeNode node = nodes.get(entity);

            if (node == null || node.children.isEmpty()) {
                return "none";
            }

            StringBuilder result = new StringBuilder();

            for (TreeNode child : node.children) {
                if (result.length() > 0) {
                    result.append(" ");
                }
                result.append(child.name);
            }

            return result.toString();
        }

        // AllSupervisors
        public String allSupervisors(String entity) {
            TreeNode node = nodes.get(entity);

            if (node == null || node.parent == null) {
                return "none";
            }

            StringBuilder result = new StringBuilder();
            TreeNode current = node.parent;

            while (current != null) {
                if (result.length() > 0) {
                    result.append(" ");
                }

                result.append(current.name);
                current = current.parent;
            }

            return result.toString();
        }

        // AllSubordinates in preorder
        public String allSubordinates(String entity) {
            TreeNode node = nodes.get(entity);

            if (node == null || node.children.isEmpty()) {
                return "none";
            }

            StringBuilder result = new StringBuilder();

            preorder(node, result, false);

            return result.toString();
        }

        private void preorder(TreeNode node, StringBuilder result, boolean includeNode) {
            if (includeNode) {
                if (result.length() > 0) {
                    result.append(" ");
                }
                result.append(node.name);
            }

            for (TreeNode child : node.children) {
                preorder(child, result, true);
            }
        }

        // NumberOfAllSupervisors
        public int numberOfAllSupervisors(String entity) {
            TreeNode node = nodes.get(entity);

            int count = 0;
            TreeNode current = node.parent;

            while (current != null) {
                count++;
                current = current.parent;
            }

            return count;
        }

        // NumberOfAllSubordinates
        public int numberOfAllSubordinates(String entity) {
            TreeNode node = nodes.get(entity);

            return countSubordinates(node);
        }

        private int countSubordinates(TreeNode node) {
            int count = 0;

            for (TreeNode child : node.children) {
                count++;
                count += countSubordinates(child);
            }

            return count;
        }

        // IsSupervisor entity supervisor
        public boolean isSupervisor(String entity, String supervisor) {
            TreeNode node = nodes.get(entity);
            TreeNode current = node.parent;

            while (current != null) {
                if (current.name.equals(supervisor)) {
                    return true;
                }

                current = current.parent;
            }

            return false;
        }

        // IsSubordinate entity subordinate
        public boolean isSubordinate(String entity, String subordinate) {
            TreeNode node = nodes.get(entity);

            return isInSubtree(node, subordinate);
        }

        private boolean isInSubtree(TreeNode node, String name) {
            for (TreeNode child : node.children) {
                if (child.name.equals(name)) {
                    return true;
                }

                if (isInSubtree(child, name)) {
                    return true;
                }
            }

            return false;
        }

        // CompareRank
        public String compareRank(String entity1, String entity2) {
            int depth1 = getDepth(nodes.get(entity1));
            int depth2 = getDepth(nodes.get(entity2));

            if (depth1 < depth2) {
                return "higher";
            } else if (depth1 > depth2) {
                return "lower";
            } else {
                return "same";
            }
        }

        private int getDepth(TreeNode node) {
            int depth = 0;

            while (node.parent != null) {
                depth++;
                node = node.parent;
            }

            return depth;
        }

        // ClosestCommonSupervisor
        public String closestCommonSupervisor(String entity1, String entity2) {
            TreeNode node1 = nodes.get(entity1);
            TreeNode node2 = nodes.get(entity2);

            HashSet<TreeNode> supervisors = new HashSet<>();

            TreeNode current = node1.parent;

            while (current != null) {
                supervisors.add(current);
                current = current.parent;
            }

            current = node2.parent;

            while (current != null) {
                if (supervisors.contains(current)) {
                    return current.name;
                }

                current = current.parent;
            }

            return "none";
        }
    }

    public static void main(String[] args) throws IOException {

        if (args.length < 2) {
            return;
        }

        String organizationFile = args[0];
        String queryFile = args[1];

        BufferedReader orgReader =
                new BufferedReader(new FileReader(organizationFile));

        // First line is the top entity
        String rootName = orgReader.readLine();

        if (rootName == null) {
            orgReader.close();
            return;
        }

        rootName = rootName.trim();

        Tree tree = new Tree(rootName);

        String line;

        // Remaining lines contain supervisor subordinate pairs
        while ((line = orgReader.readLine()) != null) {
            line = line.trim();

            if (line.isEmpty()) {
                continue;
            }

            String[] parts = line.split("\\s+");

            if (parts.length >= 2) {
                String supervisor = parts[0];
                String subordinate = parts[1];

                tree.addChild(supervisor, subordinate);
            }
        }

        orgReader.close();

        BufferedReader queryReader =
                new BufferedReader(new FileReader(queryFile));

        StringBuilder output = new StringBuilder();

        while ((line = queryReader.readLine()) != null) {
            line = line.trim();

            if (line.isEmpty()) {
                continue;
            }

            String[] parts = line.split("\\s+");
            String query = parts[0];

            String answer = "";

            switch (query) {

                case "DirectSupervisor":
                    answer = "DirectSupervisor "
                            + parts[1] + " "
                            + tree.directSupervisor(parts[1]);
                    break;

                case "DirectSubordinates":
                    answer = "DirectSubordinates "
                            + parts[1] + " "
                            + tree.directSubordinates(parts[1]);
                    break;

                case "AllSupervisors":
                    answer = "AllSupervisors "
                            + parts[1] + " "
                            + tree.allSupervisors(parts[1]);
                    break;

                case "AllSubordinates":
                    answer = "AllSubordinates "
                            + parts[1] + " "
                            + tree.allSubordinates(parts[1]);
                    break;

                case "NumberOfAllSupervisors":
                    answer = "NumberOfAllSupervisors "
                            + parts[1] + " "
                            + tree.numberOfAllSupervisors(parts[1]);
                    break;

                case "NumberOfAllSubordinates":
                    answer = "NumberOfAllSubordinates "
                            + parts[1] + " "
                            + tree.numberOfAllSubordinates(parts[1]);
                    break;

                case "IsSupervisor":
                    answer = "IsSupervisor "
                            + parts[1] + " "
                            + parts[2] + " "
                            + (tree.isSupervisor(parts[1], parts[2])
                            ? "yes" : "no");
                    break;

                case "IsSubordinate":
                    answer = "IsSubordinate "
                            + parts[1] + " "
                            + parts[2] + " "
                            + (tree.isSubordinate(parts[1], parts[2])
                            ? "yes" : "no");
                    break;

                case "CompareRank":
                    answer = "CompareRank "
                            + parts[1] + " "
                            + parts[2] + " "
                            + tree.compareRank(parts[1], parts[2]);
                    break;

                case "ClosestCommonSupervisor":
                    answer = "ClosestCommonSupervisor "
                            + parts[1] + " "
                            + parts[2] + " "
                            + tree.closestCommonSupervisor(parts[1], parts[2]);
                    break;

                default:
                    continue;
            }

            if (output.length() > 0) {
                output.append("\n");
            }

            output.append(answer);
        }

        queryReader.close();

        System.out.println(output);
    }
}
