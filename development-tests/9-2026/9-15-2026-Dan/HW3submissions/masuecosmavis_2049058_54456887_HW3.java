/*
  Clarence Masuecos
  mmasuecos2025@my.fit.edu
  CSE 2010
  Section 3
  description: Reads organizational ranking text file to build 
  custom linked Tree data structure. 
  Then processes multiple queries to find 
  subordinates, rank comparisons, and common supervisors.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.HashMap;

public class HW3 {

    /*
      description of TreeNode class:
      simple linked structure object 
      that holds employee's name, reference 
      to their supervisor, and 
      ArrayList of their direct children.
    */
    static class TreeNode {
        String name;
        TreeNode parent;
        ArrayList childrenList;

        public TreeNode(String name) {
            this.name = name;
            this.parent = null;
            this.childrenList = new ArrayList<>();
        }
    }

    /*
      description Tree class
      main tree class requested 
      uses HashMap for O(1) 
      node lookups 
      while maintaining linked hierarchy
    */
    static class Tree {
        TreeNode root;
        HashMap lookupMap;

        public Tree() {
            lookupMap = new HashMap<>();
        }

        /* 
          description addChild method
          links child node to parent node 
          and maintains alphabetical order 
          of the children list 
          using insertion logic
        */
        public void addChild(TreeNode node, TreeNode childNode) {
            childNode.parent = node;
            
            int i = 0;
            while (i < node.childrenList.size()) {
                TreeNode currentChild = (TreeNode) node.childrenList.get(i);
                if (childNode.name.compareTo(currentChild.name) < 0) {
                    break;
                }
                i++;
            }
            node.childrenList.add(i, childNode);
        }

        /* 
          description getChildren method:
          returns ArrayList of direct subordinates
        */
        public ArrayList getChildren(TreeNode node) {
            return node.childrenList;
        }

        /* descripton getParent method
          returns direct supervisor node
        */
        public TreeNode getParent(TreeNode node) {
            return node.parent;
        }
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Please provide data and query files.");
            return;
        }

        Tree orgTree = new Tree();

        /* 
           read the organizational data file and build tree 
        */
        try {
            Scanner dataScan = new Scanner(new File(args[0]));
            
            if (dataScan.hasNextLine()) {
                String rootName = dataScan.nextLine().trim();
                orgTree.root = new TreeNode(rootName);
                orgTree.lookupMap.put(rootName, orgTree.root);
            }

            // Loop through rest of lines to link supervisors and subordinates
            while (dataScan.hasNextLine()) {
                String line = dataScan.nextLine().trim();
                if (line.isEmpty()) continue;
                
                String[] parts = line.split(" ");
                String supervisorName = parts[0];
                String subordinateName = parts[1];

                // creaTE supervisor node
                TreeNode supNode = (HW3.TreeNode) orgTree.lookupMap.get(supervisorName);
                if (supNode == null) {
                    supNode = new TreeNode(supervisorName);
                    orgTree.lookupMap.put(supervisorName, supNode);
                }
                
                // create subordinate node
                TreeNode subNode = (HW3.TreeNode) orgTree.lookupMap.get(subordinateName);
                if (subNode == null) {
                    subNode = new TreeNode(subordinateName);
                    orgTree.lookupMap.put(subordinateName, subNode);
                }

                orgTree.addChild(supNode, subNode);
            }
            dataScan.close();
        } catch (FileNotFoundException e) {
            System.out.println("Data file not found.");
            return;
        }

        /* 
           read and execute all queries from second file 
        */
        try {
            Scanner queryScan = new Scanner(new File(args[1]));
            while (queryScan.hasNextLine()) {
                String line = queryScan.nextLine().trim();
                if (line.isEmpty()) continue;
                
                String[] qParts = line.split(" ");
                String command = qParts[0];
                String entity = qParts[1];
                TreeNode target = (HW3.TreeNode) orgTree.lookupMap.get(entity);

                // Print base part of query output first
                System.out.print(command + " " + entity);

                if (command.equals("DirectSupervisor")) {
                    TreeNode p = orgTree.getParent(target);
                    if (p != null) {
                        System.out.println(" " + p.name);
                    } else {
                        System.out.println(" none");
                    }
                } 
                else if (command.equals("DirectSubordinates")) {
                    ArrayList kids = orgTree.getChildren(target);
                    if (kids.isEmpty()) {
                        System.out.println(" none");
                    } else {
                        for (Object o : kids) {
                            TreeNode k = (TreeNode) o;
                            System.out.print(" " + k.name);
                        }
                        System.out.println();
                    }
                }
                else if (command.equals("AllSupervisors")) {
                    TreeNode curr = orgTree.getParent(target);
                    if (curr == null) {
                        System.out.println(" none");
                    } else {
                        // loop upwards to printing each parent until root
                        while (curr != null) {
                            System.out.print(" " + curr.name);
                            curr = orgTree.getParent(curr);
                        }
                        System.out.println();
                    }
                }
                else if (command.equals("AllSubordinates")) {
                    ArrayList subList = new ArrayList<>();
                    getPreOrder(target, subList);
                    if (subList.isEmpty()) {
                        System.out.println(" none");
                    } else {
                        for (Object o : subList) {
                            String s = (String) o;
                            System.out.print(" " + s);
                        }
                        System.out.println();
                    }
                }
                else if (command.equals("NumberOfAllSupervisors")) {
                    int count = 0;
                    TreeNode curr = orgTree.getParent(target);
                    while (curr != null) {
                        count++;
                        curr = orgTree.getParent(curr);
                    }
                    System.out.println(" " + count);
                }
                else if (command.equals("NumberOfAllSubordinates")) {
                    ArrayList subList = new ArrayList<>();
                    getPreOrder(target, subList);
                    System.out.println(" " + subList.size());
                }
                else if (command.equals("IsSupervisor")) {
                    String supposedSup = qParts[2];
                    System.out.print(" " + supposedSup);
                    
                    boolean found = false;
                    TreeNode curr = orgTree.getParent(target);
                    // traverse back up to see if found supervisor is ancestor
                    while (curr != null) {
                        if (curr.name.equals(supposedSup)) {
                            found = true;
                            break;
                        }
                        curr = orgTree.getParent(curr);
                    }
                    
                    if (found) System.out.println(" yes");
                    else System.out.println(" no");
                }
                else if (command.equals("IsSubordinate")) {
                    String supposedSub = qParts[2];
                    System.out.print(" " + supposedSub);
                    
                    ArrayList subList = new ArrayList<>();
                    getPreOrder(target, subList);
                
                    // if pre-order traversal found them, confirmed a subordinate
                    if (subList.contains(supposedSub)) {
                        System.out.println(" yes");
                    } else {
                        System.out.println(" no");
                    }
                }
                else if (command.equals("CompareRank")) {
                    String entity2 = qParts[2];
                    System.out.print(" " + entity2);
                    TreeNode target2 = (HW3.TreeNode) orgTree.lookupMap.get(entity2);
                    
                    int depth1 = getDepth(orgTree, target);
                    int depth2 = getDepth(orgTree, target2);
                    
                    // smaller depth means closer to root
                    if (depth1 < depth2) {
                        System.out.println(" higher");
                    } else if (depth1 > depth2) {
                        System.out.println(" lower");
                    } else {
                        System.out.println(" same");
                    }
                }
                else if (command.equals("ClosestCommonSupervisor")) {
                    String entity2 = qParts[2];
                    System.out.print(" " + entity2);
                    TreeNode target2 = (HW3.TreeNode) orgTree.lookupMap.get(entity2);
                    
                    // collect all ancestors first entity
                    ArrayList ancestors1 = new ArrayList<>();
                    TreeNode curr1 = orgTree.getParent(target);
                    while (curr1 != null) {
                        ancestors1.add(curr1.name);
                        curr1 = orgTree.getParent(curr1);
                    }
                    
                    // traverse up from second entity until match
                    TreeNode curr2 = orgTree.getParent(target2);
                    String lca = "none";
                    while (curr2 != null) {
                        if (ancestors1.contains(curr2.name)) {
                            lca = curr2.name;
                            break;
                        }
                        curr2 = orgTree.getParent(curr2);
                    }
                    
                    System.out.println(" " + lca);
                }
            }
            queryScan.close();
        } catch (FileNotFoundException e) {
            System.out.println("Query file not found.");
        }
    }

    /*
      description getPreOrder method:
     recursive helper function to gather 
     all subordinates for given node
      loops through direct children and recursively calls itself to match 
    pre-order traversal requirement
    */
    public static void getPreOrder(TreeNode node, ArrayList list) {
        for (int i = 0; i < node.childrenList.size(); i++) {
            TreeNode kid = (HW3.TreeNode) node.childrenList.get(i);
            list.add(kid.name);
            getPreOrder(kid, list); // dive deeper
        }
    }

    /*
      description getDepth method:
      helper function calculates 
      how far node is from root
      used for compare ranks between two employees
    */
    public static int getDepth(Tree tree, TreeNode node) {
        int depth = 0;
        TreeNode curr = tree.getParent(node);
        while (curr != null) {
            depth++;
            curr = tree.getParent(curr);
        }
        return depth;
    }
}