/**
 Author: Joshua Davis
 Email: joshuadavis2025@my.fit.edu
 Course: CSE 2010 - Algorithms & Data Structures
 Section: 4

 Description:
 This program takes 2 arguments before running. A file that takes data input for an organization
 and a second for queries on the given data, to answer questions based on relationships within the
 organization.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;


public class HW3 {
    public static void main(String[] args) throws FileNotFoundException {
         //create the two file variables here, at command line argument 0, and 1
         File data = new File(args[0]);
         File queries = new File(args[1]);

         //Declared a scanner to read the input files
         Scanner reader = new Scanner(data);

         //As given the first line of the data file is the ceo, and is the root of the tree
         String ceoName = reader.nextLine();
         TreeNode CEO = new TreeNode(ceoName);

         //then for every line after that a relationships is given of a supervisor and a subordinate
        //the handDataInput() method creates nodes on the tree based on the names until there are no more names left
         while (reader.hasNext()) {
             String supervisor = reader.next();
             String subordinate = reader.next();

             handleDataInput(supervisor, subordinate, CEO);

         }
         //Closed the reader from reading data, and make it read queries next
         reader.close();
         reader = new Scanner(queries);

         while (reader.hasNextLine()) {
             handleQuerie(reader.nextLine(), CEO);
         }
         //Helper methods to help view the tree, (Not used for Output)
         //System.out.println("Printing Tree");
         //printTree(CEO, "");
    }
    //This is the method used in the while loop above for the data input file
    static void handleDataInput(String boss, String junior, TreeNode treeRootToSearch ) {
        //First a node is made with the data being the supbordinates name, then we find the Node that we make the parent
        //of the subordinate node, since the children of a node are stored in an array list, we store that node in the
        // array list of that parent
        TreeNode subordinate = new TreeNode(junior);
        TreeNode nodeToAddChild = findAssociate(treeRootToSearch, boss);
        nodeToAddChild.addChild(subordinate);
    }
    //This was the method used above to handle all the queries with a series of if statements that identify
    //the querie itself
    static void handleQuerie(String line, TreeNode treeRootToSearch) {
        //first the querie type is parsed and idenitfied then it follows the if statements to excute the correct
        //code based on the querie
        int toSubstring = line.indexOf(" ");
        String querie = line.substring(0, toSubstring);

        //For direct supervisor
        //parse to get the data needed
        //then find the node, if the node is not null, if the parent is not null print the parent
        //if it is null, return none
        if (querie.equals("DirectSupervisor")) {
            String subordinate = line.substring(toSubstring + 1);
            TreeNode subordinateNode = findAssociate(treeRootToSearch, subordinate);

            if (subordinateNode != null) {
                if (subordinateNode.parent != null) {
                    System.out.println("DirectSupervisor " + subordinate + " " + subordinateNode.parent.data);
                } else  {
                    System.out.println("DirectSupervisor " + subordinate + " none");
                }

            }

        }
        //For Direct subordinates parse to get the data needed
        //find the associatenode that is needed
        //if that node is not null, if there are children, print them, if not return none
        else if (querie.equals("DirectSubordinates")) {
            String supervisor = line.substring(toSubstring + 1);
            TreeNode superVisorNode = findAssociate(treeRootToSearch, supervisor);
            if (superVisorNode != null) {
                if (!superVisorNode.children.isEmpty()) {
                    System.out.print("DirectSubordinates " + supervisor);

                    for (TreeNode subordinateNode : superVisorNode.children) {
                        System.out.print(" " + subordinateNode.data);
                    }
                    System.out.println();
                }  else  {
                    System.out.println("DirectSubordinates " + supervisor + " none");
                }
            }
        }
        //For all Supervisors
        //parse to get the data needed
        //find the node that is being queried
        //if that node is not null, we create a pointer of that node, then while that node has a parent print that parent
        //then set the current node to the parent
        //if there are no parents to begin with print out none
        else if (querie.equals("AllSupervisors")) {
            String associate = line.substring(toSubstring + 1);
            TreeNode associateNode = findAssociate(treeRootToSearch, associate);
            if (associateNode != null) {
                TreeNode currentNode = associateNode;
                if (currentNode.parent != null) {
                    System.out.print("AllSupervisors " + associate);
                    while (currentNode.parent != null) {
                        System.out.print(" " + currentNode.parent.data);
                        currentNode = currentNode.parent;
                    }
                    System.out.println();
                }
                else {
                    System.out.println("AllSupervisors " + associate + " none");
                }

            }
        }
        //For all subordinates
        //Parse to get the data needed
        //find the node that is being queried
        //if that node is not null, create a current node pointer and print the children, and then for those children
        //print their children and so on until there are none left, if there are no children to begin with, print none
        else if (querie.equals("AllSubordinates")) {
            String supervisor = line.substring(toSubstring + 1);
            TreeNode superVisorNode = findAssociate(treeRootToSearch, supervisor);
            if (superVisorNode != null) {
                TreeNode currentNode = superVisorNode;
                if (!currentNode.children.isEmpty()) {
                    System.out.print("AllSubordinates " + supervisor + " ");

                    for (TreeNode subordinate : currentNode.children) {
                        printSubordinates(subordinate);
                    }
                    System.out.println();
                }
                else {
                    System.out.println("AllSubordinates " + supervisor + " none");
                }
            }
        }
        //for number of all supervisors
        //parse to get data needed
        //find the node that is being queried
        //if that node is not null, create a current pointer, initialize count to be 0, and while there is a parent,
        //increment count and set current to the parent, then print the count
        else if (querie.equals("NumberOfAllSupervisors")) {
            String associate = line.substring(toSubstring + 1);
            TreeNode associateNode = findAssociate(treeRootToSearch, associate);
            if (associateNode != null) {
                TreeNode currentNode = associateNode;
                int count = 0;
                while (currentNode.parent != null) {
                    count++;
                    currentNode = currentNode.parent;
                }
                System.out.println("NumberOfAllSupervisors " + associate + " " + count);
            }
        }
        //for number of all subbordinates
        //parse to get data needed
        //find the node that is being queried
        //if that node is not null, set count to be equal to the method that counts all subordinates, and subordinates
        //of subordinates, and then print it
        else if (querie.equals("NumberOfAllSubordinates")) {
            String supervisor = line.substring(toSubstring + 1);
            TreeNode superVisorNode = findAssociate(treeRootToSearch, supervisor);
            if (superVisorNode != null) {
                int count = countSubordinates(superVisorNode);
                System.out.println("NumberOfAllSubordinates " + supervisor + " " + count);
            }
        }
        //for is supervisor
        //parse to get data needed
        //find the supervisor node, and the subordinate node
        //if both are not equal to null, set a boolean value to false for found
        //while there is a parent, search the parents, if found set found to true
        //if found is true print yes, if not print no
        else if (querie.equals("IsSupervisor")) {
            int space = line.indexOf(" ", toSubstring + 1);
            String subordinate = line.substring(toSubstring + 1, space);
            String supervisor = line.substring(space + 1);
            TreeNode subordinateNode = findAssociate(treeRootToSearch, subordinate);
            TreeNode supervisorNode = findAssociate(treeRootToSearch, supervisor);
            if (subordinateNode != null && supervisorNode != null) {
                TreeNode currentNode = subordinateNode;
                boolean found = false;

                while (currentNode.parent != null) {
                    currentNode = currentNode.parent;
                    if (currentNode == supervisorNode) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    System.out.println("IsSupervisor " + subordinate + " " + supervisor + " yes");
                }
                else {
                    System.out.println("IsSupervisor " + subordinate + " " + supervisor + " no");
                }
            }
        }
        //for is subordinate
        //parse to get data needed
        //find the supervisor node, and the subordinate node
        //if both are not equal to null, set a boolean value to false for found
        //while there are a subordinates to search, search them, if found set found to true
        //if found is true print yes, if not print no
        else if (querie.equals("IsSubordinate")) {
            int space = line.indexOf(" ", toSubstring + 1);
            String supervisor = line.substring(toSubstring + 1, space);
            String subordinate = line.substring(space + 1);
            TreeNode subordinateNode = findAssociate(treeRootToSearch, subordinate);
            TreeNode supervisorNode = findAssociate(treeRootToSearch, supervisor);
            if (subordinateNode != null && supervisorNode != null) {
                TreeNode currentNode = subordinateNode;
                boolean found = false;

                while (currentNode.parent != null) {
                    currentNode = currentNode.parent;
                    if (currentNode == supervisorNode) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    System.out.println("IsSubordinate " + supervisor + " " + subordinate + " yes");
                }
                else {
                    System.out.println("IsSubordinate " + supervisor + " " + subordinate + " no");
                }
            }
        }
        //for compare rank
        //parse to get data needed
        // find both associates and find their depth
        //if the first associate has a lower depth return higher, if it is a higher depth they are lower, if they are
        //the same, return same
        else if (querie.equals("CompareRank")) {
            int space = line.indexOf(" ", toSubstring + 1);
            String associate1 = line.substring(toSubstring + 1, space);
            String associate2 = line.substring(space + 1, line.length());

            TreeNode associate1Node = findAssociate(treeRootToSearch, associate1);
            TreeNode associate2Node = findAssociate(treeRootToSearch, associate2);

            if (associate1Node != null && associate2Node != null) {

                int associate1Depth = getDepth(associate1Node);
                int associate2Depth = getDepth(associate2Node);

                if (associate1Depth == associate2Depth) {
                    System.out.println("CompareRank " + associate1 + " " + associate2 + " same");
                }
                else if (associate1Depth < associate2Depth) {
                    System.out.println("CompareRank " + associate1 + " " + associate2 + " higher");
                }
                else {
                    System.out.println("CompareRank " + associate1 + " " + associate2 + " lower");
                }
            }
        }
        //and for closest common supervisor
        //parse to get data needed
        //find both associates
        //if both associates are not null, if their depths are not the same, move the deeper associate until it matches
        //the depth of the other, then move both of the assoiates pointers until they meet, and return that associate
        else if (querie.equals("ClosestCommonSupervisor")) {
            int space = line.indexOf(" ", toSubstring + 1);
            String associate1 = line.substring(toSubstring + 1, space);
            String associate2 = line.substring(space + 1, line.length());

            TreeNode associate1Node = findAssociate(treeRootToSearch, associate1);
            TreeNode associate2Node = findAssociate(treeRootToSearch, associate2);

            if (associate1Node != null && associate2Node != null) {
                int associate1Depth = getDepth(associate1Node);
                int associate2Depth = getDepth(associate2Node);

                while (associate1Depth != associate2Depth) {
                    if (associate1Depth > associate2Depth) {
                        associate1Node = associate1Node.parent;
                        associate1Depth--;
                    }
                    else if (associate1Depth < associate2Depth) {
                        associate2Node = associate2Node.parent;
                        associate2Depth--;
                    }
                }
                TreeNode current1Node = associate1Node;
                TreeNode current2Node = associate2Node;
                while (current1Node != current2Node) {
                    current1Node = current1Node.parent;
                    current2Node = current2Node.parent;
                }
                System.out.println("ClosestCommonSupervisor " + associate1 + " " + associate2 + " " + current1Node.data);
            }
        }
    }
    //this method searches the data tree for a node with the name we are looking for, so that the querie method can use
    //it to manipulate data
    static TreeNode findAssociate(TreeNode Organization, String associateName) {
        //edge case if  we are searching an empty tree
        if (Organization == null) {
            return null;
        }
        //the organization acts as the root so if the root is what we are looking for we catch it here
        else if (Organization.data.equals(associateName)) {
            return Organization;
        }
        //otherwise we recursively search the tree to find the node
        else {
            for (TreeNode child : Organization.children) {
                TreeNode current = findAssociate(child, associateName);
                if (current != null) {
                    return current;
                }

            }
            return null;
        }
    }
    //this wsa one of the helper methods commented out above to help visualise the tree
    static void printTree(TreeNode root, String prefix) {
        if (root == null) {
            return;
        }
        System.out.println(prefix + root.data);

        for (TreeNode child : root.children) {
            printTree(child, prefix + "  ");
        }
    }
    //this method prints all the children of a given node
    static void printSubordinates(TreeNode root) {
        if (root == null) {
            return;
        }
        System.out.print(root.data + " ");

        for (TreeNode child : root.children) {
            printSubordinates(child);
        }
    }
    //this method counts all of the children of the node
    static int countSubordinates(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int count = 0;
        for (TreeNode child : root.children) {
            count++;
            count += countSubordinates(child);
        }
        return count;
    }
    //this method gets the depth withing the tree of a node
    static int getDepth(TreeNode currentNode) {
        if (currentNode == null) {
            return 0;
        }
        int depth = 0;
        while (currentNode.parent != null) {
            depth++;
            currentNode = currentNode.parent;
        }
        return depth;
    }
}
//this is the treenode class used, the data holds the name of the node
//the children are stored in the array list, and there is a parent pointer
class TreeNode {
    String data;
    ArrayList<TreeNode> children;
    TreeNode parent;

    TreeNode(String name) {
        data = name;
        children = new ArrayList<TreeNode>();
        parent = null;
    }
    //the addchild method for the data input file
    void addChild(TreeNode toAdd) {
        toAdd.parent = this;
        for  (int i = 0; i < children.size(); i++) {
            TreeNode child = children.get(i);
            String childName = child.data;
            if (toAdd.data.compareTo(childName) < 0) {
                children.add(i, toAdd);
                return;
            }
        }
        children.add(toAdd);
    }
    //the method to get the children of a node
    void getChildren(TreeNode node) {
        if (node != null) {
            if (children != null) {
                for (TreeNode child : children) {
                    child.getChildren(node);
                }
            }
            else  {
                System.out.println(data);
            }
        }
    }
    //this method returns the parent of a Node
    TreeNode getParent(TreeNode node) {
        if (node != null) {
            return node.parent;
        }
        return null;
    }
}