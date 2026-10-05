/*

  Author: Alexander Williams
  Email: alexanderwil2025@my.fit.edu
  Course: CSE2010
  Section: 1
  Description of this file: HW3

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.HashMap;
import java.util.Scanner;

class TreeNode
{
    String name;
    TreeNode parent;
    TreeNode firstChild;
    TreeNode nextSibling;

    TreeNode(String name)
    {
        this.name = name;
        parent = null;
        firstChild = null;
        nextSibling = null;
    }
}

class Tree
{
    TreeNode root;

    Tree(String name)
    {
        root = new TreeNode(name);
    }

    TreeNode getParent(TreeNode node)
    {
        return node.parent;
    }

    TreeNode getChildren(TreeNode node)
    {
        return node.firstChild;
    }

    void addChild(TreeNode node, TreeNode childNode)
    {
        childNode.parent = node;

        if (node.firstChild == null)
        {
            node.firstChild = childNode;
            return;
        }

        if (childNode.name.compareTo(node.firstChild.name) < 0)
        {
            childNode.nextSibling = node.firstChild;
            node.firstChild = childNode;
            return;
        }

        TreeNode current = node.firstChild;

        while (current.nextSibling != null &&
               current.nextSibling.name.compareTo(childNode.name) < 0)
        {
            current = current.nextSibling;
        }

        childNode.nextSibling = current.nextSibling;
        current.nextSibling = childNode;
    }

    TreeNode findNode(String name)
    {
        return findNodeRecursive(root, name);
    }

    TreeNode findNodeRecursive(TreeNode node, String name)
    {
        if (node == null)
        {
            return null;
        }

        if (node.name.equals(name))
        {
            return node;
        }

        TreeNode current = node.firstChild;

        while (current != null)
        {
            TreeNode result = findNodeRecursive(current, name);

            if (result != null)
            {
                return result;
            }

            current = current.nextSibling;
        }

        return null;
    }

    void printAllSubordinates(TreeNode node)
    {
        TreeNode child = node.firstChild;

        while (child != null)
        {
            System.out.print(" " + child.name);
            printAllSubordinates(child);
            child = child.nextSibling;
        }
    }

    int numberOfAllSubordinates(TreeNode node)
    {
        int count = 0;
        TreeNode child = node.firstChild;

        while (child != null)
        {
            count = count + 1 + numberOfAllSubordinates(child);
            child = child.nextSibling;
        }

        return count;
    }

    int numberOfAllSupervisors(TreeNode node)
    {
        int count = 0;
        TreeNode current = node.parent;

        while (current != null)
        {
            count++;
            current = current.parent;
        }

        return count;
    }

    boolean isSupervisor(TreeNode entity, TreeNode supervisor)
    {
        TreeNode current = entity.parent;

        while (current != null)
        {
            if (current == supervisor)
            {
                return true;
            }

            current = current.parent;
        }

        return false;
    }

    int getRank(TreeNode node)
    {
        int rank = 0;
        TreeNode current = node.parent;

        while (current != null)
        {
            rank++;
            current = current.parent;
        }

        return rank;
    }

    TreeNode closestCommonSupervisor(TreeNode first, TreeNode second)
    {
        TreeNode secondSupervisor = second.parent;

        while (secondSupervisor != null)
        {
            TreeNode firstSupervisor = first.parent;

            while (firstSupervisor != null)
            {
                if (firstSupervisor == secondSupervisor)
                {
                    return secondSupervisor;
                }

                firstSupervisor = firstSupervisor.parent;
            }

            secondSupervisor = secondSupervisor.parent;
        }

        return null;
    }
}

public class HW3
{
    
     //main method reads the organization file, builds the tree, then reads the query file and answers each query.     
     //Parameters:args - command-line arguments containing the organization filename and query filename

    public static void main(String[] args)
    {
        //variables used for the input files, tree, and scanners. 
        String dataFileName = args[0];
        String queryFileName = args[1];
        Tree tree = null;
        Scanner dataScanner = null;
        Scanner queryScanner = null;
        HashMap<String, TreeNode> nodes = new HashMap<String, TreeNode>();

        
         //opens the organization file
        try
        {
            dataScanner = new Scanner(new File(dataFileName));
            String rootName = dataScanner.nextLine();
            tree = new Tree(rootName);
            nodes.put(rootName, tree.root);

            while (dataScanner.hasNextLine())
            {
                String line = dataScanner.nextLine();
                if (line.trim().isEmpty())
                {
                 continue;
                }
                Scanner lineScanner = new Scanner(line);

                String supervisorName = lineScanner.next();
                String subordinateName = lineScanner.next();

                TreeNode supervisor = nodes.get(supervisorName);
                TreeNode subordinate = nodes.get(subordinateName);

                if (supervisor == null)
                {
                    supervisor = new TreeNode(supervisorName);
                    nodes.put(supervisorName, supervisor);
                }

                if (subordinate == null)
                {
                    subordinate = new TreeNode(subordinateName);
                    nodes.put(subordinateName, subordinate);
                }

                tree.addChild(supervisor, subordinate);
                lineScanner.close();
            }

            dataScanner.close();
        }
        catch (FileNotFoundException e)
        {
            return;
        }

        
         //opens the query file
        try
        {
            queryScanner = new Scanner(new File(queryFileName));

            while (queryScanner.hasNextLine())
            {
                String line = queryScanner.nextLine();
                if (line.trim().isEmpty())
                {
                 continue; 
                }
                Scanner lineScanner = new Scanner(line);

                String query = lineScanner.next();

                if (query.equals("DirectSupervisor"))
                {
                    String entityName = lineScanner.next();
                    TreeNode entity = nodes.get(entityName);
                    TreeNode supervisor = tree.getParent(entity);

                    if (supervisor == null)
                    {
                        System.out.println("DirectSupervisor " + entityName + " none");
                    }
                    else
                    {
                        System.out.println("DirectSupervisor " + entityName + " " + supervisor.name);
                    }
                }
                else if (query.equals("DirectSubordinates"))
                {
                    String entityName = lineScanner.next();
                    TreeNode entity = nodes.get(entityName);
                    TreeNode child = tree.getChildren(entity);

                    System.out.print("DirectSubordinates " + entityName);

                    if (child == null)
                    {
                        System.out.println(" none");
                    }
                    else
                    {
                        while (child != null)
                        {
                            System.out.print(" " + child.name);
                            child = child.nextSibling;
                        }

                        System.out.println();
                    }
                }
                else if (query.equals("AllSupervisors"))
                {
                    String entityName = lineScanner.next();
                    TreeNode entity = nodes.get(entityName);
                    TreeNode current = tree.getParent(entity);

                    System.out.print("AllSupervisors " + entityName);

                    if (current == null)
                    {
                        System.out.println(" none");
                    }
                    else
                    {
                        while (current != null)
                        {
                            System.out.print(" " + current.name);
                            current = tree.getParent(current);
                        }

                        System.out.println();
                    }
                }
                else if (query.equals("AllSubordinates"))
                {
                    String entityName = lineScanner.next();
                    TreeNode entity = nodes.get(entityName);

                    System.out.print("AllSubordinates " + entityName);

                    if (tree.getChildren(entity) == null)
                    {
                        System.out.println(" none");
                    }
                    else
                    {
                        tree.printAllSubordinates(entity);
                        System.out.println();
                    }
                }
                else if (query.equals("NumberOfAllSupervisors"))
                {
                    String entityName = lineScanner.next();
                    TreeNode entity = nodes.get(entityName);
                    int count = tree.numberOfAllSupervisors(entity);

                    System.out.println("NumberOfAllSupervisors " + entityName + " " + count);
                }
                else if (query.equals("NumberOfAllSubordinates"))
                {
                    String entityName = lineScanner.next();
                    TreeNode entity = nodes.get(entityName);
                    int count = tree.numberOfAllSubordinates(entity);

                    System.out.println("NumberOfAllSubordinates " + entityName + " " + count);
                }
                else if (query.equals("IsSupervisor"))
                {
                    String entityName = lineScanner.next();
                    String supervisorName = lineScanner.next();

                    TreeNode entity = nodes.get(entityName);
                    TreeNode supervisor = nodes.get(supervisorName);

                    if (tree.isSupervisor(entity, supervisor))
                    {
                        System.out.println("IsSupervisor " + entityName + " " + supervisorName + " yes");
                    }
                    else
                    {
                        System.out.println("IsSupervisor " + entityName + " " + supervisorName + " no");
                    }
                }
                else if (query.equals("IsSubordinate"))
                {
                    String entityName = lineScanner.next();
                    String subordinateName = lineScanner.next();

                    TreeNode entity = nodes.get(entityName);
                    TreeNode subordinate = nodes.get(subordinateName);

                    if (tree.isSupervisor(subordinate, entity))
                    {
                        System.out.println("IsSubordinate " + entityName + " " + subordinateName + " yes");
                    }
                    else
                    {
                        System.out.println("IsSubordinate " + entityName + " " + subordinateName + " no");
                    }
                }
                else if (query.equals("CompareRank"))
                {
                    String entity1Name = lineScanner.next();
                    String entity2Name = lineScanner.next();

                    TreeNode entity1 = nodes.get(entity1Name);
                    TreeNode entity2 = nodes.get(entity2Name);

                    int rank1 = tree.getRank(entity1);
                    int rank2 = tree.getRank(entity2);

                    if (rank1 < rank2)
                    {
                        System.out.println("CompareRank " + entity1Name + " " + entity2Name + " higher");
                    }
                    else if (rank1 > rank2)
                    {
                        System.out.println("CompareRank " + entity1Name + " " + entity2Name + " lower");
                    }
                    else
                    {
                        System.out.println("CompareRank " + entity1Name + " " + entity2Name + " same");
                    }
                }
                else if (query.equals("ClosestCommonSupervisor"))
                {
                    String entity1Name = lineScanner.next();
                    String entity2Name = lineScanner.next();

                    TreeNode entity1 = nodes.get(entity1Name);
                    TreeNode entity2 = nodes.get(entity2Name);
                    TreeNode common =
                        tree.closestCommonSupervisor(entity1, entity2);

                    if (common == null)
                    {
                        System.out.println("ClosestCommonSupervisor " + entity1Name + " " + entity2Name + " none");
                    }
                    else
                    {
                        System.out.println("ClosestCommonSupervisor " + entity1Name + " " + entity2Name + " " + common.name);
                    }
                }

                lineScanner.close();
            }

            queryScanner.close();
        }
        catch (FileNotFoundException e)
        {
            return;
        }
    }
}

