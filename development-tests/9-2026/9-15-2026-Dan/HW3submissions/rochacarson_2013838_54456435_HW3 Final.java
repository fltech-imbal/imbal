
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

/*

  Author:Carson Rocha
  Email: crocha2025@my.fit.edu
  Course: CSE 2010
  Section:930 am
  Description of this file:
  This program builds an organization tree and answers queries
  about supervisors, subordinates, and ranks.

 */

class TreeNode
{
    String name;
    TreeNode parent;
    ArrayList<TreeNode> children;

    /*
      Creates one tree node with a name no parent and no kids.
    */
    public TreeNode(String name)
    {
        this.name = name;
        parent = null;
        children = new ArrayList<TreeNode>();
    }
}

class Tree
{
    TreeNode root;
    ArrayList<TreeNode> nodes;

    /*
      Creates a tree and makes rootmname the top entity.
    */
    public Tree(String rootName)
    {
        root = new TreeNode(rootName);
        nodes = new ArrayList<TreeNode>();
        nodes.add(root);
    }

    /*
      Finds a nodes by name. Returns nothing if the node do not exist.
    */
    public TreeNode findNode(String name)
    {
        for (int i = 0; i < nodes.size(); i++)
        {
            if (nodes.get(i).name.equals(name))
            {
                return nodes.get(i);
            }
        }

        return null;
    }

    /*
      Returns an existing node, or creates it if it is not in the tree yet.
    */
    public TreeNode getOrCreateNode(String name)
    {
        TreeNode node = findNode(name);

        if (node == null)
        {
            node = new TreeNode(name);
            nodes.add(node);
        }

        return node;
    }

    /*
      Adds kidNode as a kid of node. kids are kept in
      alphabetical/lexicographical order.
    */
    public void addChild(TreeNode node, TreeNode childNode)
    {
        int position = 0;

        while (position < node.children.size()
                && node.children.get(position).name.compareTo(childNode.name) < 0)
        {
            position++;
        }

        node.children.add(position, childNode);
        childNode.parent = node;
    }

    /*
      Returns the direct kid of a node.
    */
    public ArrayList<TreeNode> getChildren(TreeNode node)
    {
        return node.children;
    }

    /*
      Returns the direct parent of node, or null if node is the root.
    */
    public TreeNode getParent(TreeNode node)
    {
        return node.parent;
    }
}

public class HW3
{
    /*
      Adds all kids of node to list in preorder.
    */
    public static void getAllSubordinates(TreeNode node, ArrayList<TreeNode> list)
    {
        for (int i = 0; i < node.children.size(); i++)
        {
            TreeNode child = node.children.get(i);
            list.add(child);
            getAllSubordinates(child, list);
        }
    }

    /*
      Returns true if possibleSupervisor is any supervisor of entity.
    */
    public static boolean isSupervisor(TreeNode entity, TreeNode possibleSupervisor)
    {
        TreeNode current = entity.parent;

        while (current != null)
        {
            if (current == possibleSupervisor)
            {
                return true;
            }

            current = current.parent;
        }

        return false;
    }

    /*
      Returns the number of levels from the root to node.
      The root has depth 0.
    */
    public static int getDepth(TreeNode node)
    {
        int depth = 0;
        TreeNode current = node;

        while (current.parent != null)
        {
            depth++;
            current = current.parent;
        }

        return depth;
    }

    /*
      Finds the closest node that is a supervisor of both entities.
      Returns null if they have no common supervisor.
    */
    public static TreeNode closestCommonSupervisor(TreeNode entity1, TreeNode entity2)
    {
        ArrayList<TreeNode> supervisors1 = new ArrayList<TreeNode>();
        TreeNode current = entity1.parent;

        while (current != null)
        {
            supervisors1.add(current);
            current = current.parent;
        }

        current = entity2.parent;

        while (current != null)
        {
            for (int i = 0; i < supervisors1.size(); i++)
            {
                if (supervisors1.get(i) == current)
                {
                    return current;
                }
            }

            current = current.parent;
        }

        return null;
    }

    /*
      Reads the organization data file and builds the tree.
    */
    public static Tree buildTree(String filename) throws FileNotFoundException
    {
        Scanner input = new Scanner(new File(filename));

        String rootName = input.nextLine().trim();
        Tree tree = new Tree(rootName);

        while (input.hasNextLine())
        {
            String line = input.nextLine().trim();

            if (!line.equals(""))
            {
                Scanner lineScanner = new Scanner(line);
                String supervisorName = lineScanner.next();
                String subordinateName = lineScanner.next();

                TreeNode supervisor = tree.getOrCreateNode(supervisorName);
                TreeNode subordinate = tree.getOrCreateNode(subordinateName);

                tree.addChild(supervisor, subordinate);
                lineScanner.close();
            }
        }

        input.close();
        return tree;
    }

    /*
      Reads each query, finds the needed nodes, and prints the answer.
    */
    public static void answerQueries(Tree tree, String filename) throws FileNotFoundException
    {
        Scanner input = new Scanner(new File(filename));

        while (input.hasNextLine())
        {
            String line = input.nextLine().trim();

            if (line.equals(""))
            {
                continue;
            }

            Scanner query = new Scanner(line);
            String type = query.next();

            if (type.equals("DirectSupervisor"))
            {
                String entityName = query.next();
                TreeNode entity = tree.findNode(entityName);
                TreeNode parent = tree.getParent(entity);

                System.out.print(type + " " + entityName + " ");

                if (parent == null)
                {
                    System.out.println("none");
                }
                else
                {
                    System.out.println(parent.name);
                }
            }
            else if (type.equals("DirectSubordinates"))
            {
                String entityName = query.next();
                TreeNode entity = tree.findNode(entityName);
                ArrayList<TreeNode> children = tree.getChildren(entity);

                System.out.print(type + " " + entityName);

                if (children.size() == 0)
                {
                    System.out.println(" none");
                }
                else
                {
                    for (int i = 0; i < children.size(); i++)
                    {
                        System.out.print(" " + children.get(i).name);
                    }
                    System.out.println();
                }
            }
            else if (type.equals("AllSupervisors"))
            {
                String entityName = query.next();
                TreeNode entity = tree.findNode(entityName);
                TreeNode current = entity.parent;

                System.out.print(type + " " + entityName);

                if (current == null)
                {
                    System.out.println(" none");
                }
                else
                {
                    while (current != null)
                    {
                        System.out.print(" " + current.name);
                        current = current.parent;
                    }
                    System.out.println();
                }
            }
            else if (type.equals("AllSubordinates"))
            {
                String entityName = query.next();
                TreeNode entity = tree.findNode(entityName);
                ArrayList<TreeNode> subordinates = new ArrayList<TreeNode>();

                getAllSubordinates(entity, subordinates);
                System.out.print(type + " " + entityName);

                if (subordinates.size() == 0)
                {
                    System.out.println(" none");
                }
                else
                {
                    for (int i = 0; i < subordinates.size(); i++)
                    {
                        System.out.print(" " + subordinates.get(i).name);
                    }
                    System.out.println();
                }
            }
            else if (type.equals("NumberOfAllSupervisors"))
            {
                String entityName = query.next();
                TreeNode entity = tree.findNode(entityName);
                int count = 0;
                TreeNode current = entity.parent;

                while (current != null)
                {
                    count++;
                    current = current.parent;
                }

                System.out.println(type + " " + entityName + " " + count);
            }
            else if (type.equals("NumberOfAllSubordinates"))
            {
                String entityName = query.next();
                TreeNode entity = tree.findNode(entityName);
                ArrayList<TreeNode> subordinates = new ArrayList<TreeNode>();

                getAllSubordinates(entity, subordinates);
                System.out.println(type + " " + entityName + " " + subordinates.size());
            }
            else if (type.equals("IsSupervisor"))
            {
                String entityName = query.next();
                String supervisorName = query.next();
                TreeNode entity = tree.findNode(entityName);
                TreeNode supervisor = tree.findNode(supervisorName);

                System.out.print(type + " " + entityName + " " + supervisorName + " ");

                if (isSupervisor(entity, supervisor))
                {
                    System.out.println("yes");
                }
                else
                {
                    System.out.println("no");
                }
            }
            else if (type.equals("IsSubordinate"))
            {
                String entityName = query.next();
                String subordinateName = query.next();
                TreeNode entity = tree.findNode(entityName);
                TreeNode subordinate = tree.findNode(subordinateName);

                System.out.print(type + " " + entityName + " " + subordinateName + " ");

                if (isSupervisor(subordinate, entity))
                {
                    System.out.println("yes");
                }
                else
                {
                    System.out.println("no");
                }
            }
            else if (type.equals("CompareRank"))
            {
                String entity1Name = query.next();
                String entity2Name = query.next();
                TreeNode entity1 = tree.findNode(entity1Name);
                TreeNode entity2 = tree.findNode(entity2Name);
                int depth1 = getDepth(entity1);
                int depth2 = getDepth(entity2);

                System.out.print(type + " " + entity1Name + " " + entity2Name + " ");

                if (depth1 < depth2)
                {
                    System.out.println("higher");
                }
                else if (depth1 > depth2)
                {
                    System.out.println("lower");
                }
                else
                {
                    System.out.println("same");
                }
            }
            else if (type.equals("ClosestCommonSupervisor"))
            {
                String entity1Name = query.next();
                String entity2Name = query.next();
                TreeNode entity1 = tree.findNode(entity1Name);
                TreeNode entity2 = tree.findNode(entity2Name);
                TreeNode common = closestCommonSupervisor(entity1, entity2);

                System.out.print(type + " " + entity1Name + " " + entity2Name + " ");

                if (common == null)
                {
                    System.out.println("none");
                }
                else
                {
                    System.out.println(common.name);
                }
            }

            query.close();
        }

        input.close();
    }

    /*
      Main receives the organizational data filename and query filename
      from the command line, builds the tree, and answers all queries.
    */
    public static void main(String[] args) throws FileNotFoundException
    {
        /* The first argument is the organization data file.
           The second argument is the query file. */
        String dataFilename = args[0];
        String queryFilename = args[1];

        /* Build the organization tree from the data file.
           Each child is inserted in alphabetical order. */
        Tree tree = buildTree(dataFilename);

        /* Read the query file and print one answer for each query. */
        answerQueries(tree, queryFilename);
    }
}
