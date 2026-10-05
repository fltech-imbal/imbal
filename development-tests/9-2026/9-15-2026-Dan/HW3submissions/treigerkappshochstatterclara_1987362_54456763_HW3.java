/*

  Author: Clara Treiger
  Email: mtreigerkapp2024@my.fit.edu
  Course: CSE2010
  Section: 1
  Description of this file: This program builds an organizational tree from a data file and
  processes queries about the relationships between people in the
  organization. It supports finding supervisors and subordinates,
  comparing ranks, counting supervisors and subordinates, and finding
  the closest common supervisor between two people.

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;


/*
 * Node represents one person in the organization tree.
 *
 * name stores the person's name.
 * parent points to the person's direct supervisor.
 * firstChild points to the person's first direct subordinate.
 * nextSibling points to the person's next subordinate.
 */
class Node
{
    String name;
    Node parent;
    Node firstChild;
    Node nextSibling;

    /*
     * Description:
     * Constructs a new Node with the given name.
     *
     * Parameters:
     * name - the name of the person represented by this node.
     */
    Node(String name)
    {
        this.name = name;
        parent = null;
        firstChild = null;
        nextSibling = null;
    }
}


/*
 * Tree represents the organizational hierarchy.
 *
 * The tree stores the root of the organization and provides
 * methods for finding nodes, adding children, finding supervisors,
 * finding subordinates, comparing ranks, and finding common supervisors.
 */
class Tree
{
    Node root;

    /*
     * Description:
     * Constructs a new organizational tree.
     *
     * Parameters:
     * rootName - the name of the person at the top of the organization.
     */
    Tree(String rootName)
    {
        root = new Node(rootName);
    }

    /*
     * Description:
     * Finds a node in the organizational tree using its name.
     *
     * Parameters:
     * name - the name of the person to find.
     *
     * Returns:
     * The Node containing the given name, or null if it is not found.
     */
    Node find(String name)
    {
        return find(root, name);
    }

    /*
     * Description:
     * Recursively searches the tree for a node with the given name.
     *
     * Parameters:
     * current - the node where the search currently begins.
     * name - the name of the person to find.
     *
     * Returns:
     * The Node containing the given name, or null if it is not found.
     */
    Node find(Node current, String name)
    {
        if (current == null)
        {
            return null;
        }

        if (current.name.equals(name))
        {
            return current;
        }

        Node child = current.firstChild;

        while (child != null)
        {
            Node result = find(child, name);

            if (result != null)
            {
                return result;
            }

            child = child.nextSibling;
        }

        return null;
    }

    /*
     * Description:
     * Finds the direct supervisor of a person.
     *
     * Parameters:
     * name - the name of the person whose supervisor is needed.
     *
     * Returns:
     * The name of the direct supervisor, or "none" if the person
     * is the root of the organization.
     */
    String getDirectSupervisor(String name)
    {
        Node node = find(name);

        if (node.parent == null)
        {
            return "none";
        }

        return node.parent.name;
    }

    /*
     * Description:
     * Prints all direct subordinates of a person in alphabetical order.
     *
     * Parameters:
     * name - the name of the person whose direct subordinates are needed.
     */
    void printDirectSubordinates(String name)
    {
        Node node = find(name);

        System.out.print("DirectSubordinates " + name);

        Node child = node.firstChild;

        if (child == null)
        {
            System.out.print(" none");
        }
        else
        {
            while (child != null)
            {
                System.out.print(" " + child.name);
                child = child.nextSibling;
            }
        }

        System.out.println();
    }

    /*
     * Description:
     * Determines whether one person is a supervisor of another person.
     *
     * Parameters:
     * entity - the person whose supervisors are being checked.
     * supervisor - the person being checked as a supervisor.
     *
     * Returns:
     * true if supervisor is an ancestor of entity, otherwise false.
     */
    boolean isSupervisor(String entity, String supervisor)
    {
        Node node = find(entity);

        Node current = node.parent;

        while (current != null)
        {
            if (current.name.equals(supervisor))
            {
                return true;
            }

            current = current.parent;
        }

        return false;
    }

    /*
     * Description:
     * Determines whether one person is a subordinate of another person.
     *
     * Parameters:
     * entity - the person whose subordinate relationship is checked.
     * subordinate - the person being checked as a subordinate.
     *
     * Returns:
     * true if subordinate is below entity in the tree, otherwise false.
     */
    boolean isSubordinate(String entity, String subordinate)
    {
        Node node = find(subordinate);

        Node current = node.parent;

        while (current != null)
        {
            if (current.name.equals(entity))
            {
                return true;
            }

            current = current.parent;
        }

        return false;
    }

    /*
     * Description:
     * Prints every supervisor of a person, starting with the direct
     * supervisor and continuing upward toward the root.
     *
     * Parameters:
     * name - the name of the person whose supervisors are needed.
     */
    void printAllSupervisors(String name)
    {
        Node node = find(name);

        System.out.print("AllSupervisors " + name);

        Node current = node.parent;

        if (current == null)
        {
            System.out.print(" none");
        }
        else
        {
            while (current != null)
            {
                System.out.print(" " + current.name);
                current = current.parent;
            }
        }

        System.out.println();
    }

    /*
     * Description:
     * Prints every subordinate of a person using preorder traversal.
     *
     * Parameters:
     * name - the name of the person whose subordinates are needed.
     */
    void printAllSubordinates(String name)
    {
        Node node = find(name);

        System.out.print("AllSubordinates " + name);

        if (node.firstChild == null)
        {
            System.out.print(" none");
        }
        else
        {
            printSubordinates(node.firstChild);
        }

        System.out.println();
    }

    /*
     * Description:
     * Recursively prints a node and all of its subordinates.
     *
     * Parameters:
     * current - the node from which the traversal begins.
     */
    void printSubordinates(Node current)
    {
        while (current != null)
        {
            System.out.print(" " + current.name);

            if (current.firstChild != null)
            {
                printSubordinates(current.firstChild);
            }

            current = current.nextSibling;
        }
    }

    /*
     * Description:
     * Counts the number of supervisors above a person.
     *
     * Parameters:
     * name - the name of the person being checked.
     *
     * Returns:
     * The number of supervisors above the given person.
     */
    int numberOfAllSupervisors(String name)
    {
        Node node = find(name);

        int count = 0;
        Node current = node.parent;

        while (current != null)
        {
            count++;
            current = current.parent;
        }

        return count;
    }

    /*
     * Description:
     * Counts all descendants of a person in the organizational tree.
     *
     * Parameters:
     * name - the name of the person whose descendants are counted.
     *
     * Returns:
     * The total number of direct and indirect subordinates.
     */
    int numberOfAllSubordinates(String name)
    {
        Node node = find(name);

        return countSubordinates(node);
    }

    /*
     * Description:
     * Recursively counts all descendants of a node.
     *
     * Parameters:
     * current - the node whose descendants are being counted.
     *
     * Returns:
     * The total number of descendants below the current node.
     */
    int countSubordinates(Node current)
    {
        int count = 0;

        Node child = current.firstChild;

        while (child != null)
        {
            count++;

            if (child.firstChild != null)
            {
                count += countSubordinates(child);
            }

            child = child.nextSibling;
        }

        return count;
    }

    /*
     * Description:
     * Compares the organizational rank of two people.
     *
     * Parameters:
     * name1 - the name of the first person.
     * name2 - the name of the second person.
     *
     * Returns:
     * "higher" if the first person is higher in the organization,
     * "lower" if the first person is lower, or "same" if they have
     * the same number of supervisors.
     */
    String compareRank(String name1, String name2)
    {
        int rank1 = numberOfAllSupervisors(name1);
        int rank2 = numberOfAllSupervisors(name2);

        if (rank1 < rank2)
        {
            return "higher";
        }
        else if (rank1 > rank2)
        {
            return "lower";
        }
        else
        {
            return "same";
        }
    }

    /*
     * Description:
     * Finds the closest common supervisor of two people.
     *
     * Parameters:
     * name1 - the name of the first person.
     * name2 - the name of the second person.
     *
     * Returns:
     * The name of the closest common supervisor, or "none" if there
     * is no common supervisor.
     */
    String closestCommonSupervisor(String name1, String name2)
    {
        Node node1 = find(name1);
        Node node2 = find(name2);

        Node current1 = node1;

        while (current1 != null)
        {
            Node current2 = node2;

            while (current2 != null)
            {
                if (current1 == current2)
                {
                    return current1.name;
                }

                current2 = current2.parent;
            }

            current1 = current1.parent;
        }

        return "none";
    }

    /*
     * Description:
     * Adds a child node to a parent node while keeping the children
     * in alphabetical order.
     *
     * Parameters:
     * parent - the node that will become the child's supervisor.
     * child - the node that will be added as a subordinate.
     */
    void addChild(Node parent, Node child)
    {
        child.parent = parent;

        if (parent.firstChild == null)
        {
            parent.firstChild = child;
            return;
        }

        if (child.name.compareTo(parent.firstChild.name) < 0)
        {
            child.nextSibling = parent.firstChild;
            parent.firstChild = child;
            return;
        }

        Node current = parent.firstChild;

        while (current.nextSibling != null &&
               current.nextSibling.name.compareTo(child.name) < 0)
        {
            current = current.nextSibling;
        }

        child.nextSibling = current.nextSibling;
        current.nextSibling = child;
    }

    /*
     * Description:
     * Prints the organizational tree with indentation showing each level.
     *
     * Parameters:
     * current - the node where printing begins.
     * level - the current depth of the node in the tree.
     */
    void printTree(Node current, int level)
    {
        if (current == null)
        {
            return;
        }

        for (int i = 0; i < level; i++)
        {
            System.out.print("  ");
        }

        System.out.println(current.name);

        Node child = current.firstChild;

        while (child != null)
        {
            printTree(child, level + 1);
            child = child.nextSibling;
        }
    }
}


/*
 * HW3 reads an organizational tree from a data file and processes
 * queries from a second file.
 */
public class HW3
{
    /*
     * Description:
     * Reads the organization data, creates the tree, reads the queries,
     * and prints the answer for each query.
     *
     * Parameters:
     * args - command-line arguments. args[0] is the data filename
     *        and args[1] is the query filename.
     */
    public static void main(String[] args) throws FileNotFoundException
    {
        /*
         * Variables:
         * input - Scanner used to read the organization data file.
         * rootName - stores the name of the top person in the organization.
         * tree - stores the complete organizational tree.
         */
        
        Scanner input = new Scanner(new File(args[0]));

        String rootName = input.nextLine();

        Tree tree = new Tree(rootName);


        /*
         * This block reads each supervisor-subordinate pair from the
         * organization data file. For every line, the supervisor is
         * found in the tree and a new Node is created for the subordinate.
         * The new subordinate is then added to the supervisor's children.
         * The addChild method keeps the children in alphabetical order.
         */

        while (input.hasNextLine())
        {
            String line = input.nextLine();

            Scanner lineScanner = new Scanner(line);

            String supervisorName = lineScanner.next();
            String subordinateName = lineScanner.next();

            Node supervisor = tree.find(supervisorName);
            Node subordinate = new Node(subordinateName);

            tree.addChild(supervisor, subordinate);

            lineScanner.close();
        }

        input.close();


        /*
         * This block opens the query file and processes one query at a
         * time. The first word of each query identifies the type of
         * operation, and the second word identifies the main entity.
         * Additional information is read when a query needs a second
         * entity.
         */

        Scanner queries = new Scanner(new File(args[1]));

        while (queries.hasNextLine())
        {
            String line = queries.nextLine();

            Scanner queryScanner = new Scanner(line);

            String query = queryScanner.next();
            String entity = queryScanner.next();


            /*
             * This block handles the DirectSupervisor query.
             * It finds the direct supervisor of the specified entity.
             * The result is returned by the tree method and printed.
             */

            if (query.equals("DirectSupervisor"))
            {
                String supervisor = tree.getDirectSupervisor(entity);

                System.out.println(
                    "DirectSupervisor " + entity + " " + supervisor
                );
            }


            /*
             * This block handles the DirectSubordinates query.
             * It finds all of the entity's immediate children.
             * The children are already stored in alphabetical order.
             */

            else if (query.equals("DirectSubordinates"))
            {
                tree.printDirectSubordinates(entity);
            }


            /*
             * This block handles the IsSupervisor query.
             * It reads the second person's name from the query.
             * The program checks whether that person appears above
             * the first entity in the organizational tree.
             */

            else if (query.equals("IsSupervisor"))
            {
                String supervisor = queryScanner.next();

                boolean result = tree.isSupervisor(entity, supervisor);

                if (result)
                {
                    System.out.println(
                        "IsSupervisor " + entity + " " + supervisor + " yes"
                    );
                }
                else
                {
                    System.out.println(
                        "IsSupervisor " + entity + " " + supervisor + " no"
                    );
                }
            }


            /*
             * This block handles the IsSubordinate query.
             * It reads the second person's name and checks whether
             * that person appears below the first entity in the tree.
             * The result is printed as either yes or no.
             */

            else if (query.equals("IsSubordinate"))
            {
                String subordinate = queryScanner.next();

                boolean result = tree.isSubordinate(entity, subordinate);

                if (result)
                {
                    System.out.println(
                        "IsSubordinate " + entity + " " + subordinate + " yes"
                    );
                }
                else
                {
                    System.out.println(
                        "IsSubordinate " + entity + " " + subordinate + " no"
                    );
                }
            }


            /*
             * This block handles the AllSupervisors query.
             * It prints the entity's direct supervisor first and then
             * continues upward through the organization until reaching
             * the root of the tree.
             */

            else if (query.equals("AllSupervisors"))
            {
                tree.printAllSupervisors(entity);
            }


            /*
             * This block handles the AllSubordinates query.
             * It recursively visits every descendant of the entity.
             * The preorder traversal prints each subordinate before
             * visiting that subordinate's children.
             */

            else if (query.equals("AllSubordinates"))
            {
                tree.printAllSubordinates(entity);
            }


            /*
             * This block handles the NumberOfAllSupervisors query.
             * It counts every supervisor above the specified entity.
             * The count does not include the entity itself.
             */

            else if (query.equals("NumberOfAllSupervisors"))
            {
                int count = tree.numberOfAllSupervisors(entity);

                System.out.println(
                    "NumberOfAllSupervisors " + entity + " " + count
                );
            }


            /*
             * This block handles the NumberOfAllSubordinates query.
             * It counts every direct and indirect subordinate below
             * the specified entity. The entity itself is not counted.
             */

            else if (query.equals("NumberOfAllSubordinates"))
            {
                int count = tree.numberOfAllSubordinates(entity);

                System.out.println(
                    "NumberOfAllSubordinates " + entity + " " + count
                );
            }


            /*
             * This block handles the CompareRank query.
             * It reads the second entity and compares the number of
             * supervisors above each person. Fewer supervisors means
             * the person is higher in the organizational hierarchy.
             */

            else if (query.equals("CompareRank"))
            {
                String entity2 = queryScanner.next();

                String result = tree.compareRank(entity, entity2);

                System.out.println(
                    "CompareRank " + entity + " " + entity2 + " " + result
                );
            }


            /*
             * This block handles the ClosestCommonSupervisor query.
             * It reads the second entity and searches upward through
             * both paths in the tree. The first matching node is the
             * closest common supervisor.
             */

            else if (query.equals("ClosestCommonSupervisor"))
            {
                String entity2 = queryScanner.next();

                String result =
                    tree.closestCommonSupervisor(entity, entity2);

                System.out.println(
                    "ClosestCommonSupervisor " + entity + " "
                    + entity2 + " " + result
                );
            }

            queryScanner.close();
        }

        queries.close();
    }
}

