/*
  Author: Sean Delate
  Email: sdelate2025@my.fit.edu
  Course: CSE2010
  Section: 01
  Description of this file: Creates a tree for data storage that can reproduce the following commands
  addChild(node, childNode) - adds a child node
  getChildren(node) - returns all child nodes
  getParent(node) - returns the parent node
  In addition the tree should be able to excecute some commands from the files inputted
  Such as adding people to the supervisor tree
  */
import java.io.File; // Needed to read files
import java.io.FileNotFoundException; // Needed to handle errors
import java.util.ArrayList;
import java.util.Collections; // Needed to use as a changable array for inputs
import java.util.List; // needed to sort alphabeticlly
import java.util.Scanner; // Needed for collection
public class HW3<E extends Comparable<E>>
{
    //Basic node class
    public static class Node<E extends Comparable<E>> {
        
        private E data; //Stores name

        //Empty constructor
        public Node() {
            data = null;
        }

        //Constructor
        public Node(E e) {
            data = e;
        }

        //Gets data
        public E data() {
            return data;
        }

    }

    // Tree class
    public static class Tree<E extends Comparable<E>> extends Node<E> {
        
        private Tree<E> parent; //points to parent node
        
        private List<Tree<E>> children = new ArrayList<>(); //Points to all children in a list

        //Empty construtcor
        public Tree() {
            super();
        }

        //Constructor
        public Tree(E e) {
            super(e);
        }

        //returns list of children
        public List<Tree<E>> getChildren() {
            return children;
        }

        //returns parent
        public Tree<E> getParent() {
            return parent;
        }

        //Adds a child then sorts the list
        public void addChild(Tree<E> p, Tree<E> c) {
            c.parent = p;
            p.children.add(c);
            p.children.sort((child1, child2) -> child1.data().compareTo(child2.data())); //Couldent figure out how to use collections.sort here so i used a different method
        }
    }

    private Tree<E> top = new Tree<>(); //Top of the tree - used to find nodes

    //Returns top
    public Tree<E> top() { 
        return top;
    }

    //Adds a new top
    public void addTop(E e) {
        top = new Tree<>(e);
    }

    //Finds the node using a recursive method
    public Tree<E> findNode(E e) {
        if (top == null) {
            return null;
        }
        return findNode(top, e);
    }

    //recursively checks every child under a parent for a specified node
    private Tree<E> findNode(Tree<E> node, E e) {
        if (node == null) {
            return null;
        }
        if (node.data().equals(e)) {
            return node;
        }
        for (Tree<E> child : node.getChildren()) {
            Tree<E> match = findNode(child, e);
            if (match != null) {
                return match;
            }
        }
        return null;
    }

    //Gets direct subordinates or none
    public void directSubordinates(E e) {
        Tree<E> node = findNode(e);
        if (node.getChildren().isEmpty()) {
            System.out.print(" none");
        } else {
            List<E> list = new ArrayList<>();
            for (int i = 0; i < node.getChildren().size(); i++) {
                list.add(node.getChildren().get(i).data());
            }
            Collections.sort(list);
            printAll(list);
        }
    }

    //uses collect descendednts to get all children and childrens children
    public void allSubordinates(E e) {
        Tree<E> node = findNode(e);
        if (node == null || node.getChildren().isEmpty()) {
            System.out.print(" none");
            return;
        }

        List<E> list = new ArrayList<>();
        collectDescendants(node, list);
        printAll(list);
    }

    //recursivly gets each child from the node and then looks at their children
    private void collectDescendants(Tree<E> node, List<E> list) {
        for (Tree<E> child : node.getChildren()) {
            list.add(child.data());
            collectDescendants(child, list);
        }
    }

    //counts the number of subordinates or none using count descendents
    public int numAllSubordinates(E e) {
        Tree<E> node = findNode(e);
        if (node == null) {
            return 0;
        }
        return countDescendants(node);
    }

    //Basicly the same as the collect descendents but adds to a counter instead of a list
    private int countDescendants(Tree<E> node) {
        int count = 0;
        for (Tree<E> child : node.getChildren()) {
            count++;
            count += countDescendants(child);
        }
        return count;
    }

    //Checks if something is a subordinate by going up the chain from c to the head and looks for p
    public void isSubordinate(E p, E c) {
        Tree<E> current = findNode(c);
        while (current != null) {
            if (current.data().equals(p)) {
                System.out.print(" yes");
                return;
            }
            current = current.getParent();
        }
        System.out.print(" no");
    }

    //Just checks what the found node with data e's parent is
    public void directSupervisor(E e) {
        Tree<E> node = findNode(e);
        System.out.print(node.getParent() == null ? " none" : (" " + node.getParent().data()));
    }

    //Goes up the hain like in is subordinate but instead adds them all to a list
    public List<E> allSupervisors(E e) {
        Tree<E> node = findNode(e);
        List<E> list = new ArrayList<>();
        while (node != null && node.getParent() != null) {
            node = node.getParent();
            list.add(node.data());
        }
        return list;
    }

    //Same thing as all supervisors but adds to a counter instead of a list
    public int numAllSupervisors(E e) {
        Tree<E> node = findNode(e);
        int count = 0;
        while (node.parent != null) {
            count++;
            node = node.getParent();
        }
        return count;
    }

    //does the same thing as issubordinate but it goes in the opposite direction
    public void isSupervisor(E c, E p) {
        Tree<E> child = findNode(c);
        boolean isSup = false;
        if (child.getParent() == null) {
            System.out.print(" no");
        } else {
            while (child.parent != null && !isSup) {
                if (child.data().equals(p)) {
                    isSup = true;
                }
                child = child.getParent();
            }
            System.out.print(isSup ? " yes" : " no");
        }
    }

    //finds the number for numallsupervisors for the nodes with data e1 and e2 and then uses that to find which is higher rank
    public void compareRank(E e1, E e2) {
        int x = numAllSupervisors(e1);
        int y = numAllSupervisors(e2);
        if (x < y) {
            System.out.print(" higher");
        } else if (x == y) {
            System.out.print(" same");
        } else {
            System.out.print(" lower");
        }
    }

    //compares every supervisor for e2 to the first supervisor in e1 then the second and it increents through them all
    public String closestCommonSupervisor(E e1, E e2) {
        List<E> list1 = allSupervisors(e1);
        List<E> list2 = allSupervisors(e2);
        if (list1 == null || list2 == null) {
            return " none";
        }
        for (E i : list1) {
            for (E j : list2) {
                if (j.equals(i)) {
                    return " " + j;
                }
            }
        }
        return " none";
    }

    //Used to print lists or none if the list is empty
    public void printAll(List<E> list) {
        if (list.isEmpty()) {
            System.out.print(" none");
        } else {
            for (E e : list) {
                System.out.print(" " + e);
            }
        }
    }

    //Main
    public static void main(String[] args)
    {
        HW3<String> systemRequests = new HW3<>(); // Used to run methods

		File data = new File(args[0]); // Takes the command line arguement for the file and then uses it to make a new file object to be read later

        File input = new File(args[1]); // Takes the command line arguement for the file and then uses it to make a new file object to be read later

        // A try catch is needed to handle if the file with the chat commands cannot be found
		try (Scanner scan = new Scanner(data)) {
			// Looks through the file line by line
            String root = scan.nextLine().trim(); //Trim is probably not neccesary but it catches formatting errors
            //Adds the first line to the top
            systemRequests.addTop(root);

            //Increments through every line
            while (scan.hasNextLine()) {
                String treeStart = scan.nextLine().trim();
                //If the line is empty continues
                if (treeStart.isEmpty()) {
                    continue;
                }

                String[] peopleTree = treeStart.split("\\s+"); //Splits line into seperate strings
                Tree<String> parent = systemRequests.findNode(peopleTree[0]); //Looks for the parent in the tree using the data given by the first string on the line
                Tree<String> child = new Tree<>(peopleTree[1]); //Creates a new child node using the data from the second string on the line

                parent.addChild(parent, child); //Adds the child node to the parent node
            }
		} catch (FileNotFoundException e) {
			System.out.println("Error: file not found");
			e.printStackTrace();
		}

        // A try catch is needed to handle if the file with the chat commands cannot be found
		try (Scanner scan = new Scanner(input)) {
			// Looks through the file line by line
			while (scan.hasNextLine()) {
				// Reads the full line of text as a string input and then splits it into an array by spaces
				String info = scan.nextLine();
				String[] instruction = info.split("\\s");
					// Looks at the first element of the string array and depending on the command excecutes a method
                    // usually prints names and instructions before the instruction results
                    // With the instructions without printAll it will typically use a println at the end to get a new line
					switch (instruction[0]) {
						case "DirectSupervisor":
							System.out.print(instruction[0] + " " + instruction[1]);
							systemRequests.directSupervisor(instruction[1]);
							System.out.println();
							break;
						case "DirectSubordinates":
							System.out.print(instruction[0] + " " + instruction[1]);
							systemRequests.directSubordinates(instruction[1]);
							System.out.println();
							break;
						case "AllSupervisors":
							System.out.print(instruction[0] + " " + instruction[1]);
							systemRequests.printAll(systemRequests.allSupervisors(instruction[1]));
							System.out.println();
							break;
						case "AllSubordinates":
							System.out.print(instruction[0] + " " + instruction[1]);
							systemRequests.allSubordinates(instruction[1]);
							System.out.println();
							break;
						case "NumberOfAllSupervisors":
							System.out.print(instruction[0] + " " + instruction[1]);
							System.out.println(" " + systemRequests.numAllSupervisors(instruction[1]));
							break;
						case "NumberOfAllSubordinates":
							System.out.print(instruction[0] + " " + instruction[1]);
							System.out.println(" " + systemRequests.numAllSubordinates(instruction[1]));
							break;
						case "IsSupervisor":
							System.out.print(instruction[0] + " " + instruction[1] + " " + instruction[2]);
							systemRequests.isSupervisor(instruction[1], instruction[2]);
							System.out.println();
							break;
						case "IsSubordinate":
							System.out.print(instruction[0] + " " + instruction[1] + " " + instruction[2]);
							systemRequests.isSubordinate(instruction[1], instruction[2]);
							System.out.println();
							break;
						case "CompareRank":
							System.out.print(instruction[0] + " " + instruction[1] + " " + instruction[2]);
							systemRequests.compareRank(instruction[1], instruction[2]);
							System.out.println();
							break;
						case "ClosestCommonSupervisor":
							System.out.print(instruction[0] + " " + instruction[1] + " " + instruction[2]);
							System.out.println(systemRequests.closestCommonSupervisor(instruction[1], instruction[2]));
							break;
						default:
							break;
					}
			}
		} catch (FileNotFoundException e) {
			System.out.println("Error: file not found");
			e.printStackTrace();
		}
    }
}
