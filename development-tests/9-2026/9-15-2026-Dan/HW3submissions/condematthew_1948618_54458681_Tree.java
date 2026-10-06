/*

  Author: Matthew Conde
  Email: mconde2022@my.fit.edu
  Course: CSE 2010
  Section: 02
  Description of this file:
			Code that takes in two file inputs, first a list of names that consists first of the highest supervisor, and then assigns subordinates to supervisors, 
			creating a tree structure. The second file contains queries that ask for information about a given person in the tree, the tree is organized as a 
			linked list with each node having an array list of children, a name, and a pointer to its parent node.

 */

import java.io.File;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Arrays;

public class Tree {	
			// creating the node class to store name, point to parent, and to arraylist of children
	private static class Node {
		Node parent;
		String name;
		ArrayList<Node> children;
		
		Node (String name, Node parent) {
			this.name = name;
			this.parent = parent;
			this.children = new ArrayList<Node>();
		}
	}
			// initially empty root
	public Node root = null;	
			// function to set root
	public void setRoot (String name){
		root = new Node(name, null);
	}
			// function to get node, starts at a given node and searches all children recursively to find a name
	public Node getNode(String name, Node start){
		Node current = start;
		
		if(current == null){
			return null;
		}
		
		if (current.name.equals(name)){
			return current;
		}
	
		for (Node child : current.children){
			Node found = getNode(name, child);
			if (found != null){
				return found;
			}
		}
		return null;
	}
			// function starts at a node and cycles through parents until root is reached, counts each step
	public int depth (Node current){
		int count = 0;
		while (current !=null){
			count += 1;
			current = current.parent;
		}
		return count;
	}
			// function to add a child, searches for a child that is alphabetically behind the given node's name and inserts node here
	public void addChild (Node parent, Node child){
		if(parent.children.isEmpty()){
			parent.children.add(child);
			return;
		}

		for (int i = 0; i< parent.children.size(); i++){
			Node currentChild = parent.children.get(i);

			if (child.name.compareTo(currentChild.name)<0){
				parent.children.add(i, child);
				return; 
			}
		}
		parent.children.add(child);	
		return;
	}
			// function that returns list of children for a given node
	public ArrayList<Node> getChildren(Node parent){
		return parent.children;
	}
			// function that returns parent node of a given node
	public Node getParent(Node child){
		return child.parent;
	}
			// function serves two purposes, recursively prints all nodes subordinate to a given node, or cycles through all nodes subordinate to count the amount of descendants
	public int printPreOrder(Node current, Boolean counting){

		if (current == null){
			return 0;
		}
		if(counting){
			int totalSubs = current.children.size();

	                for (int i = 0; i< current.children.size(); i++){
        	                totalSubs += printPreOrder(current.children.get(i), true);
                	}
			return totalSubs;
		}
		for (int i = 0; i< current.children.size(); i++){
			System.out.print(" " + current.children.get(i).name);
			printPreOrder(current.children.get(i), false); 
		}
		return 0;

	}
	

    /*
      Description of each method, including parameters 
    */
	public static void main(String[] args) throws Exception
	{	
		Tree myTree = new Tree();		//create new instance of tree, pull first input file, open scanner
		File dataFile = new File(args[0]);
		Scanner sc1 = new Scanner(dataFile);       

		myTree.setRoot(sc1.next());		// sets root from first line oof first input file, intializes strings to be used in the rest of the function
		String supervisor, subordinate = null;
				
		while (sc1.hasNext()){			// while there is another line available, pull supervisor and subordinate, then create node for child and point to supervisor
			supervisor = sc1.next();
			subordinate = sc1.next();

			Node parent = myTree.getNode(supervisor, myTree.root);
			Node child = new Node(subordinate, parent);

			myTree.addChild(parent, child); 	//add child to parent node's list
		}

		sc1.close();

                File inputFile = new File(args[1]);	// pulls info from second file input, new scanner opened
                Scanner sc = new Scanner(inputFile);

		String keyword, entity1, entity2, status  = null;	//initializes all variables to be used in switch case
		Node found, node1, node2, current;
		boolean match = false;
		int count = 0;

        	while (sc.hasNext()) {                     
			keyword = sc.next();		// while there is still a line of input, pull first string as the query operative

            		switch (keyword) {                                     
                		case "DirectSupervisor":       // when looking for direct super, get node, then check if it has a parent, print appropriately
                    			entity1 = sc.next();
					found = myTree.getNode(entity1, myTree.root);
					
					System.out.print(keyword + " " + entity1);

					if (found != null && found.parent != null){
						System.out.println(" " + found.parent.name);
					} else {
						System.out.println(" none");
					}					
                    			break;

				case "DirectSubordinates":	// when looking for direct subs, get node, cycle through children and print, otherwise print none
					entity1 = sc.next();
					found = myTree.getNode(entity1, myTree.root);
					match = false;

					System.out.print(keyword+ " " + entity1);
					for (int i = 0; i< found.children.size(); i++){
						System.out.print(" " + found.children.get(i).name);
						match = true;	
					}
					
					if (!match){
						System.out.print(" none");
					}
					System.out.println();
					break;

				case "AllSupervisors":		// when looking for all supers, cycle through all parents until no parent left, print output or print none if root
					entity1 = sc.next();
					found = myTree.getNode(entity1, myTree.root);					

					System.out.print(keyword+ " " + entity1);

					if (found.parent == null){
						System.out.println(" none");
					} else {
						Node next = found.parent;
						while(next != null){
							System.out.print(" " + next.name);
							next = next.parent;
						}
					}
					System.out.println();
					break;

				case "AllSubordinates":		// when looking for all subordinates, print none if none, otherwise call printPreOrder, use false in args if not counting
					entity1 = sc.next();
					found = myTree.getNode(entity1, myTree.root);

					System.out.print(keyword+ " "+ entity1);

					if (found == null || found.children.isEmpty()) {
						System.out.println(" none");
					} else {
						myTree.printPreOrder(found, false);
						System.out.println();
					}
					break; 

				case "NumberOfAllSupervisors":		// when looking for number of all supers, cycle through parents until hit root, print appropriate output
					entity1 = sc.next();
				        found = myTree.getNode(entity1, myTree.root);

                                        System.out.print(keyword+ " " + entity1);
					
					found = found.parent;
					count = 0;
                                        while(found != null){
						count += 1;
                                                found = found.parent;
                                        }
					System.out.println(" " + count);
                                        break;

                                case "NumberOfAllSubordinates":		// when looking for number of subs, call printPreOrder with true argument to count
                                        entity1 = sc.next();
                                        found = myTree.getNode(entity1, myTree.root);

                                        System.out.print(keyword + " " + entity1 + " ");

                                        if (found == null){
                                                System.out.println(0);
                                               
                                        } else {
						System.out.println(myTree.printPreOrder(found, true));
					}

                                        break;

				case "IsSupervisor":		// when checking if someone is someone else super, get node and cycle through parents until found or root, print appropriately
					entity1 = sc.next();
					supervisor = sc.next();

					found = myTree.getNode(entity1, myTree.root);

					System.out.print(keyword + " " + entity1 + " " + supervisor);
					current = found.parent;
					
					match = false;
					while (current != null){
						if (current.name.equals(supervisor)){
							System.out.println(" yes");
							match = true;
							break;
						}
						current = current.parent;
					}
					if (!match){
						System.out.println(" no");
					}
					break;

				case "IsSubordinate":		// when checking if someone is someone elses sub, start at child and loop through parents to see if super is a supervisor
					entity1 = sc.next();
				        subordinate = sc.next();

					found = myTree.getNode(subordinate, myTree.root);
					match = false;
						
					System.out.print(keyword + " " + entity1 + " " + subordinate);
					current = found.parent;

					while (current != null){
						if(current.name.equals(entity1)){
							System.out.println(" yes");
							match = true;
							break;
						}
						current = current.parent;
					} 
					if (!match){
						System.out.println(" no");
					}
					break;

				case "CompareRank":			// compare two peoples rank, cycle through parents of both nodes simultaneously, if either is null then end loop
									// check if they reached null at same time or if one reached null first, whoever reached null first has higher rank
					entity1 = sc.next();		// print appropriately
					entity2 = sc.next();
					
					node1 = myTree.getNode(entity1, myTree.root);
					node2 = myTree.getNode(entity2, myTree.root);
					
					while (node1 != null &&  node2 != null){

						node1 = node1.parent;
						node2 = node2.parent;

					}

					if (node1 == null && node2 == null) { 
					        status = " same";
    					} else if (node1 == null) { 
					        status = " higher";
    					} else if (node2 == null) { 
					        status = " lower";
    					}
					System.out.println(keyword +" "+ entity1 +" "+ entity2 + status);
					break;
					
				case "ClosestCommonSupervisor":		// to check for closest common super, start at one node and move to parent, for each parent of node1
									// cycle through parents of node2 to check if there's a common super, then move to next parent of node1
					entity1 = sc.next();		// if match is found then break cycle and print appropriately
					entity2 = sc.next();

					node1 = myTree.getNode(entity1, myTree.root);
					node2 = myTree.getNode(entity2, myTree.root);
					
					current = node1.parent;
					match = false;
					supervisor = " none";
				
					while (current != null && !match) {
						Node check = node2.parent;
						
						while(check != null) {
							if (current.name.equals(check.name)){
								supervisor = " "+ check.name;
								match = true;
								break;
							}
							check = check.parent;
						}
						current = current.parent;
					}

					System.out.println(keyword+ " "+ entity1 + " " + entity2+ supervisor);
					break;
			}	
        	}
		sc.close();
	}

}
