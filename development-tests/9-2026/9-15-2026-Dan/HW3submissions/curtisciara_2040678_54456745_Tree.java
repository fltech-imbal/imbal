/*	Author: Ciara Curtis
	Email: ccurtis2025@my.fit.edu
	Course: CSE2010
	Section: 4
	Description of this file: A similar class file to a singly
	linked list, but the nested node class holds a child variable
	can create its own sub tree by linking it's children nodes.
*/


import java.util.ArrayList;
public class Tree {

  //Nested private class that holds the information of each item added to the tree
  private class Node {
    String name;
    Node parent;
    Node sibling;
    Node children;  //Stores a reference to the Node's children. Will act as a singly
		    //linked list through the sibling reference variable

    //Constructor just for the name of Node
    public Node(String childName) {
      name = childName;
      parent = null;
      sibling = null;
      children = null;
    }

    // Node class getter and setter methods
    public void setParent(Node parentNode) {
      parent = parentNode;
    }

    public void setSibling(Node siblingNode) {
      sibling = siblingNode;
    }
		
    public void setChildren(Node childrenNode) {
      children = childrenNode;
    }
		
    public String getName() { return this.name; }
    public Node getParent() { return this.parent; }
    public Node getSibling() { return this.sibling; }
    public Node getChildren() { return this.children; }

  }  //-------------End of Nested Node Class----------------

  private Node root; //holds the start of the tree

  //Tree Constructor that preestablishes the root
  public Tree (String treeRoot) {
    root = new Node(treeRoot);
  }


  /*Method that finds and returns a specific Node based on
    a given string */
  public Node findNode(String name) {
    SLLTree<Node> nodesToCheck = new SLLTree<Node>();
    nodesToCheck.addLast(root);
    //Holds the nodes that will be compared to the child string
    //and immediately adds the root of the tree to the list

    while(!nodesToCheck.isEmpty()) { //while there are still nodes to check
      Node temp = nodesToCheck.removeFirst();
      if(temp.getName().equals(name)) { //if the node being taken from the list has the same name as the given string
        return temp; }

      // If the chosen node from the nodesToCheck ArrayList does not match the parameter,
      // add the node's children to the list and continue the loop.
      else {
        if(temp.getChildren() != null) {
          Node current = temp.getChildren();
          while(current != null) {
            nodesToCheck.addLast(current);
            current = current.getSibling();
          }
        }
      }
    }
  
    return null; //The node was never found

  }

  //method to add children to the tree and presort any preexsisting children
  public void addChild(String parent, String child) {
    Node newChild = new Node(child);
    Node parentNode = findNode(parent);  //finds the parent node using the findNode method
    newChild.setParent(parentNode);      //after finding the parent, newChild node's parent pointer is set to the parent

    //if parentNode children pointer is null, then set children pointer to newChild
    if(parentNode.getChildren() == null) parentNode.setChildren(newChild);
    else { //else sort the new child with the other children
      if(parentNode.getChildren().getName().compareTo(child) > -1) {
      //if the first child's name in the list comes after lexigraphically to the newChild name
        newChild.setSibling(parentNode.getChildren());
	parentNode.setChildren(newChild);
      }

      //Covers the special cases where the new child is placed in the middle or end of the child list.
      else {
        Node current = parentNode.getChildren();
	Node previous = null; 
	while (current != null) {   //while loop searches list to see if the order in the middle of the list needs to be changed
	  if (current.getName().compareTo(child) > -1) {
	  newChild.setSibling(current);
          previous.setSibling(newChild);
	  }
	  previous = current;
	  current = current.getSibling();
	}
	
	//if newChild node is lexigraphically after every node in sibling list,
	//set the last child's sibling to the newChild
        if (current == null) {
	  previous.setSibling(newChild);
        }
      } 
    }
  }
       
  /* Method that finds the parent of a child given from
     the String parameter. Utilizes the findNode method
     to find the child node and returns the parent*/
  public String findParent(String child) {
    Node childNode = findNode(child);
    if(childNode.getParent() != null) return childNode.getParent().getName();
    else return "none";    
  }


  /* Method that returns a parent node's
     string list of children
  */
  public String listChildren(String parent) {
    Node parentNode = findNode(parent);
    if (parentNode.getChildren() == null) return "none";  //checks if parent node has children
    else {  //the node's children get added to the children string and returned
      String children = "";
      Node current = parentNode.getChildren();
      while(current != null) { //appends all the childen of the node to a string variable to be returned
	children += current.getName() + " ";
        current = current.getSibling();
      }
      return children;
    }
  }


  /* Method that returns the parent of a child, and the,
     parent of the parent, and so on until a node's parent 
     pointer is null */
  public String allParents(String child) {
    Node childNode = findNode(child);
    if(childNode.getParent() == null) return "none";  //checks if node has a parent
    Node current = childNode.getParent();
    String parents = "";
    while(current != null) {    //Builds a string of a node's parent, grandparent, and so on
      parents += current.getName();
      current = current.getParent();
    }
    return parents;
  }

  // Method that returns the amount of parents a child has
  public int countParents(String child) {
    Node childNode = findNode(child);
    if(childNode.getParent() == null) return 0;

    Node current = childNode;
    int parentAmount = 0;      //Used to count how many generations/rank of a specific node
    while(current != null) {   //each time current updates to a new parent, the generation counter increases
      parentAmount++;   
      current = current.getParent();
    }

    return --parentAmount;
  }

  //Method that returns a String containing all of the children a parent node has
  public String allChildren(String parent) {
    Node parentNode = findNode(parent);
    if(parentNode.getChildren() == null) return "none";

    String children = "";  //Will hold all the children
    ArrayList<Node> nodesToAdd = new ArrayList<Node>(); //Will hold all the children needed to be added to the return string
    nodesToAdd.add(parentNode.getChildren());

    while(!nodesToAdd.isEmpty()) { //while there are still nodes to add
      Node temp = nodesToAdd.remove(0);
      children += temp.getName() + " ";  
      
      if(temp.getChildren() != null) {        //checks if the node has children to add to the arrayList
        Node current = temp.getChildren();
        while(current != null) {	      //adds to the arrayList all the children in the list for a specific node
          nodesToAdd.add(current);
          current = current.getSibling();
        }
      }
    }

    return children;

  }



  //Method that returns the amount of children a parent node has {
  public int countChildren(String parent) { 
    Node parentNode = findNode(parent);
    if(parentNode.getChildren() == null) return 0;

    int children = 0;  //children counter
    ArrayList<Node> nodesToAdd = new ArrayList<Node>(); //Will hold all the children needed to be added to the return string
    nodesToAdd.add(parentNode.getChildren());

    while(!nodesToAdd.isEmpty()) { //while there are still nodes to add
      Node temp = nodesToAdd.remove(0);
      children++;

      if(temp.getChildren() != null) {        //checks if the node has children to add to the arrayList
        Node current = temp.getChildren();
        while(current != null) {              //adds to the arrayList all the children in the list for a specific node
          nodesToAdd.add(current);
          current = current.getSibling();
        }
      }
    }

    return --children;

  }


  //Method that returns the common parent between two nodes
  public String commonParent(String child1, String child2) {
    Node child1Node = findNode(child1);
    Node child2Node = findNode(child2);
    int child1Rank = countParents(child1);
    int child2Rank = countParents(child2);
    int rankDifference = child1Rank - child2Rank;

    //special case: either of the nodes are the root node with no parent
    /*Checks if either node is the root node with no parent; if true, then
      returns none because there is no parent to the root node to compare
      so there is no common parent */
      if(child1Rank == 0 || child2Rank == 0) return "none";


    //special case: child 1 is of higher rank than child 2
    if(rankDifference < 0) {
      while(rankDifference < 0) {
        child2Node.getParent();
        rankDifference++;
      }
    }

    //special case: child 2 is of higher rank than child 1
    if(rankDifference > 0) {
      while(rankDifference > 0) {
        child1Node.getParent();
        rankDifference--;
      }
    }

    /*Equalizing the rank of the nodes (the two previous if statements) allows
      you to properly compare the parent nodes to find a common parent */
    //while loop that continues until both children reach the root node
    while(child1Node != null && child2Node != null) {
        if(child1Node.getParent().getName().equals(child2Node.getParent().getName()))
          //if the parent of child 1 equals the parent of child 2, return the common parent
          return child1Node.getParent().getName();
        else {
          child1Node = child1Node.getParent();
          child2Node = child2Node.getParent();
        }
    }

    return "none";

  }




}
