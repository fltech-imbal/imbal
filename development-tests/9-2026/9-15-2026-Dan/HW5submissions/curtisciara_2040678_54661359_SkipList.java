/* Author: Ciara Curtis
   Email: ccurtis2025@my.fit.edu
   Course: CSE2010
   Section: 4
   Description of this file: This class contains a nested QuadNode
	class that holds a key and a value, along with four reference 
	variables that points towards other QuadNode Objects. The
	SkipList class utilizes the QuadNode class to create horizontal
        lists that hold data. These horizontal lists are connected vertically
        and specific values can be searched for through the list. */


import java.util.ArrayList;
public class SkipList {

  /*Private QuadNode class that creates an object with a integer key
    to sort and a string value to hold. The QuadNode class also holds
    four reference variables to other QuadNodes: a previous and next node
    to keep track of a horizontal list positoning, an above and below
    node to keep track of the vertical list positioning and a connection
    to another horizontal list*/
  private class QuadNode {

    // nested class instance variables
    int key;
    String value;
    QuadNode above;
    QuadNode below;
    QuadNode previous;
    QuadNode next;

    //empty constructor to create an empty QuadNode
    public QuadNode() { }
    
    //setter methods
    public void setKey(int QuadNodeKey) { this.key = QuadNodeKey; }
    public void setValue(String QuadNodeValue) { this.value = QuadNodeValue; }
    public void setAbove(QuadNode QuadNodeAbove) { this.above = QuadNodeAbove; }
    public void setBelow(QuadNode QuadNodeBelow) { this.below = QuadNodeBelow; }
    public void setPrevious(QuadNode QuadNodePrevious) { this.previous = QuadNodePrevious; }
    public void setNext(QuadNode QuadNodeNext) { this.next = QuadNodeNext; }

    //getter methods
    public int getKey() { return this.key; }
    public String getValue() { return this.value; }
    public QuadNode getAbove() { return this.above; }
    public QuadNode getBelow() { return this.below; }
    public QuadNode getPrevious() { return this.previous; }
    public QuadNode getNext() { return this.next; }
  }//---------------End of Nested QuadNode class---------------------

  //bottom non-data nodes that will be a reference
  //to the bottom list
  QuadNode NEG_INFbottom;
  QuadNode POS_INFbottom;
  
  //top non-data nodes that will be a reference
  //to the top list
  QuadNode NEG_INFtop;
  QuadNode POS_INFtop;

  //holds the amount of horizontal lists
  int size;

  //height generator for each node in the list
  FakeRandHeight heights = new FakeRandHeight();

  //Constructor to instantiate all the QuadNode
  //references, and their reference pointers
  public SkipList() {
    NEG_INFbottom = new QuadNode();
    POS_INFbottom = new QuadNode();
    NEG_INFtop = new QuadNode();
    POS_INFtop = new QuadNode();

    POS_INFtop.setKey(5000);
    POS_INFbottom.setKey(5000);

    /*Points the positive infinity nodes to
      their respective negative infinity
      nodes, and vice-versa */ 
    NEG_INFbottom.setNext(POS_INFbottom);
    POS_INFbottom.setPrevious(NEG_INFbottom);
    NEG_INFtop.setNext(POS_INFtop);
    POS_INFtop.setPrevious(NEG_INFtop);

    //Sets the bottom list nodes to point
    //to the top list nodes, and vice-versa
    NEG_INFbottom.setAbove(NEG_INFtop);
    NEG_INFtop.setBelow(NEG_INFbottom);
    POS_INFbottom.setAbove(POS_INFtop);
    POS_INFtop.setBelow(POS_INFbottom);

    size = 1; //there is now one level to the skiplist
  }

  public boolean isEmpty() {
    //if the bottom nodes are still connected to each other
    return NEG_INFbottom.getNext() == POS_INFbottom;
  }

  /*Method that takes an integer as a key to find
    search for. A temporary node is used to keep track
    of which node will be checked in the list. At every 
    tier, the temporary node's key is compared to the
    key given and if the key given is larger than
    the temporary node's key, the temporary node is
    updated to the next node in the list */
  public String get(int key) {
    //starts at the topmost horizontal list
    QuadNode current = NEG_INFtop;

    //while there is a still a horizontal list below the
    //one currently being checked
    while(current.getBelow() != null) {
      //drop down to the tier below
      current = current.getBelow();

      //while the key being searched for is still greater than
      //or equal to the next node's key, update the current
      //node to the next node
      while(key >= current.getNext().getKey()) {
        //update the current node to the next node 
        current = current.getNext();
      }
    }

    //if the key is found, return the value associated with that key
    if(current.getKey() == key) { return current.getValue(); }
    else return null;
  }

  /*Method to delete any empty rows left over */
  public void deleteEmpty() {
    QuadNode current = NEG_INFtop.getBelow();
    while(current != NEG_INFbottom) {
      //if no other nodes besides the placeholder nodes, remove the horizontal list
      if(current.getNext().getKey() == 5000) {
        current.getAbove().setBelow(current.getBelow());
        current.getBelow().setAbove(current.getAbove());
        current.getNext().getAbove().setBelow(current.getNext().getBelow());
        current.getNext().getBelow().setAbove(current.getNext().getAbove());
	size--;
      }
      current = current.getBelow();
    }
  }

  /*Almost exactly like the get() method,
    but instead of returning null when the
    key is not found, it returns the quadNode
    with the largest key less than or equal
    to the key given. */
  public QuadNode floorEntry(int key) {
    //starts at the topmost horizontal list
    QuadNode current = NEG_INFtop;

    //while there is a still a horizontal list below the
    //one currently being checked
    while(current.getBelow() != null) {
      //drop down to the tier below
      current = current.getBelow();

      //while the key being searched for is still greater than
      //or equal to the next node's key, update the current
      //node to the next node
      while(key >= current.getNext().getKey()) {
        //update the current node to the next node
        current = current.getNext();
      }
    }

    return current;
  }  //end of method


  /* Method that utilizes floorEntry() method, to
     find the entry closest to the given key without
     going over. If the key given is equal to the found 
     key, then the floor node is returned. Else, the entry
     directly after is returned*/
  public QuadNode ceilingEntry(int key) {
    QuadNode floor = floorEntry(key);
    if(floor.getKey() == key) return floor;
    else return floor.getNext();
  }  


  /* Method that inserts a key,value entry into
     the skiplist but utilizing the floorEntry method
     to find the nearest quadNode with a value close to
     but not over the given key. If the floorNode found
     has a key equal to the given key, then previous
     value is saved and returned after changing each value
     in the found tower. If the key is not found, then the
     value is inserted normally.*/
  public String put(int key, String value) {

    //Finds the closest entry to the given key
    QuadNode floorNode = floorEntry(key);
    
    /* If the key is found, save the value to return
       later and replace the old value with the new value
       given */
    if(floorNode.getKey() == key) { 
      String returnValue = floorNode.getValue();

      //while there are still horizontal lists to change
      while(floorNode != null) {
        //replaces the floorNode value
	floorNode.setValue(value);
        floorNode = floorNode.getAbove();
      }

      //return the saved value after everything is changed
      return returnValue;
    }

    else {
      int nodeHeight = heights.get();
      /* If the height of the node is greater than the height
         of the SkipList, make another tier and replace the
         class reference variables for the top of the list */
      while(nodeHeight >= this.size) {

        size++;

	/*Creates new quadNodes that will replace the top
          positive and negative infinity placeholder nodes */
	QuadNode newNEG_INFtop = new QuadNode();
        QuadNode newPOS_INFtop = new QuadNode();
        newPOS_INFtop.setKey(5000);
          
        /*Makes the new negative and positive infinity placement
          nodes point towards each other, then updates the below
          reference variable to point to the old placement nodes*/
        newNEG_INFtop.setNext(newPOS_INFtop);
        newPOS_INFtop.setPrevious(newNEG_INFtop);
	newNEG_INFtop.setBelow(this.NEG_INFtop);
        newPOS_INFtop.setBelow(this.POS_INFtop);

        /*Finally points the old placement nodes above references
          to the new top placement nodes. Then updates the class
          reference variables to point to the new placement nodes */
        this.NEG_INFtop.setAbove(newNEG_INFtop);
        this.POS_INFtop.setAbove(newPOS_INFtop);
        NEG_INFtop = newNEG_INFtop;
        POS_INFtop = newPOS_INFtop;

      }  //end of while loop

        QuadNode belowNode = null;

        for(int i = 0; i <= nodeHeight; i++) {
	  //creates new node
	  QuadNode newNode = new QuadNode();
          newNode.setKey(key);
	  newNode.setValue(value);

          //sets the new node right after the floorNode found
	  newNode.setPrevious(floorNode);
          newNode.setNext(floorNode.getNext());
          floorNode.getNext().setPrevious(newNode);
          floorNode.setNext(newNode);

          //connects the new node to the list below it, if there
          //is one
          if(belowNode != null) {
	    newNode.setBelow(belowNode);
            belowNode.setAbove(newNode);
          }

	  //updates the below node to the current list
	  belowNode = newNode;

          if(i < nodeHeight) { 
            //while loop to find the next node with an above
            //node pointer that isn't null
	    while(floorNode.getAbove() == null) {
	      floorNode = floorNode.getPrevious();
            }

            floorNode = floorNode.getAbove();
	   }

        }  //end of for loop

    }    //end of else statement

    //key did not already exsist so nothing is returned
    return null;

  }

  /* Method that utilizes the floor method to find the entry
     that matches the given key. If the key is found, then
     the horizontal list is updated similar to a doubly 
     linked list, where the node previous to the found node
     will point to the found node's next and vice versa. A
     while loop is used to remove the node from each tier the
     node is present */
  public String remove(int key) {
    QuadNode floorNode = floorEntry(key);
    if(floorNode.getKey() == key) {
      String returnValue = floorNode.getValue();
      while(floorNode != null) {
        floorNode.getPrevious().setNext(floorNode.getNext());
        floorNode.getNext().setPrevious(floorNode.getPrevious());

	//after delting the node from the horizontal list,
        //the floorNode is changed to the one above it
        floorNode = floorNode.getAbove();
      }
      this.deleteEmpty();
      return returnValue;
    }    
    else return null;
  }


  /*Print method that automatically prints the entire list
    in tiered format */
  public void toPrint() {
    QuadNode currentTier = NEG_INFtop.getBelow();  //keeps track of the horizontal list
    QuadNode currentNode = currentTier; //keeps track of the index of the horizontal list

    System.out.println("(S" + size + ") empty");
    int index = size - 1;
    while(currentTier != null) {
      System.out.print("(S" + index + ") ");

      //while currentNode does not equal POS_INF placeholder
      while(currentNode.getKey() != 5000) {
	//if placeholder node, iterate to next node in sequence
	if(currentNode.getValue() == null) {
	  currentNode = currentNode.getNext();
	}
	else {
	  System.out.printf("%04d:%s ", currentNode.getKey(), currentNode.getValue());
          currentNode = currentNode.getNext();
	}
      }//end of inner while loop

      currentTier = currentTier.getBelow();
      currentNode = currentTier;
      System.out.println();
      index--;

    } //end of outer while loop
  }

  
  /*Prints the entire bottom list */
  public void printBottom() {
    if(this.isEmpty()) System.out.println("none");
    else {
      QuadNode current = NEG_INFbottom;
    
      //while current does not equal POS_INF placeholder
      while(current.getKey() != 5000) {
        //if current equals the NEG_INF placeholder, iterate to next node
        if(current.getValue() == null) current = current.getNext();
        else {
          System.out.printf("%04d:%s ", current.getKey(), current.getValue());
          current = current.getNext();
        }
      }

    }

    System.out.println();
  }

  /* Method that prints all events in between a
     two given keys */
  public void subMap(int key1, int key2) {

    //special case: if the list is empty
    if(this.isEmpty()) System.out.println("none");

    //special case: if key1 is greater than key2
    else if(key1 > key2) System.out.println("none");

    else {
      QuadNode current = ceilingEntry(key1);
      int index = 0;
      while(current != null && current.getKey() <= key2) {
	System.out.printf("%04d:%s ", current.getKey(), current.getValue());
        current = current.getNext();
        index++;
      } //end of while loop

      //if index equals zero, that means the while loop never executed, so no nodes were found
      if(index == 0) { System.out.println("none"); return; }
      System.out.println();
    } //end of else statement

  } //end of method


}
