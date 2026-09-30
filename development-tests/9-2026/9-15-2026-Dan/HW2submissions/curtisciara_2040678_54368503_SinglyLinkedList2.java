/*
  Author: Ciara Curtis
  Email: ccurtis2025@my.fit.edu
  Course: CSE2010
  Section: 4
  Description of this file: A data type that uses a private node class (that holds one string and one int variable,
                            and a reference to a second node) as links to create the entire list.
 */

public class SinglyLinkedList2 {

  //Private class that holds the information of each item added to the list
  private static class Node {

    private String element;    			   //reference to the String element stored at this node
    private Node next;         		           //reference to the subsequent node in the list

    public Node(String e, Node n) { 	   	   //Constructor for a new node
      element = e;                         
      next = n;
    }

    public String getElement() { return element; } //returns the node's string element
    public Node getNext() { return next; }         //returns the node's next Node in the list
    public void setNext(Node n) { next = n; }      //allows user to change the next node of this node.
  } //----------- end of nested Node class -----------

  private Node head = null;               	   // head node of the list (or null if empty)
  private Node tail = null;               	   // last node of the list (or null if empty)
  private int size = 0;                   	   // number of nodes in the list
  public SinglyLinkedList2() { }           	   // constructs an initially empty list

  //----------------------SinglyLinkedList "get" methods-------------------------
  public int size() { return size; } 		   //returns the size of list

  public String firstElement() {     		   // returns (but does not remove) the first Node's element
    if (isEmpty()) return null;                  
    return head.getElement();
  }

  public String lastElement() {             	   // returns (but does not remove) the last Node's element
    if (isEmpty()) return null;
    return tail.getElement();
  }

  public boolean isEmpty() { return size == 0; }

  //-----------------------SinglyLinkedList other Methods------------------------------------ 
  public void addFirst(String e) {                // adds element e to the front of the list
    head = new Node(e, head);                         // create and link a new node
    if (size == 0)
      tail = head;                                       // special case: new node becomes tail also
    size++;
  }

  public void addLast(String e) {                 // adds element e to the end of the list
    Node newest = new Node(e, null);                  // node will eventually be the tail
    if (isEmpty())
      head = newest;                                     // special case: previously empty list
    else
      tail.setNext(newest);                              // new node after existing tail
    tail = newest;                                       // new node becomes the tail
    size++;
  }

  //adds element to list while pre-emptively sorting
  public void addElement(String e) {
    if(isEmpty()) {					//special case: list is empty
	head = new Node(e, null);
	tail = head;
	size++;
    }
    else {
      
      if(head.getElement().compareTo(e) > -1) {		//special case: head is replaced
	head = new Node(e, head);
	size++;
        return;
      }
      
      Node current = head;
      Node previous = head.getNext();			 
      while(current != null) {
	if(current.getElement().compareTo(e) > -1) {    //special case: Node is placed in middle of list
	  Node newNode = new Node(e, current);
	  previous.setNext(newNode);
	  size++;
	  return;
	}
	previous = current;
	current = current.getNext();
      }

      if (current == null) {				//special case: Tail node is getting replaced
	Node newNode = new Node(e, null);
	previous.setNext(newNode);
	tail = newNode;
	size++;
      }

    }

  }


  public String removeFirst() {                          // removes and returns the first element
    if (isEmpty()) return null;                          // nothing to remove
    String answer = head.getElement();
    head = head.getNext();                               // will become null if list had only one node
    size--;
    if (size == 0)
      tail = null;                                       // special case as list is now empty
    return answer;
  }

  public String removeLast() {
    if(isEmpty()) return null;				//special case: list is empty
    else if(size == 1) {				//special case: head is removed
      String str = head.getElement();
      size--;
      head = null;
      tail = head;
      return str;
    }
    else {
      Node current = head;
      Node previous = null;
      while(current.getNext() != null) {
        previous = current;
        current = current.getNext();
      }
      String str = current.getElement();
      tail = previous;
      tail.setNext(null);
      size--;
      return str;
    }
  }

  public String removeNode(String e) {                   //returns and removes a specific node in the list
    Node current = head;
    Node previous = current;
    if (current.getElement().equals(e)) {                //special case: head is removed
      head = head.getNext();
      size--;
      if (size == 0) tail = null;
      return current.getElement();
    }
    else {                                               //used to search the list for the node
      while(current != null) {
        if (current.getElement().equals(e)) break;       //current Node holds the element to return so the loop is broken
        previous = current;
        current = current.getNext();
      }
    }
    
    if (current == null) return null;                    //special case: Node was never found
    else if(current.getNext() == null) {                 //special case: tail is removed
      previous.setNext(null);
      size--;
      if(size == 0) tail = null;
      else tail = previous;
      return current.getElement();
    }
    else {                                               //special case: node is in middle of list
      size--;
      if(size == 0) tail = null;                         
      previous.setNext(current.getNext());
      return current.getElement();
    }
  }

  @SuppressWarnings({"unchecked"})
  public boolean equals(Object o) {			 //Compares class objects
    if (o == null) return false;
    if (getClass() != o.getClass()) return false;
    SinglyLinkedList2 other = (SinglyLinkedList2) o;       // use nonparameterized type
    if (size != other.size) return false;
    Node walkA = head;                                   // traverse the primary list
    Node walkB = other.head;                             // traverse the secondary list
    while (walkA != null) {
      if (!walkA.getElement().equals(walkB.getElement())) return false; //mismatch
      walkA = walkA.getNext();
      walkB = walkB.getNext();
    }
    return true;                                         // if we reach this, everything matched successfully
  }

  /**
   * Produces a string representation of the contents of the list.
   * This exists for debugging purposes only.
   */
  public String toString() {
    StringBuilder sb = new StringBuilder("");
    Node walk = head;
    while (walk != null) {
      sb.append(walk.getElement());
      if (walk != tail)
        sb.append(" ");
      walk = walk.getNext();
    }
    sb.append("");
    return sb.toString();
  }

  public void toPrint() {
    Node current = head;
    while(current != null) {
      System.out.println(current.getElement());
      current = current.getNext();
    }
  }
}
