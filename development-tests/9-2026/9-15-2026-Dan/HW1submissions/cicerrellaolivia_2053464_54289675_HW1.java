
/**
 * Author:Olivia Cicerrella
 * Email: ocicerrella2025@fit.edu
 * Course: Algor and data Struc
 * Section: section 1
 * Decription:This program simulates a customer chat service using singly linked lists.
 */
import java.io.FileNotFoundException;
import java.io.File;
import java.util.Scanner;

public class HW1 {
    
    // objects to Store Inside the SinglyLinkedLists
// the class for reps this sores the name and returns it when called.
    static class Representative {
        String name;
        Representative(String name) { this.name = name; }
        @Override
        public String toString() { return name; }
    }

// the class for customers storing the needed data such as name and the time they started their request
    static class Customer {
        String name; // the name of the customer
        String requestTimeStr;
        int requestTimeInt;
        String decision;
        Customer(String name, String requestTimeStr, int requestTimeInt, String decision) {
            this.name = name;
            this.requestTimeStr = requestTimeStr;
            this.requestTimeInt = requestTimeInt;
            this.decision = decision;
        }
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            Customer other = (Customer) obj;
            return name.equals(other.name);
        }
    }
// the class for the active sessions 
    static class ActiveSession {
        String customerName; // the name of the customers 
        String repName; // the name of the rep so they can go with the customers
        ActiveSession(String customerName, String repName) {
            this.customerName = customerName;
            this.repName = repName;
        }
        @Override
        public boolean equals(Object obj) {
            if (this == obj) return true;
            if (obj == null || getClass() != obj.getClass()) return false;
            ActiveSession other = (ActiveSession) obj;
            return customerName.equals(other.customerName);
        }
    }

    // from the text book for the singlylinkedlsit class to help with what is needed
    private static SinglyLinkedList<Representative> availableReps = new SinglyLinkedList<>();
    private static SinglyLinkedList<Customer> holdQueue = new SinglyLinkedList<>();
    private static SinglyLinkedList<ActiveSession> activeSessions = new SinglyLinkedList<>();

    // the max wiat time that has been recorded so far being stored here 
    private static int maxWaitTimeSoFar = 0;

    // event buffers for when it is processing multiple evens at the same time
    private static int currentTimestamp = -1;
    private static SinglyLinkedList<String> endedBuffer = new SinglyLinkedList<>();
    private static SinglyLinkedList<String> requestBuffer = new SinglyLinkedList<>();
    private static SinglyLinkedList<String> assignmentBuffer = new SinglyLinkedList<>();
    private static SinglyLinkedList<String> holdBuffer = new SinglyLinkedList<>();
    private static SinglyLinkedList<String> tryLaterBuffer = new SinglyLinkedList<>();
    private static SinglyLinkedList<String> quitBuffer = new SinglyLinkedList<>();
    private static SinglyLinkedList<String> printRepBuffer = new SinglyLinkedList<>();
    private static SinglyLinkedList<String> printWaitBuffer = new SinglyLinkedList<>();

    //clear the events  
    private static void clearEvents() {
        while (!endedBuffer.isEmpty()) System.out.println(endedBuffer.removeFirst());
        while (!requestBuffer.isEmpty()) System.out.println(requestBuffer.removeFirst());
        while (!assignmentBuffer.isEmpty()) System.out.println(assignmentBuffer.removeFirst());
        while (!holdBuffer.isEmpty()) System.out.println(holdBuffer.removeFirst());
        while (!tryLaterBuffer.isEmpty()) System.out.println(tryLaterBuffer.removeFirst());
        while (!quitBuffer.isEmpty()) System.out.println(quitBuffer.removeFirst());
        while (!printRepBuffer.isEmpty()) System.out.println(printRepBuffer.removeFirst());
        while (!printWaitBuffer.isEmpty()) System.out.println(printWaitBuffer.removeFirst());
    }

    private static void checkTimestamp(int newTime) {
        if (currentTimestamp != newTime) {
            clearEvents();
            currentTimestamp = newTime;
        }
    }

    // this does the time convertion 
    private static int calculateMinutes(int timeHHMM) {
        return (timeHHMM / 100) * 60 + (timeHHMM % 100);
    }

    private static int calculateWaitTime(int startTimeInt, int endTimeInt) {
        return calculateMinutes(endTimeInt) - calculateMinutes(startTimeInt);
    }

    // main section/ loop part of the program
    public static void main(String[] args) {
	// ensure that a file is ran with the program
	// if not prints the error message if this is trhe case.
        if (args.length < 1) {
            System.out.println("Error: Provide the input file name as a command-line argument.");
            return;
        }

        // initialize the representatives 
        availableReps.addLast(new Representative("Alice"));
        availableReps.addLast(new Representative("Bob"));
        availableReps.addLast(new Representative("Carol"));
        availableReps.addLast(new Representative("David"));
        availableReps.addLast(new Representative("Emily"));

	// handles the file that is used to test the progrwm.
        try {
            File file = new File(args[0]);
            Scanner scanner = new Scanner(file);

		// reads the file lines
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) continue;

		// splits the line when it sees more spaces then should have
                String[] info = line.split("\\s+");
                String command = info[0];

                switch (command) {
	// the firt case when a costomer requests a chat
                    case "ChatRequest": {
                        String timeStr = info[1]; // for the time
                        int timeInt = Integer.parseInt(timeStr);
                        String custName = info[2]; //stores the name
                        String decision = info[3]; // get the wait or later section

                        checkTimestamp(timeInt);
                        requestBuffer.addLast("ChatRequest " + timeStr + " " + custName + " " + decision);

		// if the rep is free then talk
                        if (!availableReps.isEmpty()) {
                            Representative assignedRep = availableReps.removeFirst();
                            activeSessions.addLast(new ActiveSession(custName, assignedRep.name));
                            assignmentBuffer.addLast("RepAssignment " + custName + " " + assignedRep.name + " " + timeStr);
                        } else { // all the reps are busy 
                            if (decision.equals("wait")) {// if they wait on hold
                                holdQueue.addLast(new Customer(custName, timeStr, timeInt, decision));
                                holdBuffer.addLast("PutOnHold " + custName + " " + timeStr);
                            } else { // of they try later
                                tryLaterBuffer.addLast("TryLater " + custName + " " + timeStr);
                            }
                        }
                        break;
                    }
		// different case then they quit on hold 
                     case "QuitOnHold": {
                        String timeStr = info[1]; // stores time
                        int timeInt = Integer.parseInt(timeStr);
                        String custName = info[2]; // the name

                        checkTimestamp(timeInt);

		// goes to the list node to find the customer
                        Customer dummy = new Customer(custName, "", 0, "");
                        SinglyLinkedList.Node<Customer> walk = holdQueue.getHeadNode();
                        Customer target = null;
                        while (walk != null) {
                            if (walk.getElement().equals(dummy)) {
                                target = walk.getElement();
                                break;
                            }
                            walk = walk.getNext();// moves to next node in list
                        }

			// if the customer is found in the que removes them and tracks info
                        if (target != null) {
                            holdQueue.removeElement(target);
                            int wait = calculateWaitTime(target.requestTimeInt, timeInt);// gets how long they waited
                            if (wait > maxWaitTimeSoFar) {
                                maxWaitTimeSoFar = wait;
                            }
                            quitBuffer.addLast("QuitOnHold " + timeStr + " " + custName);
                        }
                        break;
                    }
		// the next case when the converstan or chat ends
                    case "ChatEnded": {
                        String custName = info[1];
                        String repName = info[2];
                        String timeStr = info[3];
                        int timeInt = Integer.parseInt(timeStr);

                        checkTimestamp(timeInt);
                        activeSessions.removeElement(new ActiveSession(custName, repName));
                        endedBuffer.addLast("ChatEnded " + custName + " " + repName + " " + timeStr);
			
			//checks to see if the que for customers is empty
                        if (!holdQueue.isEmpty()) {
                            Customer nextCustomer = holdQueue.removeFirst();// gets the person waited the longest
                            activeSessions.addLast(new ActiveSession(nextCustomer.name, repName));
                            
				// gets the wait time claculated
                            int wait = calculateWaitTime(nextCustomer.requestTimeInt, timeInt);
                            if (wait > maxWaitTimeSoFar) {
                                maxWaitTimeSoFar = wait;
                            }
                            assignmentBuffer.addLast("RepAssignment " + nextCustomer.name + " " + repName + " " + timeStr);
                        } else {
                            availableReps.addLast(new Representative(repName));
                        }
                        break;
                    }

		// the next case prints the rep avalabe 
                    case "PrintAvailableRepList": {
                        String timeStr = info[1];
                        int timeInt = Integer.parseInt(timeStr);
                        checkTimestamp(timeInt);

			// makes list of reps waiting
                        StringBuilder sb = new StringBuilder("AvailableRepList " + timeStr);
                        SinglyLinkedList.Node<Representative> walk = availableReps.getHeadNode();
                        while (walk != null) {
                            sb.append(" ").append(walk.getElement().name);
                            walk = walk.getNext();
                        }
                        printRepBuffer.addLast(sb.toString());
                        break;
                    }
		// next case is the max wiat time so far
                    case "PrintMaxWaitTime": {
                        String timeStr = info[1];
                        int timeInt = Integer.parseInt(timeStr);
                        checkTimestamp(timeInt);
                        printWaitBuffer.addLast("MaxWaitTime " + timeStr + " " + maxWaitTimeSoFar);
                        break;
                    }
                    default:
                        break;
                    }
                }
                clearEvents();
        }
        catch (FileNotFoundException e) {
               System.out.println("File not found: " + args[0]);
        }
    }
    

/*
 * Copyright 2014, Michael T. Goodrich, Roberto Tamassia, Michael H. Goldwasser
 *
 * Developed for use with the book:
 *
 *    Data Structures and Algorithms in Java, Sixth Edition
 *    Michael T. Goodrich, Roberto Tamassia, and Michael H. Goldwasser
 *    John Wiley & Sons, 2014
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
//package net.datastructures;

/**
 * A basic singly linked list implementation.
 *
 * @author Michael T. Goodrich
 * @author Roberto Tamassia
 * @author Michael H. Goldwasser
 */

  //---------------- nested Node class ----------------
  /**
   * Node of a singly linked list, which stores a reference to its
   * element and to the subsequent node in the list (or null if this
   * is the last node).
   */
public static class SinglyLinkedList<E> {  
  public static class Node<E> {

    /** The element stored at this node */
    private E element;            // reference to the element stored at this node

    /** A reference to the subsequent node in the list */
    private Node<E> next;         // reference to the subsequent node in the list

    /**
     * Creates a node with the given element and next node.
     *
     * @param e  the element to be stored
     * @param n  reference to a node that should follow the new node
     */
    public Node(E e, Node<E> n) {
      element = e;
      next = n;
    }

    // Accessor methods
    /**
     * Returns the element stored at the node.
     * @return the element stored at the node
     */
    public E getElement() { return element; }

    /**
     * Returns the node that follows this one (or null if no such node).
     * @return the following node
     */
    public Node<E> getNext() { return next; }

    // Modifier methods
    /**
     * Sets the node's next reference to point to Node n.
     * @param n    the node that should follow this one
     */
    public void setNext(Node<E> n) { next = n; }
  } //----------- end of nested Node class -----------

  // instance variables of the SinglyLinkedList
  /** The head node of the list */
  private Node<E> head = null;               // head node of the list (or null if empty)

  /** The last node of the list */
  private Node<E> tail = null;               // last node of the list (or null if empty)

  /** Number of nodes in the list */
  private int size = 0;                      // number of nodes in the list

  /** Constructs an initially empty list. */
  public SinglyLinkedList() { }              // constructs an initially empty list

  // access methods
  /**
   * Returns the number of elements in the linked list.
   * @return number of elements in the linked list
   */
  public int size() { return size; }

  /**
   * Tests whether the linked list is empty.
   * @return true if the linked list is empty, false otherwise
   */
  public boolean isEmpty() { return size == 0; }

  /**
   * Returns (but does not remove) the first element of the list
   * @return element at the front of the list (or null if empty)
   */
  public E first() {             // returns (but does not remove) the first element
    if (isEmpty()) return null;
    return head.getElement();
  }

  /**
   * Returns (but does not remove) the last element of the list.
   * @return element at the end of the list (or null if empty)
   */
  public E last() {              // returns (but does not remove) the last element
    if (isEmpty()) return null;
    return tail.getElement();
  }

  // update methods
  /**
   * Adds an element to the front of the list.
   * @param e  the new element to add
   */
  public void addFirst(E e) {                // adds element e to the front of the list
    head = new Node<>(e, head);              // create and link a new node
    if (size == 0)
      tail = head;                           // special case: new node becomes tail also
    size++;
  }

  /**
   * Adds an element to the end of the list.
   * @param e  the new element to add
   */
  public void addLast(E e) {                 // adds element e to the end of the list
    Node<E> newest = new Node<>(e, null);    // node will eventually be the tail
    if (isEmpty())
      head = newest;                         // special case: previously empty list
    else
      tail.setNext(newest);                  // new node after existing tail
    tail = newest;                           // new node becomes the tail
    size++;
  }

  /**
   * Removes and returns the first element of the list.
   * @return the removed element (or null if empty)
   */
  public E removeFirst() {                   // removes and returns the first element
    if (isEmpty()) return null;              // nothing to remove
    E answer = head.getElement();
    head = head.getNext();                   // will become null if list had only one node
    size--;
    if (size == 0)
      tail = null;                           // special case as list is now empty
    return answer;
  }

  @SuppressWarnings({"unchecked"})
  public boolean equals(Object o) {
    if (o == null) return false;
    if (getClass() != o.getClass()) return false;
    SinglyLinkedList other = (SinglyLinkedList) o;   // use nonparameterized type
    if (size != other.size) return false;
    Node walkA = head;                               // traverse the primary list
    Node walkB = other.head;                         // traverse the secondary list
    while (walkA != null) {
      if (!walkA.getElement().equals(walkB.getElement())) return false; //mismatch
      walkA = walkA.getNext();
      walkB = walkB.getNext();
    }
    return true;   // if we reach this, everything matched successfully
  }

  @SuppressWarnings({"unchecked"})
  public SinglyLinkedList<E> clone() throws CloneNotSupportedException {
    // always use inherited Object.clone() to create the initial copy
    SinglyLinkedList<E> other = (SinglyLinkedList<E>) super.clone(); // safe cast
    if (size > 0) {                    // we need independent chain of nodes
      other.head = new Node<>(head.getElement(), null);
      Node<E> walk = head.getNext();      // walk through remainder of original list
      Node<E> otherTail = other.head;     // remember most recently created node
      while (walk != null) {              // make a new node storing same element
        Node<E> newest = new Node<>(walk.getElement(), null);
        otherTail.setNext(newest);     // link previous node to this one
        otherTail = newest;
        walk = walk.getNext();
      }
    }
    return other;
  }

  public int hashCode() {
    int h = 0;
    for (Node walk=head; walk != null; walk = walk.getNext()) {
      h ^= walk.getElement().hashCode();      // bitwise exclusive-or with element's code
      h = (h << 5) | (h >>> 27);              // 5-bit cyclic shift of composite code
    }
    return h;
  }

  /**
   * Produces a string representation of the contents of the list.
   * This exists for debugging purposes only.
   */
  public String toString() {
    StringBuilder sb = new StringBuilder("(");
    Node<E> walk = head;
    while (walk != null) {
      sb.append(walk.getElement());
      if (walk != tail)
        sb.append(", ");
      walk = walk.getNext();
    }
    sb.append(")");
    return sb.toString();
  }
    public boolean removeElement(E e) {
             if (isEmpty()) return false;
             if (head.getElement().equals(e)) {
                 removeFirst();
                 return true;
                }
             return false;   
    }
    public Node<E> getHeadNode() { 
    return head; 
    }
}
}
