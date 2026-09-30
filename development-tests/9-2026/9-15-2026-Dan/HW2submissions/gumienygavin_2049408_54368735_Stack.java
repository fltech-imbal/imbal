/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
  This file implements a stack using a singly linked list.

 */

public class Stack<E> {
  //---------------- nested Node class ----------------
  /**
   * Node of a singly linked list, which stores a reference to its
   * element and to the subsequent node in the list (or null if this
   * is the last node).
   */
  private static class Node<E> {

    // The element stored at this node
    private E element;

    // A reference to the subsequent node in the stack
    private Node<E> next;   
    
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

    /**
     * Sets the node's next reference to point to Node n.
     * @param n    the node that should follow this one
     */
    public void setNext(Node<E> n) { next = n; }
  } //----------- end of nested Node class -----------

  // instance variables of the stack
  // The top node of the stack
  private Node<E> top = null; 
  
  // Number of nodes in the stack
  private int size = 0;           

  // Constructs an initially empty stack.
  public Stack() { }              

  /**
   * Returns the number of elements in the stack.
   * @return number of elements in the stack
   */
  public int size() { return size; }

  /**
   * Tests whether the stack is empty.
   * @return true if the stack is empty, false otherwise
   */
  public boolean isEmpty() { return size == 0; }

  /**
   * Inserts an element to the top of the stack.
   * @param e  the new element to be inserted
   */
  public void push(E e) {         // adds element e to the top of the stack
    top = new Node<>(e, top);     // create and link a new node
    size++;
  }
  
  /**
   * Returns, but does not remove, the element at the top of the stack.
   * @return element at the top of the stack (or null if empty)
   */
  public E top() {        // returns (but does not remove) the first element
    if (isEmpty()) return null;
    return top.getElement();
  }

  /**
   * Removes and returns the top element from the stack.
   * @return the removed element (or null if empty)
   */
  public E pop() {                   // removes and returns the top element
    if (isEmpty()) return null;           // nothing to remove
    E answer = top.getElement();
    top = top.getNext();       // will become null if the stack had only one node
    size--;
    return answer;
  }
}