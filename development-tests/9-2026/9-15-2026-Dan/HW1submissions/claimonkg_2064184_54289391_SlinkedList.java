/*@author KG Claimon*/ 


public class SlinkedList<E> {
    public class Node { //makes a class a node
        public E element; 
        public Node next; 
        
        public Node(E element, Node next) { //makes the elements variables
            this.element = element; 
            this.next = next;
        } 

        public E getElement() { //returns one element
            return this.element; 

        } 

        public Node getNext() { //returns the next
            return this.next;

        } 

        public void setNext(Node next) { //This one is here in case a node needs to be set to the next node
            this.next = next;
        }

        public String toString() { //returns an element
            return "" + this.element;

        } 

    } 
    
    public Node head; 
    public Node current; 
    public Node tail; 
    public int size = 0; 
    public E nodeValue; 
    public Node previous; 

    public E key;

     
    public String toString() { //returns all the elements in the list
        current = head;
        String string = "";
        while(current != null) {
            string = string + " " + current.getElement();
            current = current.next;
        } 
        return string;
    }
    

    
    
    public Node createSlinkedList(E element) { //creates a list, but is not used in main code
        head = new Node(element, null);
        current = head;
        previous = null;
        nodeValue = element;
        tail = head;
        size = 1;
        return head;
    }

    public int size() {return size;} 

    public boolean isEmpty() {return size == 0;} 

    public E first() { //this one returns first
        if (isEmpty()) {return null;} 
        return head.element;
    } 

    public E last() { //this one returns last
        if(isEmpty()) {return null;} 
        return tail.element;
    } 

    public void addFirst(E element) { // adds first
        Node newNode = new Node(element, head);
        if (isEmpty()) {
            tail = newNode;
        } 
        head = newNode;  
        size++;
    }  

    public void addLast(E element) { //adds last
       
        Node newest = new Node(element, null); 
        if (isEmpty()) {
            head = newest; 
            tail = newest; 
            
        } else {
            tail.setNext(newest); 
            
        } 
        tail = newest; 
        size++;
        
    }

   
    

    public boolean search(E key) { //search function
       //checks for data
        if (head == null) {
            return false;
        } else {
            current = head; 
            
            while (current != null) {
                if(current.getElement().equals(key)) {
                    return true;
                } 
                current = current.getNext();
            }
           
        } 
        return false;
        
    } 

    public E removeFirst() { //a removeFirst function was created so I could remove the first item quickly
        
        if (head == null) {
            return null;
        } 

        E foundElement = head.getElement();
        head = head.next; 
        size--; 
        return foundElement;
    }


    public E remove(E element) { //remove function
        //checks for any data 
        
        if (head == null) {
            
            
            return null;
        } 
        
        //Case 1: Removing the Head 
        if (head.getElement().equals(element)) { 
            
            E foundElement = head.getElement();
            head = head.next; 
            size--;  
            if (size == 0) {
                tail = null;
            }
            return foundElement;
        } 

        //Case 2:removing the middle or tail
        
        previous = head;
        current = head.next; 
        

        while (current != null) {
            if(current.getElement().equals(element)) { 
                
                E foundCurrent = current.getElement(); 
                if (current == tail) {
                    tail = previous;
                } 
                
                previous.next = current.next; 
                size--; 
                return foundCurrent;
            } 
            previous = current; 
            current = current.next;
        }
        return null;
    }

    


}
