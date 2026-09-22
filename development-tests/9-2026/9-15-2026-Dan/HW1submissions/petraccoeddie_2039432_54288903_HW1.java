
/**
 Eddie Petracco III
 epetracco2025@my.fit.edu
 CSE 2010
 Section 3
 Simulates an online service that allows chatting on a website with certain representatives. 
 This includes requesting calls, ending calls, going on hold, and assigning representatives.
 The SinglyLinkedList class was implemented into this file for simplicity.
 */

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
public class HW1<E>
{
    /* Nested class to simplify the creation of multiple LinkedLists and store other 
     needed operations */
    public static class SinglyLinkedList<E> {
        //NESTED NODE CLASS
        private static class Node<E> {
            private E element; // The element stored at this node
            private Node<E> next; // reference to next node in list
            
            // Creates a node with element e and n pointer to the next node
            public Node(E e, Node<E> n) {
                element = e;
                next = n;
            }
            
            // returns element stored at the node
            public E getElement() {
                return element;
            }
            
            // returns the node that follows this one or null if node is tail
            public Node<E> getNext() {
                return next;
            }
            
            // Sets the node's pointer to Node n
            public void setNext(Node<E> n) {
                next = n;
            }
        }
        private Node<E> head = null; // head node of the list, null if empty
        private Node<E> tail = null; // last node of the list
        private int size = 0; // number of nodes in the list
        
        //ACCESSOR METHODS
        // size accessor
        public int size() {
            return size;
        }
        
        // checks if list has any nodes
        public boolean isEmpty() {
            return size == 0;
        }
        
        // checks if a certain element is in the list and removes it if its there
        public E removeMember(E element) {
          Node<E> current = head;
          Node<E> prev = null;
          E data = current.getElement();
          while (current != null) {
              if ((current.getElement()).equals(element)) {
                  size--;
                  // these are all special cases
                  if (prev == null) {
                      head = current.getNext();
                  }
                  else {
                      prev.setNext(current.getNext());
                  }
                  if (current == tail) {
                      tail = prev;    
                  }
                  if (isEmpty()) {
                      head = null;
                      tail = null;
                  }
                  // returns the data removed 
                  return data;
              }
              // moves through the list one at a time
              prev = current;
              current = current.getNext();
              data = current.getElement();
          }
          return(null);
        }
        
        // removes two members from list but returns the requestTime for onHold method.
        public E removeTwoMembers(E element) {
          if (isEmpty()) {
              return null;
          }
          
          Node<E> current = head;
          Node<E> prev = null;
          E data = current.getElement(); // the customer is grabbed here for comparison
          while (current != null) {
              // searches for the customer in the list
              if ((current.getElement()).equals(element)) { 
                  Node<E> timeNode = current.getNext();
                  if (timeNode == null) {
                      return null;
                  }
                  E requestTime = timeNode.getElement(); // this is the requestTime that is needed for max wait time calculation later
                  size-=2;
                  
                  Node <E> nextAfterTwo = timeNode.getNext(); // this is because this method removes two members
                  
                  // special cases
                  if (prev == null) {
                      head = nextAfterTwo;
                  }
                  else {
                      prev.setNext(nextAfterTwo);
                  }
                  if (timeNode == tail || nextAfterTwo == null) {
                      tail = prev;
                  }
                  if (isEmpty()) {
                      head = null;
                      tail = null;
                  }
                  return requestTime;
              }
              // moving along in the list ensuring we skip count up two values instead of just one
              prev = current.getNext();
              if (prev == null) {
                  break;
              }
              current = prev.getNext();
          }
          return(null);
        }
        
        // accessor for the first element in the list
        public E first () {
            if (isEmpty()) {
                return null;
            }
            return head.getElement();
        }
        
        // accessor for the last element in the list
        public E last() {
            if (isEmpty()) {
                return null;
            }
            return tail.getElement();
        }
        
        // UPDATE METHODS 
        // method to add new nodes to the start of the list 
        public void addFirst(E e) {
            head = new Node<>(e, head); // creates and adds the first Node with element e as the head
            if (size == 0) {
                tail = head; //special case with one element in list
            }
            size++;
        }
        
        // method to add new nodes to the end of the list
        public void addLast(E e) {
            Node<E> newest = new Node<>(e, null);
            if (isEmpty()) {
                head = newest; // special case if list is empty
            }
            else {
                tail.setNext(newest); // new node after existing tail
            }
            tail = newest; //new node becomes tail now
            size++;
        }
        
        // method to remove the first element of the list and returns it
        public E removeFirst() {
            if (isEmpty()) {
                return null; // nothing to remove case
            }
            E answer = head.getElement();
            head = head.getNext(); // will become null if list had only one node
            size--;
            if (size == 0) {
                tail = null; // special case if the list is now empty 
            }
            return answer;
        }
        
        public String toString(E data) {
            return(String.valueOf(data));
        }
     }
    
    private static int maxWaitTime = 0;
    
    public static void ChatEnded(String customer, String rep, String endTime, SinglyLinkedList repsList, SinglyLinkedList chatSessions, SinglyLinkedList customersOnHold) {
        // if no one is on hold, end the chat session, print the output statement, add rep back to list
        if (customersOnHold.isEmpty()) {
            System.out.println("ChatEnded " + customer + " " + rep + " " + endTime); 
            repsList.addLast(chatSessions.removeMember(rep));
            }
        else {
            // actually keep the rep in chatsessions since their going to be there anyway. Simply remove the first customer from hold 
            System.out.println("ChatEnded " + customer + " " + rep + " " + endTime);
            chatSessions.addLast(chatSessions.removeMember(rep));
            String customerOffHold = customersOnHold.removeFirst().toString();
            System.out.println("RepAssignment " + customerOffHold + " " + chatSessions.last() + " " + endTime); // special case for output
            String intialTime = customersOnHold.removeFirst().toString();
            CalculateMaxWaitTime(intialTime, endTime);
        }
        }
    
    // Main method for assigning a rep to a chatsessions or putting a customer on hold.
    public static void ChatRequest(String requestTime, String customer, String waitOrLater, SinglyLinkedList repsList, SinglyLinkedList chatSessions, SinglyLinkedList customersOnHold) {
        System.out.println("ChatRequest " + requestTime + " " + customer + " " + waitOrLater);
        
        if (repsList.isEmpty()) {
            //go through wait or later option then put that customer on hold or call the try again later method
            if (waitOrLater.equals("wait")) {
                PutOnHold(customer, requestTime, customersOnHold);
            }
            else {
                TryLater(customer, requestTime);
            }
        }
        else {
            // Assign a rep to the customer and add them to current ChatSessions. Also call RepAssignment method to print successful assingment message.
            chatSessions.addLast(repsList.removeFirst());
            RepAssignment(customer, chatSessions, requestTime);
        }
    }
    
    // Method for ensuring a successful rep assingmment in the output. 
    public static void RepAssignment(String customer, SinglyLinkedList chatSessions, String requestTime) {
        System.out.println("RepAssignment " + customer + " " + chatSessions.last() + " " + requestTime);    
    }

    // Method for putting customers in the on hold list along with their request time for calculating maxwait time later when their chat is accepted eventually or they quit on hold
    public static void PutOnHold(String customer, String requestTime, SinglyLinkedList customersOnHold) {
        System.out.println("PutOnHold " + customer + " " + requestTime); 
        customersOnHold.addLast(customer); // customer added to ensure they can be taken off hold easier later
        customersOnHold.addLast(requestTime); // requestTime added after the customer to calculate max wait time later
    }
    
    // Method for outputting the customer opted to try again later
    public static void TryLater(String customer, String requestTime) {
        System.out.println("TryLater " + customer + " " + requestTime);    
    }
    
    public static void QuitOnHold(String quitOnHoldTime, String customer, SinglyLinkedList customersOnHold) {
        // if the customer quits on hold then calculate their max wait time and save that value
        System.out.println("QuitOnHold " + quitOnHoldTime + " " + customer);
        String requestTime = customersOnHold.removeTwoMembers(customer).toString();
        CalculateMaxWaitTime(requestTime, quitOnHoldTime);
    }
    
    // Method for printing the entire availible reps list. Only param is the current repsList.
    public static void PrintAvailableRepList(SinglyLinkedList repsList, String requestTime) {
        System.out.print("AvailableRepList " + requestTime + " ");
        if (!repsList.isEmpty()) {
            System.out.print(repsList.first() + " ");
            HW1.SinglyLinkedList.Node currentNext = repsList.head.getNext();
            while (currentNext != null) {
                System.out.print(currentNext.getElement() + " ");
                currentNext = currentNext.getNext();    
            } 
        }
        System.out.println();
    }
    
    // Mehtod for printing the time requested and the max amount of minutes waited
    public static void PrintMaxWaitTime(String requestTime) {
        String minutesFormat = FormatFixer(maxWaitTime);
        System.out.println("MaxWaitTime " + requestTime + " " + minutesFormat);  
    }
    
    // this method calculates the max wait time and ensures that the military time doesn't cause issues
    public static void CalculateMaxWaitTime(String startTime, String endTime) {
        int startHours = Integer.parseInt(startTime.substring(0,2));
        int startMinutes = Integer.parseInt(startTime.substring(2,4));
        int endHours = Integer.parseInt(endTime.substring(0,2));
        int endMinutes = Integer.parseInt(endTime.substring(2,4));
        
        int waitTime = ((endHours * 60) + endMinutes) - ((startHours * 60) + startMinutes);
        if (waitTime > maxWaitTime) {
            maxWaitTime = waitTime; // global assingment to max wait time 
        }
    }
    
    // This helps fix the format when printing the amount of minutes waited so theres a 0 in front
    public static String FormatFixer(int time) {
        return(String.format("%04d", time));
    }
    
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in); //intilizing scanner to take inputs 
        
        SinglyLinkedList repsList = new SinglyLinkedList(); //creating the linked list for the representatives
        repsList.addFirst("Alice");
        repsList.addLast("Bob");
        repsList.addLast("Carol");
        repsList.addLast("David");
        repsList.addLast("Emily");
        
        SinglyLinkedList customersOnHold = new SinglyLinkedList(); // creating the linked list for customers on hold
        
        SinglyLinkedList chatSessions = new SinglyLinkedList(); // creating the linked list for chatsessions
        
        String filePath = args[0];
        
        try (Scanner fileScanner = new Scanner(new File(filePath))) {
            while (fileScanner.hasNext()) {
                String keyWord = fileScanner.next();
                // Group of if statements for different key words that could be inputted. Scanner reads the data from the file inputted in this case
                // and saves respective data if needed.
                if (keyWord.equals("ChatEnded")) {
                    String customer = fileScanner.next();
                    String rep = fileScanner.next();
                    String endTime = fileScanner.next();
                    ChatEnded(customer, rep, endTime, repsList, chatSessions, customersOnHold);
                }
                else if (keyWord.equals("ChatRequest")) {
                    String requestTime = fileScanner.next();
                    String customer = fileScanner.next();
                    String waitOrHold = fileScanner.next();
                    ChatRequest(requestTime, customer, waitOrHold, repsList, chatSessions, customersOnHold); 
                }
                else if (keyWord.equals("QuitOnHold")) {
                    String quitTime = fileScanner.next();
                    String customer = fileScanner.next();
                    QuitOnHold(quitTime, customer, customersOnHold);    
                }
                else if (keyWord.equals("PrintAvailableRepList")) {
                    String requestTime = fileScanner.next();
                    PrintAvailableRepList(repsList, requestTime);
                }
                else if (keyWord.equals("PrintMaxWaitTime")) {
                    String requestTime = fileScanner.next();
                    PrintMaxWaitTime(requestTime);
                }
            }
        }
        catch (FileNotFoundException e) {
            System.err.println("File name: " + filePath + " was not found!");    
        }
    }
}