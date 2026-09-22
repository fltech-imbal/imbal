
/*
Author:Israel Caballero
Email: icaballeroes2025@my.fit.edu
Course: CSE2010-Algorithms and Data Structures
Section: 01
Description of this file: Program that simulates customers chatting with customer service representatives
The program uses singly linked lists to organize available reps, chat sessions, and on hold customers
*/
import java.util.Scanner;
import java.io.File;
public class HW1
{
//Blueprint for creating a Singly linked List
public static class SinglyLinkedList<E>{
    //Nested Node class creates references to the element and the next Node
    private static class Node<E>{
        private E element;
        private Node<E> next;
        public Node(E e, Node<E> n){
            element = e;
            next = n;
        }
        public E getElement() {return element;} //get current element
        public Node<E> getNext() {return next;} //get next Node
        public void setNext(Node<E> n) {next = n;} //Set next node
    }
    private Node<E> head = null;
    private Node<E> tail = null;
    private int size = 0;
    public int getSize(){return size;} //gets size of the list
    public SinglyLinkedList(){} //Constructs initially empty list

    public boolean isEmpty(){return size == 0;} //checks if the list is empty
    //returns the first Nodes element
    public E first(){
        if(isEmpty()) return null;
        return head.getElement();
    }
    // returns the last Nodes element
    public E last(){
        if(isEmpty()) return null;
        return tail.getElement();
    }
    //adds a node to the beginning of the list
    public void addFirst(E e){
        head = new Node<>(e,head);
        if(size == 0)
            tail = head;
        size++;
    }
    //adds a node to the end of the list
    public void addLast(E e){
        Node<E> newest = new Node<>(e, null);
        if(isEmpty())
            head = newest;
        else
            tail.setNext(newest);
        tail = newest;
        size++;
    } 
    //Removes a node from the list
    public void remove(int index){
        if(isEmpty()) return;
        //if you want to remove the first node
        if(index == 0){removeFirst();}
        //if you want to remove a node in the middle or the end
        else{
            Node<E> current = head;
            for(int i = 1; i < index; i++){
                current = current.getNext();
            }
            if(index == size -1){
                    current.setNext(null);
                    tail = current;
                }
            else{
            current.setNext(current.getNext().getNext());
            }
            size--;
        }
    }
    //removes the first node in the list
    public E removeFirst(){
        if(isEmpty()) return null;
        E answer = head.getElement();
        head = head.getNext();
        size--;
        if(size == 0)
            tail = null;
        return answer;
    }
    //gets the element at the nodes index
    public E get(int index){
        if(isEmpty()) return null;
        Node<E> current = head;
        for(int i= 0; i <= index-1; i++){
            current = current.getNext();
        }
        return current.getElement();
    }
} //end of SinglyLinkedList<E> class

//creates a chat session object that holds representative, customer, assignment time, and endtime
private static class ChatSession{
    private String representative; //rep name
    private String customer;// customer name
    private int assignmentTime;//time the chat started
    //private int endTime;// time the chat ended

    public ChatSession(String r, String c, int a){
        representative = r;
        customer = c;
        assignmentTime = a;
    }
    //public void setEndTime(int e){endTime = e;}

    public String getRepresentative(){return representative;}
    public String getCustomer(){return customer;}
    //public int getAssignmentTime(){return assignmentTime;}
    //public int getEndTime(){return endTime;}
}
//Creates an object for an on hold customer that holds the customer name and request time
private static class OnHoldCustomer{
    private String customer;
    private int requestTime;

    public OnHoldCustomer(String c, int r){
        customer = c;
        requestTime = r;
    }
    public String getCustomer(){return customer;} //gets customer name
    public int getRequestTime(){return requestTime;}//gets request time
}   
//Goes through the available reps list and turns their names into a string
public static String repListToString(SinglyLinkedList<String> list){
    if(list.isEmpty())return "";
    String string = "";
    for(int i = 0; i < list.getSize(); i++){
        String rep = list.get(i);
        string = string + " " + rep;
    }
    return string;
}
//finds the index of a node in chatSessions that has the same customer and representative name as the parameters
public static int findInChatSession(SinglyLinkedList<ChatSession> list, String customer, String representative){
    for(int i = 0; i < list.getSize(); i++){
        String currentCustomer = list.get(i).getCustomer(); //current customers name in the node
        String currentRep = list.get(i).getRepresentative(); //current reps name in the node

        if(currentCustomer.equals(customer) && currentRep.equals(representative)){
            return i;
        }
    }
    return -1;
}
//Finds the index of a node in customersOnHold that has the same customer name as the parameters
public static int findInCustomersOnHold(SinglyLinkedList<OnHoldCustomer> list, String customer){
    for(int i = 0; i < list.getSize(); i++){
            String currentCostumer = list.get(i).getCustomer(); // current customers name in the node
            if(currentCostumer.equals(customer)){
                return i;
            }
        }
        return -1;
}
//format the time into HH:MM format
public static String formatTime(int time){
    return String.format("%04d", time);
}
public static void main(String[] args) throws Exception
{
    //Scanner reads input from file
    Scanner input = new Scanner(new File(args[0]));
    int maxWaitTime = 0; // max time waited to get assigned to a rep or until they quit on hold

    //Creates 3 empty Singly Linked Lists for availableReps, chatSessions, and customersOnHold
    SinglyLinkedList<String> availableReps = new SinglyLinkedList<>();
    SinglyLinkedList<ChatSession> chatSessions = new SinglyLinkedList<>();
    SinglyLinkedList<OnHoldCustomer> customersOnHold = new SinglyLinkedList<>();

    //Adds the 5 initial available reps to the availableReps List
    availableReps.addFirst("Alice");
    availableReps.addLast("Bob");
    availableReps.addLast("Carol");
    availableReps.addLast("David");
    availableReps.addLast("Emily");

    //Reads each line of the given file until there are no lines left
    while (input.hasNextLine()){
        //Makes each line into a string and then breaks the line into words after any white space and assigns those words to tokens
        String line = input.nextLine();
        String[] token = line.split("\\s+");


        //checks if the first word is "ChatRequest"
        if(token[0].equals("ChatRequest")){
            int requestTime = Integer.parseInt(token[1]);//request time
            String name = token[2];//name of customer
            String waitOrLater = token[3]; //if the customer decides to wait or try later

            System.out.println(line);

            //If customer wants to wait or go later, and there are reps available
            if((waitOrLater.equals("wait") && availableReps.isEmpty() == false) || (waitOrLater.equals("later") && availableReps.isEmpty() == false)){
                //removes the rep from the available reps list and puts them in a chat session with the customer
                String rep = availableReps.removeFirst();
                ChatSession s1 = new ChatSession(rep, name, requestTime);
                chatSessions.addLast(s1);
                
                System.out.println("RepAssignment " + name  + " " + rep + " " + formatTime(requestTime));
            }
            //If customer wants to wait for a rep but there are no reps available
            else if(waitOrLater.equals("wait") && availableReps.isEmpty()){
                //Customer is put on hold at the end of the list to preserve first come first serve
                OnHoldCustomer h1 = new OnHoldCustomer(name, requestTime);
                customersOnHold.addLast(h1);
                System.out.println("PutOnHold " + name + " " + formatTime(requestTime));
            }
            //If customer does not want to wait and there are no reps availible
            else if(waitOrLater.equals("later") && availableReps.isEmpty()){
                System.out.println("TryLater " + name + " " + formatTime(requestTime));

            }

        }
        //checks if the first word is "PrintAvailableRepList"
        else if(token[0].equals("PrintAvailableRepList")){
            //prints the list of available representatives
            int requestTime = Integer.parseInt(token[1]); // request time
            System.out.println("AvailableRepList " + formatTime(requestTime) + repListToString(availableReps));
        }
        //checks if the first word is "ChatEnded"
        else if(token[0].equals("ChatEnded")){
            String customerName = token[1]; // customer name 
            String repName = token[2];// representative name
            int chatEndTime = Integer.parseInt(token[3]); // time the chat ended

            System.out.println(line);
            //if customer and the rep of the chat's names are found, remove the chat session they are in and add the rep back into available reps.
            if(findInChatSession(chatSessions, customerName, repName) != -1){
                chatSessions.remove(findInChatSession(chatSessions, customerName, repName));
                availableReps.addLast(repName);

            }
            //if there are available reps and there are customers on hold
            if((availableReps.isEmpty() == false) && (customersOnHold.isEmpty() == false)){
                //saves customer name,requestTime and representative
                String customer = customersOnHold.first().getCustomer(); //name of the customer on hold
                int requestTime = customersOnHold.first().getRequestTime(); //request time of when the customer first started to wait for a representative
                String rep = availableReps.removeFirst(); //representatives name, also removes them from list
                //Creates a chat session with the customer on hold and an available rep then removes the customer from the onHoldList
                ChatSession s1 = new ChatSession(rep, customer, chatEndTime);
                chatSessions.addLast(s1);
                customersOnHold.removeFirst();
                
                System.out.println("RepAssignment " + customer  + " " + rep + " " + formatTime(chatEndTime));
                //checks the wait time of the customer on hold and if its bigger than the previous max it is now the previous max
                if((chatEndTime - requestTime) > maxWaitTime){
                    maxWaitTime = (chatEndTime - requestTime);
                }
            }
        }
        //checks if the first word is "QuitOnHold"
        else if(token[0].equals("QuitOnHold")){
            int quitTime = Integer.parseInt(token[1]); //time the customer decided to quit being on hold
            String name = token[2]; //Customers name
            //if the onHoldlist is not empty, find the index the customer is at and remove them
            if(customersOnHold.isEmpty() == false){
                int customerIndex = findInCustomersOnHold(customersOnHold, name); //index the node the customer is at in the list.
                int originalRequestTime = customersOnHold.get(customerIndex).getRequestTime(); //the original time the customer decided to wait for a representative.
                customersOnHold.remove(findInCustomersOnHold(customersOnHold, name));
                //checks the wait time of the customer on hold and if its bigger than the previous max it is now the previous max
                if((quitTime-originalRequestTime) > maxWaitTime){
                    maxWaitTime = quitTime - originalRequestTime;
                }
            }
            System.out.println(line);
        }
        //If the first word is "PrintMaxWaitTime", print the max wait time.
        else if(token[0].equals("PrintMaxWaitTime")){
            int requestTime = Integer.parseInt(token[1]); //time the request was made
            System.out.println("MaxWaitTime " + formatTime(requestTime) + " " + formatTime(maxWaitTime) );
        }
    }
}
}

