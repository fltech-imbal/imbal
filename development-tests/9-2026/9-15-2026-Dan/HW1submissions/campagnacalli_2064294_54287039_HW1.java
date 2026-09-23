import java.util.Scanner;
import java.io.File; 
import java.io.FileNotFoundException;
/*
Author: Calli Campagna 
Email:ccampagna2025@code01.fit.edu
Course:CSE2010
Section:1-4
Description of this file: 
This program simulates an online customer chat service using singly linked lists.
It maintains lists of available representatives, customers waiting on hold,
and active chat sessions. Customer requests are processed in the order they
are received, and representatives are assigned based on their availability.
*/
public class HW1{

/* the methods below each perform an operation on one of the linked list. The methods add, remove, search, and print using parameters to
identify the list or information being added 
*/
  /* This class creates a node in a single linked list storing data and a reference to the next node. Then it sets the next data to
 null wehn its first created 
parameter: data- the infroamtion that will be stored in the node
*/
    class Node {

        String data;
        Node next;

        public Node(String data) {
            this.data = data;
            this.next = null;
        }
    }

/* This class creates a node in a single linked list storing the customer and representative names  and a reference to the next node. 
Then it sets the next data to  null wehn its first created 
parameter: customer- the infroamtion that will be stored in the customer node which is the cutomers name
representative- the infroamtion staored in the representative ode which is the representatives name
*/
   class chatNode{
       String customer;
       String representative;
       chatNode next;
        public chatNode(String customer, String representative){
            this.customer = customer;
            this.representative = representative;
            this.next = null;
        }
   }
/* This class creates a node in a single linked list storing the customer and requestTime nodes and a reference to the next node. 
Then it sets the next data to  null wehn its first created 
parameter: cutomer -the infroamtion that will be stored in the customer node which is the cutomers name
requestTime - the infroamtion that will be stored in the requesTime node which is the time the cutomer reuqested a chat 
*/
   class custNode{
       String customer;
       int requestTime;
       custNode next;
        public custNode(String customer, int requestTime){
            this.customer = customer;
            this.requestTime = requestTime;
            this.next = null;
        }
   }
/*
 Variebales representing the head of each linked list so the methods can acces the lists 
*/ 
   Node rHead;
   chatNode cHead;
   custNode sHead;
   int maxWaitTime = 0;
/* 
Checks whether a representative is available for  a customer request. If a representative is available, the customer is assigned to the
 first representative in the list, a chat session is created, and the representative is removed from the available representative list.
 If no representative is available, the customer is either placed on hold or told to try later.
Parameters:decision - indicates whether the customer wants to wait or try later
name - the name of the customer making the request
requestTime - the time the customer made the request
*/
     public  void representativeList(String decision, String name, int requestTime) {
            
            if (rHead != null){
                String representative = rHead.data;
                System.out.println("RepAssignment " + name + " " + representative + " " + String.format("%04d",requestTime));
                int waitTime = requestTime - requestTime; 
                if (waitTime > maxWaitTime){
                    maxWaitTime = waitTime;
                }
                chatSessionList(name,representative);
                removeRepList(representative);
            }else{
                if (decision.equals("wait")){
                    customerHoldList(name, requestTime);
                    System.out.println("PutOnHold " + name + " " + String.format("%04d",requestTime));
                }else{
                    tryLater(name, requestTime);
                }
            }            
            
   }
/* 
Removes a representative from the available representative linked list.The method checks the first node in the list and removes it if
 its data matches the given representative's name.
Parameter: name - the name of the representative to be removed
*/
   public void removeRepList(String name){
       Node current = rHead;
       if (current != null && current.data.equals(name)){
           rHead= current.next;
           return;
        }
    }
/* 
Adds a representative to the available representative linked list. A new node is created containing the representative's name. If the list
is empty, the new node becomes the first node in the list.
Parameter:name - the name of the representative being added
*/
   public void addRepList(String name){
        
        Node newNode = new Node(name);
        if (rHead == null) {
                rHead = newNode;
                return;
            }
        Node current = rHead;
        while (current.next != null) {
                current = current.next;
            }
        
        current.next = newNode;
           
       }

/*
Adds a new customer chat session to the end of the chat session linked list. The new node stores the customer's name and the 
representative assigned to the customer. If the list is empty, the new node becomes the first node.
Otherwise, the method travels to the end of the list and adds the new node.
Parameters:customer - the name of the customer in the chat session
representative - the name of the representative assigned to the customer
*/   
   public void chatSessionList(String customer, String representative ){
       chatNode newNode = new chatNode(customer, representative);
       if (cHead == null) {
                cHead = newNode;
                return;
            }

       chatNode current = cHead;
        while (current.next != null) {
                current = current.next;
            }

        current.next = newNode;
   }
/*
Removes a customer's chat session from the chat session linked list. When the customer is found, the representative assigned to that 
customer is added back to the available representative list. The method checks both the first node and the remaining nodes in the list.
Parameter: customer - the name of the customer that the chat session should be removed
*/
   public void removeChatSession(String customer){
       chatNode current = cHead;
       if (current != null && current.customer.equals(customer)){
           addRepList(current.representative);
           cHead = current.next;
           return;
       }
       while (current != null && current.next != null ) {
                if (current.next.customer.equals(customer)){
                    addRepList(current.next.representative);
                    current.next= current.next.next;
                    return;
                
                }
                current = current.next;
            }
   }

/*
Adds a customer who has chosen to wait to the customer hold linked list. The new node stores the customer's name and request time.
If the hold list is empty, the new customer becomes the first node. Otherwise, the method travels to the end of the list
and adds the customer there.
Parameters: customer - the name of the customer being placed on hold
requestTime - the time the customer made the chat request
*/
        public void customerHoldList(String customer, int requestTime) {
            custNode newNode = new custNode(customer,requestTime);

            if (sHead == null) {
                sHead = newNode;
                return;
            }

            custNode current = sHead;
            while (current.next != null) {
                current = current.next;
            }

            current.next = newNode;
            
   }
/*
Assigns a representative to the first customer in the hold list when a representative becomes available. The method calculates 
the customer's wait time, updates the maximum wait time if necessary, removes the customer from the hold list, removes the
representative from the available list, and creates a new chat session.
Parameter: assignmentTime - the time the customer is assigned to a representative
*/
   public void holdTimeCustomer(int assignmentTime){
       
       
       if (sHead == null){
           return;
       }
       if (rHead == null){
           return;
       }
       String customer = sHead.customer;
       int requestTime = sHead.requestTime;
       String representative = rHead.data;
       
       int waitTime = assignmentTime - requestTime;
       if (waitTime > maxWaitTime) {
           maxWaitTime = waitTime;
       }
       System.out.println("RepAssignment " + customer + " " + representative + " " + String.format("%04d",assignmentTime));
       
       sHead = sHead.next;
       
       removeRepList(representative);
       
       chatSessionList(customer, representative);
   }
/*
Removes a customer from the customer hold linked list. The method searches the list for the specified customer and removes the customer
when found. It returns the customer's original request time so that it can be used when calculating the customer's wait time.
Parameter:customer - the name of the customer to be removed from the hold list
Returns:The request time of the removed customer. Returns -1 if the customer is not found in the hold list.
*/
   public int removeHoldCustomer(String customer){
       custNode current = sHead;
       if (current != null && current.customer.equals(customer)) {
        int requestTime = current.requestTime;
        sHead = current.next;
        return requestTime;
       }
       while (current != null && current.next != null ) {
                if (current.next.customer.equals(customer)){
                    int requestTime = current.next.requestTime;
                    current.next = current.next.next;
                    return requestTime;
                
                }
                current = current.next;
            }
       return -1;
   }
/*
Prints the names of all representatives currently available. The method printing each representative's name in order along
with the time the list was requested.
Parameter: printTime - the time when the available representative list is printed
*/
   public void PrintAvailableRepList(int printTime){
       Node current = rHead;
       System.out.print("AvailableRepList " + String.format("%04d",printTime));
       while(current != null){
           System.out.print(" " + current.data);
           current = current.next;
       }
       System.out.println();
   }
/* 
Prints a message indicating that a customer has chosen not to wait for an available representative and will try again later.
Parameters:name - the name of the customer who will try again later
requestTime - the time the customer made the request
*/
   public void tryLater(String name, int requestTime){
       System.out.println("TryLater " + name + " " + String.format("%04d",requestTime));
   }
    
/*
Prints the information for a new customer chat request, including the request time, customer's name, and whether the customer wants
to wait or try again later.
Parameters:requestTime - the time the customer made the request
customer - the name of the customer making the request
waitOrLater - indicates whether the customer wants to wait or try later
*/
   public void chatRequest(int requestTime, String customer, String waitOrLater){
    
       System.out.println("ChatRequest " +  String.format("%04d",requestTime) + " " + customer 
        + " "+ waitOrLater);
            
     }
/*
Prints a message indicating that a customer has stopped waiting while on hold.
Parameters:quitOnHoldTime - the time the customer stopped waiting
customer - the name of the customer who stopped waiting
*/
   public void QuitOnHold(int quitOnHoldTime, String customer) {
          System.out.println("QuitOnHold " + String.format("%04d",quitOnHoldTime) + " " + customer);
     }
/*
Prints a message indicating that a customer's chat session has ended. The message includes the customer's name, representative's name,
and the time the chat ended.
Parameters: customer - the name of the customer whose chat ended
rep - the name of the representative who handled the chat
endTime - the time the chat session ended
*/
   public void ChatEnded(String customer, String rep, int endTime) {
       System.out.println("ChatEnded " + customer + " " + rep + " " + String.format("%04d",endTime));
            
    }
/*
Prints the maximum amount of time that a customer waited for a representative. The output includes the time the maximum wait was
requested and the maximum wait time.
Parameters: printTime - the time when the maximum wait time is printed
waitTime - the maximum amount of time a customer waited
*/
   public void PrintMaxWaitTime(int printTime, int waitTime) {
       System.out.println("MaxWaitTime " + String.format("%04d",printTime) + " " + String.format("%04d",waitTime)); 
   }
/*
The main method creates the initial linked lists, reads the input file, and processes each command in the file. It handles 
customer chat requests,customers leaving the hold list, and completed chat sessions.
Parameter: args - contains the command-line arguments, including the name of the input file used by the program.
*/ 
   public static void main(String[] args)throws FileNotFoundException  {
/*
homework - creates an instance of the HW1 class so its methods and linked-list variables can be used.
alice, bob, carol, david, emily - nodes representing the five representatives who are initially available.
*/
    HW1 homework = new HW1();
    Node alice = homework.new Node("Alice");
    Node bob = homework.new Node("Bob");
    Node carol = homework.new Node("Carol");
    Node david = homework.new Node("David");
    Node emily = homework.new Node("Emily");

/*
This block creates the five representative nodes and connects them in the required order. Alice is first, followed by Bob, Carol, David,
 and Emily. The representative list head is then set to Alice, making Alice the first available representative.
*/


    alice.next = bob;
    bob.next = carol;
    carol.next = david;
    david.next = emily;
    homework.rHead = alice;


/* 
sc - Scanner used to read commands and information from the input file.
condition - stores the current command being read from the input file.
requestTime - stores the time a customer makes a chat request.
customer - stores the name of the customer.
waitOrLater - stores whether the customer wants to wait or try later.
quitOnHoldTime - stores the time a customer quits waiting.
requestTime - stores the original request time of a customer on hold.
waitTime - stores the amount of time a customer waited.
*/
    

/*
This block creates a Scanner using the input file provided through the command line. If the command is ChatRequest, the request time, 
customer name, and wait decision are read. The information is then printed and the customer is assigned a representative,
 placed on hold,or told to try later.
*/

 
    Scanner sc = new Scanner(new File(args[0]));
        while(sc.hasNext()){
            String condition = sc.next();
            if (condition.equals("ChatRequest")){
                int requestTime = sc.nextInt();
                String customer = sc.next();
                String waitOrLater = sc.next();
                homework.chatRequest(requestTime, customer, waitOrLater);
                homework.representativeList(waitOrLater, customer,requestTime);
           
/*
This block handles a customer who chooses to quit while on hold.The customer is removed from the hold list and the original request time is returned.
If the customer was found, the wait time is calculated and compared with the current maximum wait time.
*/

              }else if(condition.equals("QuitOnHold")){
                int quitOnHoldTime = sc.nextInt();
                String customer = sc.next();
                int requestTime = homework.removeHoldCustomer(customer);
                if (requestTime !=-1){
                    int waitTime = quitOnHoldTime - requestTime;
                    if (waitTime > homework.maxWaitTime){
                        homework.maxWaitTime = waitTime;
                    }
                }
                homework.QuitOnHold( quitOnHoldTime,  customer);

/*
This block handles a customer whose chat has ended. It reads the customer, representative, and ending time from the input
file and processes the completed chat session.
*/


            }else if ( condition.equals("ChatEnded")){
                String customer = sc.next();
                String rep = sc.next();
                int endTime = sc.nextInt();
                homework.removeChatSession(customer);
                homework.ChatEnded(customer, rep, endTime);
                if(homework.sHead != null){
                 homework.holdTimeCustomer(endTime); 
                }

/* This block prints the list of representatives who are currently available. The printTime is the parameter for PrintAvailableRepList
 which is read from the input file. Then The print time is read from the input file, and the current maximum
The print time is read from the input file, and the current maximum
*/
            }else if(condition.equals("PrintAvailableRepList")){
                int printTime = sc.nextInt();
                homework.PrintAvailableRepList(printTime);
            }else if(condition.equals("PrintMaxWaitTime")){
            int printTime = sc.nextInt();
             
          homework.PrintMaxWaitTime(printTime, homework.maxWaitTime);
        }
        
    }
     }
}



     



 

