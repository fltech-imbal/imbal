
/*
Author: Hannah Clarke
Email: hclarke2025@my.fit.edu
Course: CSE2010
Section: 01
Description of this file: This program simulates a customer chatting with a customer service representative. The customer can 
choose to be put on hold, try again later, or quit while on hold. Customers are taken off hold in the order of their call
and representatives are chosen based on the order they become available.
*/
public class HW1
{
    //customer waiting list
    static Node customersOnHold;
    //completed chats
    static Node chatSessions;
    //available reps list
    static Node availableReps;
    
    
    //creates a generic node
    public static class Node {
        String customer;
        String rep;
        int time;
        Node next;
     
         //initializes generic node   
         Node(String customer, String rep, int time) {
             this.customer = customer;
             this.rep = rep;
             this.time = time;
             this.next = null;
         }
    }
   
    public void ChatRequest(int requestTime, String customer, String waitOrLater) {
        //if there is a rep, it'll be assigned to customer, if not, they can wait or quit
        System.out.println("ChatRequest " + requestTime + " " + customer + " " + waitOrLater);
        
        if (availableReps != null) {
            //rep is available
            Node repNode = availableReps;
            String rep = repNode.rep;
            availableReps = availableReps.next;
            
            Node newChatSession = new Node (customer, rep, requestTime);
            newChatSession.next = chatSessions;
            chatSessions = newChatSession;
            
            System.out.println("RepAssignment " + customer + " " + rep + " " + requestTime);
        } else {
            //rep is not available
            
            if (waitOrLater.equals("wait")) {
                
                Node newCustomer = new Node(customer, null, requestTime);
                
                //put on hold
                if (customersOnHold == null) {
                    customersOnHold = newCustomer;
                } else {
                    Node current = customersOnHold;
                    
                    while (current.next != null) {
                        current = current.next;
                    }
                    current.next = newCustomer;
                }
                
                System.out.println("PutOnHold " + customer + " " + requestTime);
            } else if (waitOrLater.equals("later")) {
                
                System.out.println("TryLater " + customer + " " + requestTime);
            }
        }
        
    }

    public void QuitOnHold(int quitOnHoldTime, String customer) {
        System.out.println("QuitOnHold " + quitOnHoldTime + " " + customer);
        
        Node current = customersOnHold;
        Node previous = null;
        
        //find customer
        while (current != null) {
            if (current.customer.equals(customer)) {
                break;
            }
            previous = current;
            current = current.next;
        }
        
        //if customer is not in list
        if (current == null) {
            return;
        }
        
        //customer in list, remove customer from hold list
        if (previous == null) {
            customersOnHold = current.next;
        } else {
            previous.next = current.next;
        }
    }

    public void ChatEnded(String customer, String rep, int endTime){
        System.out.println("ChatEnded " + customer + " " + rep + " " + endTime);
        
        Node current = chatSessions;
        Node previous = null;
        
        //find chat session in list
        while (current != null) {
            if (current.customer.equals(customer) && current.rep.equals(rep)) {
                break;
            }
            previous = current;
            current = current.next;
        }
        
        //chat session not found
        if (current == null) {
            return;
        }
        
        //remove found session from list
        if (previous == null) {
            chatSessions = current.next;
        } else {
            previous.next = current.next;
        }
        
        //assign next customer on hold to the freed rep
        if (customersOnHold != null) {
            Node nextCustomer = customersOnHold;
            
            //remove customer from hold
            customersOnHold = customersOnHold.next;
            
            //new chat session
            Node newChatSession = new Node (nextCustomer.customer, rep, endTime);
            
            newChatSession.next = chatSessions;
            chatSessions = newChatSession;
            
            System.out.println("RepAssignment " + nextCustomer.customer + " " + rep + " " + endTime);
        } else {
            //no customers on hold, add rep to available reps list
            Node newRep = new Node(null, rep, endTime);
            
            if (availableReps == null) {
                availableReps = newRep;
            } else {
                Node last = availableReps;
                
                while (last.next != null) {
                    last = last.next;
                }
                
            }
        }
    }

    public void PrintAvailableRepList(int printTime){
        
    }
    
    public void PrintMaxWaitTime(int printTime){
        
    }


    public static void main(String[] args)
    {
    //initialize customer waiting list
    customersOnHold = null;
    //initialize completed chats
    chatSessions = null;
    //available reps list
    availableReps = new Node(null, "Alice", 0);
    availableReps.next = new Node (null, "Bob", 0);
    availableReps.next.next = new Node (null, "Carol", 0);
    availableReps.next.next.next = new Node (null, "David", 0);
    availableReps.next.next.next.next = new Node (null, "Emily", 0);
    
    /* description of each block (around 5-10 lines) of instructions */
    }
}
