/*

  Author:Dayton Milanes
  Email: dmilanes2019@my.fit.edu
  Course: CSE 2010
  Section: 01
  Description of this file:

 */
import java.io.File; // File scanner, and FileNotFoundException are necessary imports
import java.util.Scanner;
import java.io.FileNotFoundException;

class customer { // customer is a custom object which contains a name and a time variable to keep track of these things for each customer
    String name;
    int time; // The time the customer started waiting
    
    customer(String n, int t){
        name = n;
        time = t;
    }
    
    String getName(){ //getter methods are needed to ensure access to these variables
        return name;
    }
    
    int getTime(){
        return time;
    }
}

class session { // session object. Similar to customer, but contains two strings instead of a string and an int
    String customer;
    String representative;
    
    session(String c, String r){
        customer = c;
        representative = r;
    }
    
    String getRep(){
        return representative;
    }
}

public class HW1
{
    public static void main(String[] args)
    {
        SinglyLinkedList<String> representatives = new SinglyLinkedList<>(); //set up the linked list of representatives that are *free* to take customers
        representatives.addLast("Alice");
        representatives.addLast("Bob");
        representatives.addLast("Carol");
        representatives.addLast("David");
        representatives.addLast("Emily");
        
        SinglyLinkedList<customer> customers = new SinglyLinkedList<>(); //a linked list of customers who will be handled on a first come first served basis
        SinglyLinkedList<session> sessions = new SinglyLinkedList<>(); //list of active sessions, who will take people from the representative and customer lists
        
            try{
                Scanner log = new Scanner(new File(args[0])); // scanner to scan for inputs
                
                String event = log.next();  //read from the input document to determine the next event and what to do
                
                String name; // variables to track information for each event are declared ahead of time
                String status;
                String rep; 
                int time;
                int maxTime = 0; // This tracks the highest wait time, and starts at 0
                
                customer previous;
        
                while(log.hasNextLine()){ // while loop runs while the input has not been fully read
                    switch (event){ // switch case based on the event variable determines which code executes depending on what happens in the document
                        case "ChatRequest":
                            time = log.nextInt();
                            name = log.next();
                            status = log.next(); // log the time, name, and status of the customer making the request
                            System.out.println("Chat Request " + time + " " + name + " " + status);
                            if(representatives.isEmpty() && status.equals("wait")){ // the customer wants to wait and there is no one to take them, so they will be added to the list of waiting customers
                                customers.addLast(new customer(name, time)); // add them to the back of the line and record the time they joined the line
                                System.out.println("Put on Hold " + name + " " + time);
                            } else if(status.equals("wait")) { // the customer is here and there is a representative for them
                                rep = representatives.removeFirst(); // get the first available representative
                                sessions.addLast(new session(name, rep));
                                System.out.println("Representative Assignment " + name + " " + rep + " " + time);
                            } else {
                                System.out.println("Try Later " + name + " " + time); // if they are coming back later there is no need to make an object
                            }
                            break;
                        case "QuitOnHold":
                            time = log.nextInt();
                            name = log.next();
                            System.out.println("Quit On Hold " + time + " " + name);
                            previous = customers.first();
                            for(int i = 0; i < customers.size(); i++){
                                if(previous.getNext().getName().equals(name)){
                                    previous.setNext(previous.getNext().getNext());
                                }
                            }
                            if ((time % 100) - ((customers.first().getTime()) % 100) < 0){
                                    time = time - 40;
                                }
                            if((time - customers.first().getTime()) > maxTime) maxTime = time - customers.first().getTime();
                            break;
                        case "ChatEnded": 
                            name = log.next();
                            rep = log.next();
                            time = log.nextInt(); //log info
                            System.out.println("Chat Ends " + name + " " + rep + " " + time);
                            rep = sessions.removeFirst().getRep(); //remove the session from the list of active sessions
                            if(customers.isEmpty()) representatives.addLast(rep); // No customers means the representative can get into the line of free representatives
                            else {
                                System.out.println("Representative Assignment " + name + " " + rep + " " + time); // if a customer is waiting the representative will be autmatically assigned to them
                                if ((time % 100) - ((customers.first().getTime()) % 100) < 0){
                                    time = time - 40; // subtracting 40 from time allows direct subtraction without errors, but is only necessary if any substitution is necessaru
                                }
                                if((time - customers.first().getTime()) > maxTime) maxTime = time - customers.first().getTime(); // if the difference between the customer's starting time and their current time, their wait, is larger than the maxTime they have been waiting the longest for assistance
                                name = customers.removeFirst().getName();
                                sessions.addLast(new session(name, rep)); // same process as pairing a customer and representative from before
                            }
                            break;
                        case "PrintAvailableRepList":
                            time = log.nextInt();
                            System.out.print("Available Representatives " + time + " " + representatives.toString()); // toString makes this part trivial (thanks Goodrich, Tamassia, and Goldwasser!)
                            break;
                        case "PrintMaxWaitTime":
                            time = log.nextInt();
                            System.out.print("Maximum wait time " + time + " " + maxTime); //print maxTime
                            break;
                    }
                }
            log.close();
        } catch (FileNotFoundException e){
            System.out.println("File not found: " + args[0]);
            System.exit(1); // catch the error incase the file is not found. This is necessary for compilation.
        }
    }
}
