/* Richie Kowalski
   Rkowalski2023@my.fit.edu
   CSE 2010
   Take a list of clients, times and requests to process and produce
   the correct output with time signatures
   */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class HW1 {

    // Represents a node in a linked list. thats all
    // gets the client name, rep name, time and next node in the list
    //used for the
    static class Node {
        String name;
        String rep;
        int time;
        Node next;
    }

    // Stores one scan line.
    //request type (EndCall, ChatRequest, QuitOnHold, ChatEnded), time+time as int for minutes, customer name, rep name and choice
    static class Requests {
        String type;
        String time;
        int timeInt;
        String customer;
        String rep;
        String choice;
    }

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length != 1) {
            System.out.println("java HW1 scan.txt");
            return;
        }
        //representative list, hold list and chat list
        //rep list holds available representatives
        Node repHead = null;
        Node repTail = null;
        //hold list holds customers that are waiting for a representative
        Node onHoldHead = null;
        Node onHoldTail = null;
        //chat list holds customers that are currently in a chat with a representative
        Node chatHead = null;
        Node chatTail = null;
        //max wait time :/
        int maxWait = 0;


        // Create the initial representative list.
        String[] availReps = {"Alice", "Bob", "Carol", "David", "Emily"};
        for (int i = 0; i < availReps.length; i++) {
            Node newRep = new Node();
            newRep.name = availReps[i];
            if (repHead == null) {
                repHead = newRep;
                repTail = newRep;
            } else {
                repTail.next = newRep;
                repTail = newRep;
            }
        }

        // Read every request from the scan file and store to requests list.
        Scanner scan = new Scanner(new File(args[0]));
        ArrayList<Requests> requests = new ArrayList<Requests>();

        //the goal of this while loop is to read the scan file and store the request with the info needed for easy look up
        while (scan.hasNext()) {
            //used to keep track of the active requests in the scan file
            Requests currRequest = new Requests();
            currRequest.type = scan.next();

            //for chat request, we need to get the time, customer and choice
            if (currRequest.type.equals("ChatRequest")) {
                currRequest.time = scan.next();
                currRequest.customer = scan.next();
                currRequest.choice = scan.next();
                
            //quit on hold get the time and customer
            } else if (currRequest.type.equals("QuitOnHold")) {
                currRequest.time = scan.next();
                currRequest.customer = scan.next();

            //chat ended get the customer, rep and time
            } else if (currRequest.type.equals("ChatEnded")) {
                currRequest.customer = scan.next();
                currRequest.rep = scan.next();
                currRequest.time = scan.next();
            } else {
                currRequest.time = scan.next();
            }
            //convert the time to an integer for easy comparison and conversion then add the current request to the list of requests
            currRequest.timeInt = Integer.parseInt(currRequest.time);
            requests.add(currRequest);
        }
        scan.close();

        // go one by one through the requests and handle them based on their type, while also keeping track of the time and max wait time
        int index = 0;
        while (index < requests.size()) {
            //index + 1 is used to cycle though the requests and handle them
            

            //get the time of the current request and convert it to minutes for easy comparison
            String time = requests.get(index).time;
            int currentMinutes = (requests.get(index).timeInt / 100) * 60 + requests.get(index).timeInt % 100;
            
            // Handle new requests.
            for (int i = index; i < index + 1; i++) {
                Requests currRequest = requests.get(i);
                if (currRequest.type.equals("ChatRequest")) {
                    System.out.println("ChatRequest " + currRequest.time + " " + currRequest.customer + " " + currRequest.choice);

                    //if there is a representative available, then assign them to the customer and print the rep assignment message to the console
                    if (repHead != null) {
                        Node rep = repHead;
                        repHead = repHead.next;
                        rep.next = null;
                        if (repHead == null) {
                            repTail = null;
                        }
                        //add the customer and representative to the chat list
                        Node chat = new Node();
                        chat.name = currRequest.customer;
                        chat.rep = rep.name;
                        if (chatHead == null) {
                            chatHead = chat;
                            chatTail = chat;
                        } else {
                            chatTail.next = chat;
                            chatTail = chat;
                        }
                        System.out.println("RepAssignment " + currRequest.customer + " " + rep.name + " " + currRequest.time);

                        //if the person says to wait, then add them to the hold list and print the put on hold message to the console
                        //else print the try later message to the console
                    } else if (currRequest.choice.equals("wait")) {
                        Node customer = new Node();
                        customer.name = currRequest.customer;
                        customer.time = (currRequest.timeInt / 100) * 60 + currRequest.timeInt % 100;
                        if (onHoldHead == null) {
                            onHoldHead = customer;
                            onHoldTail = customer;
                        } else {
                            onHoldTail.next = customer;
                            onHoldTail = customer;
                        }
                        System.out.println("PutOnHold " + currRequest.customer + " " + currRequest.time);
                    } else {
                        System.out.println("TryLater " + currRequest.customer + " " + currRequest.time);
                    }
                }
            }

            // End chats and return representatives to the available list.
            for (int i = index; i < index + 1; i++) {
                Requests currRequest = requests.get(i);
                if (currRequest.type.equals("ChatEnded")) {
                    //print the chat ended message to the console
                    System.out.println("ChatEnded " + currRequest.customer + " " + currRequest.rep + " " + currRequest.time);

                    // Remove the customer from the chat list.
                    Node current = chatHead;
                    Node previous = null;

                    // Find the customer in the chat list. Goes through each node until curr name is equal to name in list
                    while (current != null && !current.name.equals(currRequest.customer)) {
                        previous = current;
                        current = current.next;
                    }
                    // If the customer was found, remove it from the chat list.
                    // If the customer is not the head, remove it from the list by linking the previous node to the next node.
                    if (current != null) {
                        if (previous == null) {
                            chatHead = current.next;
                        } else {
                            previous.next = current.next;
                        }
                        //then update the tail if the current node is the tail
                        if (current == chatTail) {
                            chatTail = previous;
                        }
                    }

                    // used to add the representative back to the available list
                    Node newRep = new Node();
                    newRep.name = currRequest.rep;
                    if (repHead != null) {
                        repTail.next = newRep;
                        repTail = newRep;
                        
                    } else {
                        repHead = newRep;
                        repTail = newRep;
                    }
                }
            }

            // Give available representatives to old held customers first.
            // This only happens if there are both representatives and customers on hold. If either is null, then this while loop will not run.
            while (repHead != null && onHoldHead != null) {
                Node rep = repHead;
                repHead = repHead.next;
                rep.next = null;
                //if the repHead is null, then the repTail should also be null because there are no more representatives available
                if (repHead == null) {
                    repTail = null;
                }
                //get the customer from the hold list and remove it from the hold list
                Node customer = onHoldHead;
                onHoldHead = onHoldHead.next;
                customer.next = null;
                if (onHoldHead == null) {
                    onHoldTail = null;
                }

                //add the customer and representative to the chat list
                Node chat = new Node();
                chat.name = customer.name;
                chat.rep = rep.name;
                if (chatHead == null) {
                    chatHead = chat;
                    chatTail = chat;
                } else {
                    chatTail.next = chat;
                    chatTail = chat;
                }

                //calculate the wait time for the customer and update the max wait time if necessary
                int wait = currentMinutes - customer.time;
                if (wait > maxWait) {
                    maxWait = wait;
                }
                //print the representative assignment to the console
                System.out.println("RepAssignment " + customer.name + " " + rep.name + " " + time);
            }

           

            // Remove customers who quit waiting.
            for (int i = index; i < index + 1; i++) {
                Requests currRequest = requests.get(i);
                if (currRequest.type.equals("QuitOnHold")) {
                    Node current = onHoldHead;
                    Node previous = null;
                    // Find the customer in the hold list. Goes through each node until curr name is equal to name in list
                    while (current != null && !current.name.equals(currRequest.customer)) {
                        previous = current;
                        current = current.next;
                    }
                    //checks if the customer was found in the hold list, if so, remove them from the list and update the max wait time if necessary
                    if (current != null) {
                        if (previous == null) {
                            onHoldHead = current.next;
                        } else {
                            previous.next = current.next;
                        }
                        if (current == onHoldTail) {
                            onHoldTail = previous;
                        }

                        int wait = currentMinutes - current.time;
                        if (wait > maxWait) {
                            maxWait = wait;
                        }
                        //print the quit on hold message to the console
                        System.out.println("QuitOnHold " + currRequest.time + " " + currRequest.customer);
                    }
                }
            }

            // Handle the two print commands.
            for (int i = index; i < index + 1; i++) {
                Requests currRequest = requests.get(i);
                //print the available representatives in the order they are in the list
                if (currRequest.type.equals("PrintAvailableRepList")) {
                    System.out.print("AvailableRepList " + currRequest.time);
                    Node current = repHead;
                    while (current != null) {
                        System.out.print(" " + current.name);
                        current = current.next;
                    }
                    System.out.println();

                    //format the max wait time
                } else if (currRequest.type.equals("PrintMaxWaitTime")) {
                    String maxWaitTime = String.format("%02d", (maxWait / 60)) + String.format("%02d", (maxWait % 60));
                    System.out.println("MaxWaitTime " + currRequest.time + " " + maxWaitTime);
                }
            }
            //increment the index + 1 to move to the next request in the list
            index++;
        }
    }
}