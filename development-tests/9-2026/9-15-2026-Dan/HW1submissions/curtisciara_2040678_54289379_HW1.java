/*
  Author: Ciara Curtis
  Email: ccurtis2025@my.fit.edu
  Course: CSE2010
  Section: 4
  Description of this file: Homework 1 submission detailing a program that mimics a customer service chatroom,
                            where a customer is able to request a chat with a representative and either be helped
                            or put on hold to help later. The customer also has the option to quit while on hold
                            or simply try again later.
 */

import java.util.Scanner;
import java.io.File;

public class HW1 {
    private static SinglyLinkedList availableReps = new SinglyLinkedList();
    //Singly Linked List to hold the available representatives.

    private static SinglyLinkedList waitList = new SinglyLinkedList();
    //Singly Linked List to hold customers that need to be helped.

    private static int maxWaitTime = 0;
    //variable to hold the max wait time

    //Calculates the minutes Updates the maxWaitTime variable when a higher maxWaitTime is found
    public static void updateMaxWaitTime(int requestTime, String customer) {
	int totalTime = requestTime - waitList.getNodeTime(customer);
        if(maxWaitTime < totalTime) maxWaitTime = totalTime;
    }

    /*  Function that reprints the intitial request type, and then assigns a representative if
	available. If not available then the customer is moved to the waitlist to be assigned
	when either the quitOnHold or chatEnded functions execute.
    */
    public static void chatRequest(int requestTime, String customer, String waitOrLater) {
        System.out.printf("ChatRequest %04d %s %s %n", requestTime, customer, waitOrLater);
        //reprints the original input

        //if availableReps length > 0, assign rep and reduce replist
	if(availableReps.size() > 0) {
	    System.out.printf("RepAssignment %s %s %04d %n", customer, availableReps.removeFirst(), requestTime); 
	}

	//else, assign customer to waitlist if no reps available
	else {
	    if(waitOrLater.equals("later")) System.out.printf("TryLater %s %04d %n", customer, requestTime);
	    else{
            	waitList.addLast(customer, requestTime);
             	System.out.printf("PutOnHold %s %04d %n", customer, requestTime);
            }
        }
    }

    /*  Function that removes a customer from the customer wait list and 
	updates the max wait time
    */
    public static void quitOnHold(int quitOnHoldTime, String customer) {
        updateMaxWaitTime(quitOnHoldTime, customer);
        System.out.printf("QuitOnHold %04d %s %n", quitOnHoldTime, waitList.removeNode(customer));
    }
    
    /*	After a chat ends, the rep gets added back to the available rep list and if there are any extra
      	customers on the wait list, they are assigned to the next available rep
    */
    public static void chatEnded(String customer, String rep, int endTime) {
        System.out.printf("ChatEnded %s %s %04d %n", customer, rep, endTime);
        availableReps.addLast(rep, 0);
        if(waitList.size() > 0) {
	    updateMaxWaitTime(endTime, waitList.firstElement());
            System.out.printf("RepAssignment %s %s %04d %n", waitList.removeFirst(), availableReps.removeFirst(), endTime);
	}
    }
    //Uses the toString method in the SinglyLinkedList class to list all the available reps in the availableReps list
    public static void printAvailableRepList(int printTime) {
	System.out.printf("AvailableRepList %04d ", printTime);
        System.out.println(availableReps);
    }
    
    //Prints and formats the maxWaitTime global variable
    public static void printMaxWaitTime(int printTime) {
        System.out.printf("MaxWaitTime %04d %04d %n", printTime, maxWaitTime);	
    } 

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(new File(args[0])); //opens up a new scanner that reads the file from the command line argument
	availableReps.addLast("Alice",0);
	availableReps.addLast("Bob",0);
	availableReps.addLast("Carol",0);
 	availableReps.addLast("David",0);
	availableReps.addLast("Emily",0);                 //Appends all the reps into the availableReps list before processing any of the requests
                                                          //from the file

	while(scanner.hasNextLine()) {           //While there are still lines to be read, the scanner will continue accepting tokens         
	    String requestType = scanner.next(); //Holds the type of request needing to be processed
	    if(requestType.equals("ChatRequest")) chatRequest(scanner.nextInt(), scanner.next(),scanner.next());
	    else if(requestType.equals("QuitOnHold")) quitOnHold(scanner.nextInt(),scanner.next());
	    else if(requestType.equals("ChatEnded")) chatEnded(scanner.next(), scanner.next(), scanner.nextInt());
	    else if(requestType.equals("PrintAvailableRepList")) printAvailableRepList(scanner.nextInt());
	    else if(requestType.equals("PrintMaxWaitTime")) printMaxWaitTime(scanner.nextInt());
	    /*If statements are meant to respond to each request type that comes in. 
	      Each if statement compares the requestType variable to the different request types 
              that can come in and executes the corresponding method.
	    */
            if(!scanner.hasNext()) break;
	}
        scanner.close();
    }

}
