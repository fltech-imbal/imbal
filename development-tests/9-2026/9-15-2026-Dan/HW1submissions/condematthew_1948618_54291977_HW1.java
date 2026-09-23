/*

  Author: Matthew Conde
  Email: mconde2022@my.fit.edu	
  Course: CSE 2010
  Section: 02
  Description of this file: 
		This file is a set of singly-linked lists that hold information in queues, holding available representatives
		customers on hold, and customers currently chattin with a representative. It acts as a customer service system that prioritizes
		customers first-come-first serve, tracks how long hold times are taking, and gives anyone the max wait time in a given run (segment of time).
		Uses a customer class to track max wait time and classic list node pop and push methods.

 */

import java.io.File;
import java.util.Scanner;

public class HW1
{

	/* 
	   create customer object/class so that we can store request time and find wait time after waiting on hold
	   create accessor methods so we can remove a customer from the hold list and access request time 
	*/

    private static class Customer {
	String name;						// relevant variables, customer name & request time
	int requestTime;
	
	public Customer(String name, int requestTime){		// setter method for initializing a customer
	    this.name = name;
	    this.requestTime = requestTime;
	}
	public String getName(){ return name; }			// accessor methods for getting relevant variables
	public int getRequestTime() { return requestTime; }
    }

		
	/* 
	   intializing lists as static variables so they can be accessed inside the methods and 
	   not just from main 
	*/

    private static SinglyLinkedList<String> availableReps = new SinglyLinkedList<String>();	
    private static SinglyLinkedList<Customer> custsHold = new SinglyLinkedList<Customer>();
    private static SinglyLinkedList<String> chats = new SinglyLinkedList<String>();
    
    private static int maxWait = 0;						// maxwait static to access from methods



	/*-------------------------BEGIN METHODS---------------------------*/

	/* 
	   method/function for requesting a chat, if no reps are available default to wait or later
	   if they wish to wait, call put on hold, if they do not, call try later.
	   inputs are request time, the customer name, and waitOrLater string (acts as a flag)
	 */

    public static void chatRequest(int reqTime, String customer, String waitOrLater){

	System.out.println("ChatRequest " +format(reqTime)+" "+customer+" "+waitOrLater);  // output to screen for chatrequest
	if (availableReps.isEmpty()){						//check if any reps are available
	    if (waitOrLater.equals("wait")){					// if none available, check if they want to wait or try later
		putOnHold(reqTime, customer);					// wait
	    } else {
		tryLater(reqTime, customer);					// try later
	    }
	} else {
	    Customer cust  = new Customer(customer, reqTime);			//make new customer object
	    repAssignment(cust, reqTime);					// if a rep is available, assign customer to a rep
	}
        return;
    }

	/*
	    method for assigning customer to a representative, remove(pop) rep from list and output 
	    result to screen, inputs are the customer and assignment(request) time 
	*/

    public static void repAssignment(Customer cust, int assTime){
	String rep = availableReps.removeFirst();						//pop first representative from list
	System.out.println("RepAssignment " + cust.getName() + " " + rep + " " +format(assTime));	//output for rep assignment
	chats.addLast(cust.getName());  							//adding customer to active chats (do I need to add representative???!!!!

	return;
    }   
	
	/*
	    method for placing customer on hold, add customer to end of hold list and output expected line.
	    inputs are the time put on hold and the customer name 
	*/

    public static void putOnHold (int putOnHoldTime, String customer){
	System.out.println("PutOnHold "+customer+" "+format(putOnHoldTime));		// corresponding output
	Customer cust  = new Customer(customer, putOnHoldTime);			// create customer object to place on hold
	custsHold.addLast(cust);						// add node to list
	return;
    }

	/* 
	   method for trying later, outputs when the decision was made and the customer 
	   who made the decision 
	*/

    public static void tryLater (int tryLaterTime, String customer){
        System.out.println("TryLater "+customer+" "+format(tryLaterTime));		//simple output to screen
	return;
    }


	/* 
	   method for quitting while on hold, outputs when decision was made and which customer,
	   inputs are the same. removes selected customer from customer hold list by popping 
	   first until found and adding to last until back in same order with target removed
	*/

    public static void quitOnHold (int quitOnHoldTime, String cust){
	System.out.println("QuitOnHold "+format(quitOnHoldTime)+" "+cust);		//output to screen
	
	int count = custsHold.size();						//counter to loop through list
	Customer found = null;							//initialize customer looking for
	for (int i = 0; i< count; i++){						// loop through customers on hold
	    Customer c = custsHold.removeFirst();				// pop first 	

	    if (c.getName().equals(cust) && found == null ){			// check if found customer, then keep that customer info
		found = c;
	    } else {
		custsHold.addLast(c);						//if not found, add to back of list
	    }
	}
	if (found != null) {        						// if found, calculate wait time
            int requestTime = found.getRequestTime();
            waitTimeCalc(requestTime, quitOnHoldTime);
	}
	return;
    }

	/* 
	   method for ending the chat session, outputs when decision was made, which customer, and
	   which representative. inputs are the same information, removes customer chat session from 
	   chat session list 
	*/

    public static void chatEnded (String customer, String rep, int endTime){
	System.out.println("ChatEnded "+customer+" "+rep+" "+ format(endTime));			// screen output
	
	int count = chats.size();							// counter to loop through list
        String found = null;								//initialize customer looking for
        for (int i = 0; i< count; i++){ 						// loop through chat sessions
            String c = chats.removeFirst();						//pop first

            if (c.equals(customer) && found == null ){					// if found, keep customer info
                found = c;	
            } else {									// if not, add to back of list
                chats.addLast(c);
            }
        }
	
	if (!custsHold.isEmpty()){							// if someone on hold, assign them to rep
	    Customer nextCust = custsHold.removeFirst();	
	    availableReps.addLast(rep);	    						// add rep to list	
	
	    waitTimeCalc(nextCust.getRequestTime(), endTime); 				// calculate wait time
	    repAssignment(nextCust, endTime);						// assign them to rep

	} else { availableReps.addLast(rep); }						// if no one on hold, add rep to availablef 
	return;
    }


	/* 
	   print available representatives, loop through the available reps, printing each and moving to the 
	   next in the list 
	*/

    public static void availableRepList (int printTime){	
	int count = availableReps.size();						// counter to loop through list

	System.out.print("AvailableRepList " + format(printTime));				// print initial portion of ouput
	
	for (int i = 0; i< count; i++){							// for each representative, pop, print, push
	    String current  = availableReps.removeFirst();
	    System.out.print(" "+current);
	    availableReps.addLast(current);
	}
	System.out.println();								// start new line in print screen
	return; 
    }

	/* 
	   takes a given start and end time and calculates the wait time, assigning this wait to 
	   max wait time if it is the largest wait time so far, keeps it in HHMM format after doing calculations
	   only in minutes. 
	*/

    public static void waitTimeCalc(int startWait, int endWait){
	int start = ((startWait/100)*60) + (startWait % 100);				// convert both times to minutes
	int end = ((endWait/100)*60) + (endWait % 100);

	int max = end - start; 								// find difference in times
	int maxWaitTime = ((max/60)*100) + (max % 60);					// move difference back to HHMM
	if (maxWaitTime > maxWait){
	    maxWait = maxWaitTime;							// if larger than max wait time, set new max wait time
	}
    }    
	

	/* simple helped to keep HHMM format in all outputs */

    public static String format(int time) {
        return String.format("%04d", time);
    }

    public static void main(String[] args) throws Exception
    {
										// fill representative list with given reps	
	availableReps.addLast("Alice");
        availableReps.addLast("Bob");
        availableReps.addLast("Carol");
        availableReps.addLast("David");
        availableReps.addLast("Emily"); 
										// get new file from command line
        File inputFile = new File(args[0]);
	Scanner sc = new Scanner(inputFile);					// initialize scanner

	/* 
	   initialize variables for switch case, don't want to re-declare every case, 
       	   names are self-explanatory. Time of request, time quit from hold, time ended chat,
	   time printed quantity, customer name, wait or later?, and representative name 
	*/

	int reqTime, quitOnHoldTime, endTime, printTime = 0;
	String customer, waitOrLater, rep = null;
	
	while (sc.hasNext()) {							// while there is more info to be scanned from input
 	    switch (sc.next()) {						// switch based on the next scanned string, will be a command
	        case "ChatRequest":						// request a chat, pull information and then call chatRequest function
		    reqTime = sc.nextInt();
		    customer = sc.next();
		    waitOrLater = sc.next();

		    chatRequest(reqTime, customer, waitOrLater);
		    break;

		case "QuitOnHold":						// quit while on hold, pull information and call quitOnHold
		    quitOnHoldTime = sc.nextInt();
		    customer = sc.next();
		    
		    quitOnHold(quitOnHoldTime, customer);
		    break;

		case "ChatEnded":						// end chat, pull information and call chatEnded
		    customer = sc.next();
		    rep = sc.next();
		    endTime = sc.nextInt();

		    chatEnded(customer, rep, endTime);
		    break;

		case "PrintAvailableRepList":					// print available representatives, pull info and call print function
		    printTime = sc.nextInt();

		    availableRepList(printTime);
		    break;

		case "PrintMaxWaitTime":					// pull print time and print global variable maxWait
		    printTime = sc.nextInt();

		    System.out.println("MaxWaitTime "+format(printTime)+" "+format(maxWait));
		    break;
 	  }
        }
	sc.close();								// close scanner
      }
  }
