/*
Author: Jerye Demers-St.Hilaire
Email: jdemerssthil2025@my.fit.edu
Course: Data Structures & Algorithms
Section: 1
Description of this file: Computing time spent on hold, time spent on chats, and order of incoming customers. 
*/

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class HW1 { 
public static void main(String[] args) throws FileNotFoundException {
   
   Scanner scanner = new Scanner(new File(args[0]));

   // Creating a linked list for available representatives   
   SinglyLinkedList<String> AvailableReps = new SinglyLinkedList<>();
   AvailableReps.addLast("Alice");
   AvailableReps.addLast("Bob");
   AvailableReps.addLast("Carol");
   AvailableReps.addLast("David");
   AvailableReps.addLast("Emily");
   
   // Creating a linked list for unavailable representatives
   SinglyLinkedList<String> UnavailableReps = new SinglyLinkedList<>();
   
   // Creating linked lists for customers on hold and the start time of the hold
   SinglyLinkedList<String> OnHold = new SinglyLinkedList<>();
   SinglyLinkedList<Integer> HoldStartTimes = new SinglyLinkedList<>();

   int maxWaitTime = 0;
   
   // Storing the input and breaking it into parts
   while (scanner.hasNextLine()) {
   String line = scanner.nextLine();
   
   String[] input = line.split(" "); 
   
   // Checking if the first word of input is ChatRequest
   // Checking if there are any available reps, assigning the rep to the customer
   // Moving the rep to unavailable
   // If no reps are available check if the 4th word is wait, if it is put the customer on hold
   // add customer to hold list, add time to hold start time
   // If the 4th word is not wait then output TryLater
   if ("ChatRequest".equals(input[0])) {
     System.out.println(input[0] + " " + input[1] + " "	+ input[2] + " " + input[3]);
   	if (AvailableReps.isEmpty() == false) {
   		String RepAssignment = AvailableReps.first();
   		AvailableReps.removeFirst();
   		System.out.println("RepAssignment " + input[2] + " " + RepAssignment + " " + input[1]);
   		UnavailableReps.addLast(RepAssignment);
 	}
   	else if ("wait".equals(input[3])) {
   			System.out.println("PutOnHold " + input[2] + " " + input[1]);
   			String hour = input[1].substring(0,2);
   			String mins = input[1].substring(2,4);
   			int holdStartHour = Integer.parseInt(hour);
   			int holdStartMins = Integer.parseInt(mins);
   			int holdStartTime = (holdStartHour * 60) + holdStartMins;
   			OnHold.addLast(input[2]);
   			HoldStartTimes.addLast(holdStartTime);
   		}
   	      
   	else
   		     System.out.println("TryLater " + input[2] + " " + input[1]);
   		     	
   	}
   
 
   // Check if the first word is ChatEnded, if it is check if anyone is on hold
   // If someone is on hold take them off the hold list, assign them the rep, compute the start time and end time and compare to the current max wait time
   // If no one is on hold, loop the unavailable reps list until you find the rep in the input, remove the rep from unavailable and put them on available
   if ("ChatEnded".equals(input[0])) {
     System.out.println(input[0] + " " + input[1] + " " + input[2] + " " + input[3]);
        if (!OnHold.isEmpty()) {
           String Customer = OnHold.first();
           OnHold.removeFirst();
           System.out.println("RepAssignment " + Customer + " " + input[2] + " " + input[3]);
           String hourEnd = input[3].substring(0,2);
           String minsEnd = input[3].substring(2,4);
           int HEbcCEhours = Integer.parseInt(hourEnd);
           int HEbcCEmins = Integer.parseInt(minsEnd);
           int ChatEndedTime = (HEbcCEhours * 60) + HEbcCEmins;
           int CusHoldStartTime = HoldStartTimes.first();
           HoldStartTimes.removeFirst();
           int waitTimeMins = ChatEndedTime - CusHoldStartTime;
           if (waitTimeMins > maxWaitTime) {
           	maxWaitTime = waitTimeMins;
           }
        }
        
        else if (UnavailableReps.first().equals(input[2])) {
   	        UnavailableReps.removeFirst();
   	        AvailableReps.addLast(input[2]);
   	     }
   	else {
   	    while (!UnavailableReps.first().equals(input[2])) {
   	     String head = UnavailableReps.first();
   	     UnavailableReps.removeFirst();
   	     UnavailableReps.addLast(head);
   	     }
   	     
       UnavailableReps.removeFirst();
       AvailableReps.addLast(input[2]);
     
     }
   }
   
   // Check if the first word is QuitOnHold
   // Store the time, compute the length of the hold using the hold start time, check if it is larger than the current max wait time, if it is replace it
   // Loop through the on hold customers until you find the one that quit and remove them and do the same for their hold start time
   if ("QuitOnHold".equals(input[0])) {
     System.out.println(input[0] + " " + input[1] + " "	+ input[2]);
   	String hourQ = input[1].substring(0,2);
   	String minsQ = input[1].substring(2,4);
   	int hourE = Integer.parseInt(hourQ);
   	int minsE = Integer.parseInt(minsQ);
   	int quitTime = (hourE * 60) + minsE;
        int startTime;
        
        if (OnHold.first().equals(input[2])) { 
            startTime = HoldStartTimes.first();
            OnHold.removeFirst(); 
            HoldStartTimes.removeFirst();
         }
       
        else {
         while (!OnHold.first().equals(input[2])) {
          String customer = OnHold.first();
          OnHold.removeFirst();
          OnHold.addLast(customer);
          int holdTime = HoldStartTimes.first();
          HoldStartTimes.removeFirst();
          HoldStartTimes.addLast(holdTime);
          }
          
          int holdTime = HoldStartTimes.first();
          HoldStartTimes.removeFirst();
          OnHold.removeFirst();
          int waitTime = quitTime - holdTime;
          
          
          if (waitTime > maxWaitTime) {
           maxWaitTime = waitTime;
           }
          }
       }
   
   // Output the current available representative
   if ("PrintAvailableRepList".equals(input[0])) {
    	System.out.println("AvailableRepList " + input[1] + AvailableReps);
   }
   
   // Format the max wait time correctly and output it
   if ("PrintMaxWaitTime".equals(input[0])) {
     int waitHours = maxWaitTime / 60;
     int waitMinutes = maxWaitTime % 60;
     int maxWaitFormated = (waitHours * 100) + waitMinutes;
   	System.out.printf("MaxWaitTime %s  %04d%n ", input[1] , maxWaitFormated);
   }
   
  
  }
 }


}
