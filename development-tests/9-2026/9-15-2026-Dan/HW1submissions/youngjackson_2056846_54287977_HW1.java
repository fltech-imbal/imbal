/*
* Author: Jackson Young
* Email: young2025@my.fit.edu
* Course: CSE2010
* Section: 1
* Description of this file: This program simulates a chat between a customer and a customer service representative.
*   Given a set of inputs defining the actions taken during the chat, the output will list the results of these actions
*   depending on the order and availability of the representatives.
*/
/* INPUTS                                           OUTPUTS
    ChatRequest requestTime customer waitOrLater    ChatRequest requestTime customer waitOrLater
    QuitOnHold quitOnHoldTime customer              RepAssignment customer rep assignmentTime
    ChatEnded customer rep endTime                  PutOnHold customer putOnHoldTime
    PrintAvailableRepList printTime                 TryLater customer tryLaterDecisionTime
    PrintMaxWaitTime printTime                      QuitOnHold quitOnHoldTime customer
    -                                               ChatEnded customer rep endTime
    -                                               AvailableRepList printTime rep1 rep2 ...
    -                                               MaxWaitTime printTime maxWaitTimeSoFar
*/
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
public class HW1 {

    // Hidden private class called Customer to keep track of both customers' name and when they started waiting
    private static class Customer {
        public String name;
        public String waitStartTime;
        // Constructor
        public Customer(String name, String waitStartTime) {
            this.name = name;
            this.waitStartTime = waitStartTime;
        }
        // toString method for debugging
        @Override
        public String toString() {return name+" at "+waitStartTime;}
    }

    // Global Variable to store the time-formatted maximum wait time among customers
    static String maxWaitTime = "0000";

    // Initialize a SinglyLinkedList for:
    // Available representatives
    static SinglyLinkedList<String> availableReps = new SinglyLinkedList<>();
    // Customers currently chatting
    static SinglyLinkedList<Customer> customersChatting = new SinglyLinkedList<>();
    // Customers on hold
    static SinglyLinkedList<Customer> customersOnHold = new SinglyLinkedList<>();

    // main method starts the program by processing input and calling the necessary methods
    public static void main(String[] args) {
        // Scanner class will take input tokens and stream them into the program
        // Here the scanner is initialized
        Scanner scanner;
        // Exception catch block to load the file into the scanner
        // Program will terminate with a message if an IO error occurs
        try {
            scanner = new Scanner(new File(args[0]));
        } catch (FileNotFoundException e) {
            System.out.println("FileNotFoundException: Invalid Input\n" + e.getMessage() + "\nProgram Terminated");
            return; // End Program
        }
        
        // Add the starting order to the available reps list
        availableReps.addLast("Alice");
        availableReps.addLast("Bob");
        availableReps.addLast("Carol");
        availableReps.addLast("David");
        availableReps.addLast("Emily");

        // Process first input token as the action each line
        // and stop when no input is remaining
        while (scanner.hasNext()) {
            String action = scanner.next();

            /****************************\
                EVENT: Chat Request
            \****************************/
            if (action.equals("ChatRequest")) {
                // Obtain the other parameters of the input action using Scanner
                String requestTime = scanner.next();
                String customer = scanner.next();
                String waitOrLater = scanner.next();
                System.out.println("ChatRequest " + requestTime + " " + customer + " " + waitOrLater);
                
                // Check if a representative is available to chat
                // If so, take them off the availability list and put the customer on the chatting list
                // Otherwise, either put the customer on the hold list if they want to wait or skip the customer entirely if they don't
                if (!availableReps.isEmpty()) {
                    // Representative available
                    String rep = availableReps.removeFirst();
                    customersChatting.addLast(new Customer(customer, requestTime));
                    System.out.println("RepAssignment " + customer + " " + rep + " " + requestTime);
                } else {
                    // Representative unavailable
                    if (waitOrLater.equals("wait")) {
                        // Customer wants to be put on hold
                        customersOnHold.addLast(new Customer(customer, requestTime));
                        System.out.println("PutOnHold " + customer + " " + requestTime);
                    } else {
                        // Customer does not want to wait
                        System.out.println("TryLater " + customer + " " + requestTime);
                    }
                }

            /****************************\
                EVENT: Quit On Hold
            \****************************/
            } else if (action.equals("QuitOnHold")) {
                // Get other parameter tokens from Scanner
                String quitHoldTime = scanner.next();
                String customer = scanner.next();

                // If the customer no longer wants to be on hold, find and remove them from the hold list
                Customer customerRemoved = removeCustomerFromHold(customer);
                System.out.println("QuitOnHold " + quitHoldTime + " " + customer);

                // Compare wait time
                compareMaxWaitTime(customerRemoved.waitStartTime, quitHoldTime);

            /****************************\
                EVENT: Chat Ended
            \****************************/
            } else if (action.equals("ChatEnded")) {
                // Get parameters from Scanner
                String customer = scanner.next();
                String rep = scanner.next();
                String endTime = scanner.next();
                
                // Find and remove the customer from the chatting list
                removeCustomerFromChat(customer);
                System.out.println("ChatEnded " + customer + " " + rep + " " + endTime);

                // Check for any customers currently on hold and assign the rep to them
                if (!customersOnHold.isEmpty()) {
                    // Remove customer from hold
                    Customer newCustomer = customersOnHold.removeFirst();
                    // Assign the rep to a new chat with the customer
                    customersChatting.addLast(newCustomer);
                    System.out.println("RepAssignment " + newCustomer.name + " " + rep + " " + endTime);
                    // Compare wait time
                    compareMaxWaitTime(newCustomer.waitStartTime, endTime);
                } else {
                    // Add representative back to the available list
                    availableReps.addLast(rep);
                }

            /****************************\
             EVENT: Print Available Reps
            \****************************/
            } else if (action.equals("PrintAvailableRepList")) {
                // Get parameters from Scanner
                String printTime = scanner.next();

                // Attempt to make an interable copy of the availible reps list
                // If attempt succeeds, loop through the list and print each element
                // If attempt fails, catch the CloneNotSupportedException in an error message
                SinglyLinkedList<String> clonedList;
                try {
                    // Safely clone the list using provided clone() method
                    clonedList = availableReps.clone();

                    // Using the clonedList, loop until the list is empty
                    // and print out each value removed
                    System.out.print("AvailableRepList " + printTime);
                    while (!clonedList.isEmpty()) {
                        System.out.print(" " + clonedList.removeFirst());
                    }
                    // Line break
                    System.out.println();

                } catch (CloneNotSupportedException e) {
                    // Warning message to skip the print event
                    System.out.println("CloneNotSupportedException: " + e.getMessage() + "\nPrintAvailableRepList Failed");
                }

            /****************************\
             EVENT: Print Max Wait Time
            \****************************/
            } else if (action.equals("PrintMaxWaitTime")) {
                // Get parameters from Scanner
                String printTime = scanner.next();
                // Simply print the maxWaitTime global variable that keeps track
                // of the assignmentTime - requestTime
                System.out.println("MaxWaitTime " + printTime + " " + maxWaitTime);
            }
        }
    }

    // A private helper method to search and remove Customers from the customersOnHold list
    private static Customer removeCustomerFromHold(String name) {
        SinglyLinkedList<Customer> newList = new SinglyLinkedList<>();
        Customer customerToRemove = new Customer("", "");
        // Put each Customer of the original list into the newList unless it matches the name it's looking for
        while (!customersOnHold.isEmpty()) {
            Customer current = customersOnHold.removeFirst();
            if (!current.name.equals(name)) {
                newList.addLast(current);
            } else customerToRemove = current;
        }
        // Hand the new, edited list back to the original list and return the removed element
        customersOnHold = newList;
        return customerToRemove;
    }

    // A similar private helper method to search and remove Customers from the customersChatting list
    private static Customer removeCustomerFromChat(String name) {
        SinglyLinkedList<Customer> newList = new SinglyLinkedList<>();
        Customer customerToRemove = new Customer("", "");
        // Put each Customer of the original list into the newList unless it matches the name it's looking for
        while (!customersChatting.isEmpty()) {
            Customer current = customersChatting.removeFirst();
            if (!current.name.equals(name)) {
                newList.addLast(current);
            } else customerToRemove = current;
        }
        // Hand the new, edited list back to the original list and return the removed element
        customersChatting = newList;
        return customerToRemove;
    }

    // A private helper method to compare the difference between the two times with the max wait time global variable
    private static void compareMaxWaitTime(String waitStart, String waitEnd) {
        // Find wait time and convert to integer
        int timeDifferenceInt = convertToIntMinutes(waitEnd) - convertToIntMinutes(waitStart);
        // Convert global variable to integer
        int maxWaitTimeInt = convertToIntMinutes(maxWaitTime);
        // Compare for the largest value and replace maxWaitTime if necessary
        if (timeDifferenceInt > maxWaitTimeInt) maxWaitTime = convertToTimeFormat(timeDifferenceInt);
    }

    // Helper method to convert time format (HHMM) to an integer representing the total minutes
    private static int convertToIntMinutes(String timeStr) {
        int hours = Integer.valueOf(timeStr.substring(0, 2));
        int minutes = Integer.valueOf(timeStr.substring(2, 4));
        // return time as an integer
        return hours * 60 + minutes;
    }

    // Helper method to convert total minutes integer to a time format String (HHMM)
    private static String convertToTimeFormat(int totalMinutes) {
        String hours = String.format("%02d", totalMinutes / 60);
        String minutes = String.format("%02d", totalMinutes % 60);
        // return time as a formatted String HHMM
        return hours + minutes;
    }
}
