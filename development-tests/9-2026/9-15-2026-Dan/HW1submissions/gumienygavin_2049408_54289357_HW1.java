/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
  This file handles chatting with a customer service representative. This includes requesting chats,
  endings chats, quitting while on hold, printing a list of available representatives, and printing 
  the maximum wait time. Additionally, it handles customers by putting them on hold when there is no
  representative available. It takes an input file name as a command-line argument and reads each input
  line within the file. Within each line, it reads the name of the method to be called followed by each 
  of its parameters.

 */

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class HW1
{
    // Singly linked lists that maintains the information for available representatives, customers on hold, and chat sessions.
    public static SinglyLinkedList<String> availableRep = new SinglyLinkedList<>();
    public static SinglyLinkedList<CustomerOnHold> customersOnHold = new SinglyLinkedList<>();
    public static SinglyLinkedList<ChatSession> chatSessions = new SinglyLinkedList<>();
    
    // Variable to store the maximum wait time.
    private static int maxWaitTimeSoFar = 0;
    
    /** Formats the output time so it displays the leading zero(s) (if necessary)
     * 
     *  @param time  the time being formatted
     */ 
    public static String formatTime(int time) {
        return String.format("%04d", time);
    }
    
    /** Calculates the total time a customer waited on the holding list.
     * 
     *  @param startTime  the time the customer was put on the hold list
     *  @param endTime  the time the customer was taken off of the hold list
     * 
     */
    public static void calculateMaxWaitTime(int startTime, int endTime) {
        // Converts startTime to total minutes
        int startHh = startTime / 100;                    // stores the HH portion of startTime
        int startMm = startTime % 100;                    // stores the MM portion of startTime
        int startTimeMins = (startHh * 60) + startMm;
        
        // Converts endTime to total minutes
        int endHh = endTime / 100;                        // stores the HH portion of endTime
        int endMm = endTime % 100;                        // stores the MM portion of endTime
        int endTimeMins = (endHh * 60) + endMm;
        
        // Calculates the wait time in minutes
        int waitTimeMins = endTimeMins - startTimeMins;
        
        // Converts the total wait time to HHMM format
        int correctHh = waitTimeMins / 60;
        int correctMm = waitTimeMins % 60;
        int correctWaitTime = (correctHh * 100) + correctMm;
        
        // Checks if the correct wait time is greater than the maximum wait time (updates variable if yes)
        if (correctWaitTime > maxWaitTimeSoFar) {
            maxWaitTimeSoFar = correctWaitTime;
        }
    }
    
    /** Checks the time input to make sure it is within bounds of HHMM format. For example, no 1299, 1260, 2400, or negative times.
     * 
     *  @param time  the time to be evaluated
     */
    public static boolean isValidTime(int time) {
        // Checks if the time is negative
        if (time < 0) {
            return false;
        }
        
        // Splits the hours and minutes
        int hh = time / 100;
        int mm = time % 100;
        
        // Checks to see if hours and minutes are within bounds. HH(00-23), MM(00-59). Returns true if they are.
        if (hh > 23) {
            return false;
        }
        else if (mm > 59) {
            return false;
        }
        else {
            return true;
        }
    }
    
    /** Requests a chat session between a customer and a representative. If there
     *  are no representatives available, the customer is either put on hold or
     *  will try later at their discresion. 
     * 
     *   @param requestTime  the time a chate was requested
     *   @param customer  the name of the customer
     *   @param waitOrLater  whether the customer chooss to wait if there are no represenatives available
     */
    public static void chatRequest(int requestTime, String customer, String waitOrLater) {
        // Outputs that a customer requested a chat session.
        System.out.println("ChatRequest " + formatTime(requestTime) + " " + customer +
        " " + waitOrLater);
        // 1. Checks if a representative is available, then creates a new chat session and assigns a representative to a customer.
        if (!availableRep.isEmpty()) {
            ChatSession newSession = new ChatSession(customer, availableRep.removeFirst());
            chatSessions.addLast(newSession);
            // Outputs that a representative has been assigned to the customer.
            System.out.println("RepAssignment " + customer + " " + newSession.getRep() + " " + formatTime(requestTime));
        }
        // 2. If a representative is not available, we check to see if the customer would like to be put on hold (wait) or try again later.
        else {
            if (waitOrLater.equals("wait")) {
                CustomerOnHold newCustomer = new CustomerOnHold(requestTime, customer);
                customersOnHold.addLast(newCustomer);
                // Outputs that a customer has been put on hold if there was no available representatives (and the customer wanted to wait).
                System.out.println("PutOnHold " + customer + " " + formatTime(requestTime));
            }
            else if (waitOrLater.equals("later")) {
                // Outputs that a customer chose to try again at a later time.
                System.out.println("TryLater " + customer + " " + formatTime(requestTime));
            }
        }
    }
    
    /** Removes a customer from the holding list, calculates their wait time, and
     *  compares it to the max wait time.
     *  
     *  @param quitOnHoldTime  the time the customer exited the holding list
     *  @param customer  the name of the customer
     */
    public static void quitOnHold(int quitOnHoldTime, String customer) {
        // Creates a dummy customerOnHold object and searches for the customer that wants to quit on hold.
        CustomerOnHold d = new CustomerOnHold(0, customer);
        CustomerOnHold removedCustomer = customersOnHold.searchAndRemove(d);
        
        // Checks to see if the customer was found (removed), then gets the customer's 
        if (removedCustomer != null) {
            int requestTime = removedCustomer.getRequestTime();
            calculateMaxWaitTime(requestTime, quitOnHoldTime);
            
            // Outputs that a customer chose quit waiting for a representative and exit the holding list.
            System.out.println("QuitOnHold " + formatTime(quitOnHoldTime) + " " + customer);
        }
        // Does nothing if no customer of the given name was found.
        else {
            // System.out.println("Error: Customer " + customer + " not found on hold.");
        }
    }
    
    /** Terminates a chat session between a customer and a representative. If there
     *  are customers on hold, the next available representative gets assigned to
     *  a chat session with that customer.
     * 
     *  @param customer  the name of the customer ending the chat session
     *  @param endTime  the time the chat session is being ended
     */
    public static void chatEnded(String customer, String rep, int endTime) {
        // Creates a dummy chat session object and searches for the current chat session that we want to end.
        ChatSession d = new ChatSession(customer, rep);
        ChatSession currentSession = chatSessions.searchAndRemove(d);
        
        // Checks to see if we found the current session we are trying to end
        if (currentSession != null) {
            // Checks to see if there are any customers on hold, then assings the newly available representative to the first customer pulled
            // from the hold list. The total time the customer waited on the hold list is also calculated and compared with the longest wait time.
            if (!customersOnHold.isEmpty()) {
                CustomerOnHold c = customersOnHold.removeFirst();
                ChatSession newSession = new ChatSession(c.getCustomerName(), rep);
                chatSessions.addLast(newSession);
                int requestTime = c.getRequestTime();
                calculateMaxWaitTime(requestTime, endTime);
            
                // Outputs that the current chat session has ended and a representative is assinged to the new customers off the hold list (only if
                // there are customers waiting on hold).
                System.out.println("ChatEnded " + customer + " " + rep + " " + formatTime(endTime));
                System.out.println("RepAssignment " + c.getCustomerName() + " " + rep + " " + formatTime(endTime));
            }
            // If the holding list is empty, the chat session ends and the representative from the ended chat session becomes available.
            else {
                availableRep.addLast(rep);
            
                // Outputs that a chat session has ended between a customer and a representative.
                System.out.println("ChatEnded " + customer + " " + rep + " " + formatTime(endTime));
            }
        }
        // Does nothing if the current chat session between the given customer and rep could not be found.
        else {
            // System.out.println("Error: Chat session between " + customer + " and " + rep + " does not exist.");
        }
    }
    
    /** Prints a list of the available representatives.
     * 
     *  @param printTime  the time the available representatives list is being printed out
     */
    public static void printAvailableRepList(int printTime) {
        // Outputs the list of available representatives.
        System.out.println("AvailableRepList " + formatTime(printTime) + " " + availableRep.toString());
    }
    
    /** Prints the maximum time a customer waited on the holding list.
     * 
     *  @param printTime  the time the maximum wait time is being printed out
     */
    public static void printMaxWaitTime(int printTime) {
        // Outputs the maximum time a customer had to wait in the holding list.
        System.out.println("MaxWaitTime " + formatTime(printTime) + " " + formatTime(maxWaitTimeSoFar));
    }
    
    /** The main method (entry point) for the program. Adds the five representatives to their assigned list and reads inputs from a given file.
     * 
     *  @param args  an array of Strings passed into the program (in this case a single String element, the input file name).
     *  @throws FileNotFoundException  if the attempt to access the file through args fails.
     */
    public static void main(String[] args)
    {
        try {
            // Checks if there were any arguments passed into the program
            if (args.length == 0) {
                throw new FileNotFoundException("Enter a file name!");
            }
            
            // Scanner variable to handle the input file
            Scanner s = new Scanner(new File(args[0]));
        
            // Adds the five representatives, Alice, Bob, Carol, David, and Emily, to the available representatives linked list.
            availableRep.addLast("Alice");
            availableRep.addLast("Bob");
            availableRep.addLast("Carol");
            availableRep.addLast("David");
            availableRep.addLast("Emily");
        
            // Reads each line from the input file
            while (s.hasNextLine()) {
                // Puts an entire line into a string variable
                String line = s.nextLine();
            
                // Reads the text stored in each line
                Scanner lineScanner = new Scanner(line);
            
                // If a line exists, the first word in the line, the method to be called, is read and stored.
                if (lineScanner.hasNext()) {
                    String method = lineScanner.next();
                
                    // If the method to be called is "chatEnded," the parameters customer, rep, and endTime are read and stored.
                    // Then ChatEnded is called if the time input is valid.
                    if (method.equals("ChatEnded")) {
                        String customer = lineScanner.next();
                        String rep = lineScanner.next();
                        int endTime = lineScanner.nextInt();
                        if (isValidTime(endTime)) {
                            chatEnded(customer, rep, endTime);
                        }
                    }
                    // If the method to be called is "chatRequest," the parameters requestTime, customer, and waitOrLater are read and stored.
                    // Then ChatRequest is called if the time input is valid.
                    else if (method.equals("ChatRequest")) {
                        int requestTime = lineScanner.nextInt();
                        String customer = lineScanner.next();
                        String waitOrLater = lineScanner.next();
                        if (isValidTime(requestTime)) {
                            chatRequest(requestTime, customer, waitOrLater);
                        }
                    }
                    // If the method to be called is "quitOnHold," prameters quitOnHoldTime and customer are read and stored.
                    // Then QuitOnHold is called if the time input is valid.
                    else if (method.equals("QuitOnHold")) {
                        int quitOnHoldTime = lineScanner.nextInt();
                        String customer = lineScanner.next();
                        if (isValidTime(quitOnHoldTime)) {
                            quitOnHold(quitOnHoldTime, customer);
                        }
                    }
                    // If the method to be called is "printAvailableRepList," the parameter printTime is read and stored.
                    // Then PrintAvailableRepList is called if the time input is valid.
                    else if (method.equals("PrintAvailableRepList")) {
                        int printTime = lineScanner.nextInt();
                        if (isValidTime(printTime)) {
                            printAvailableRepList(printTime);
                        }
                    }
                    // If the method to be called is "printMaxWaitTime," the parameter printTime is read.
                    // Then PrintMaxWaitTime is called if the time input is valid.
                    else if (method.equals("PrintMaxWaitTime")) {
                        int printTime = lineScanner.nextInt();
                        if (isValidTime(printTime)) {
                            printMaxWaitTime(printTime);
                        }
                    }
                }
            }
        }
        // An exception that catches if the file to be read is not found.
        catch (FileNotFoundException ex){
            System.err.println("Error: File not found. " + ex.getMessage());
        }
    }
}
