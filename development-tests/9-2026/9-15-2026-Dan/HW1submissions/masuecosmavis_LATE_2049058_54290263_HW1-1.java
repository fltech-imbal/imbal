/*
  Author: Clarence Masuecos
  Email: mmasuecos2025@my.fit.edu
  Course: CSE 2010
  Section: 3

  Program replicates online store's chat service. 
  manages hold queues, assignments, and customer requests. 
  reads events from chronological order from an input file and outputs the sequence using singly linked lists.

 */


import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW1{

    /*
      store customer name and timestap of chat request.
    */

    static class Customer{
        String name;
        String requestTime;
        public Customer(String name, String requestTime){
            this.name = name;
            this.requestTime = requestTime;
        }
    }
/*
      finds a representative for the customer to chat with.
    */
    static class ChatSesh{
        String customerName;
        String repName;

        public ChatSesh(String customerName, String repName){
            this.customerName = customerName;
            this.repName = repName;
        }
    }


    static int maxWaitTimeMins = 0;

/*
      main execution method that can initialize queues, read the input files by line, gorup events by time 
      then processes them according to priority. expects input file. name at index 0
    */

    public static void main(String[] args){
        /* description of variables */
        // availableReps: tracks representatives that are watiting to start a chat
        // customerHold: queue of customer waiting for a representative
        // activeChats: tracks conversations between customers and representatives.
        // outputBuffer: array of linked lists that sort events by priority with the same timestamp.
        // pendingInput: stores all lines from the input file before it processes

        /* check for valid file name */
        if (args.length == 0){
            System.out.println("Please provide a file name for input! ");
            return;
        }
/* initialize que of representatives in appropriate order */
        SinglyLinkedList<String> availableReps = new SinglyLinkedList<>();
        availableReps.addLast("Alice");        availableReps.addLast("Alice");
        availableReps.addLast("Bob");
        availableReps.addLast("Carol");
        availableReps.addLast("David");
        availableReps.addLast("Emily");
/* state trackign lists for customers and put out priority buffers. */
        SinglyLinkedList<Customer> customerHold = new SinglyLinkedList<>();
        SinglyLinkedList<ChatSesh> activeChats = new SinglyLinkedList<>();
    
        SinglyLinkedList<String>[] outputBuffer = new SinglyLinkedList[9];
        for(int i = 1; i<=8; i++){
            outputBuffer[i] = new SinglyLinkedList<>();

        }
/* grab and read all lines from file into a pending list */
        try{
            Scanner scanner = new Scanner(new File(args[0]));
            SinglyLinkedList<String> pendingInput = new SinglyLinkedList<>();

            while (scanner.hasNextLine()){
                
                String line = scanner.nextLine().trim();

                // System.out.println("read: " + line);

                if(!line.isEmpty()){
                    pendingInput.addLast(line);
                }
            }

            scanner.close();
/* work through pending input chronological events by one timestap */
            while(!pendingInput.isEmpty()){
                String curTimestamp = extractTime(pendingInput.first());
                SinglyLinkedList<String> curEvents = new SinglyLinkedList<>();
/* group together all events that have the same timestamp */
                while(!pendingInput.isEmpty() && extractTime(pendingInput.first()).equals(curTimestamp)){
                    curEvents.addLast(pendingInput.removeFirst());
                }
/* iterate through priority events 1-8 to process them in correct order. */
                for (int p = 1; p<=8; p++){
                    int size = curEvents.size();
                    for(int i = 0; i < size; i++){
                        String event = curEvents.removeFirst();
                        /* process the event if it matches the priority  */
                        if(getEventPriority(event) == p){
                            String[] parts = event.split(" ");
/* free the representative, record the end, then check the hold queue for remaining*/
                            if(p == 1){
                                String customer = parts[1];
                                String rep = parts[2];
                                removeActChat
                                (activeChats, customer);
                                    availableReps.addLast(rep);
                                    outputBuffer[1].addLast("ChatEnded " + customer + " " + rep + " " + curTimestamp);
                            
                                if(!customerHold.isEmpty()){


                                    Customer c = customerHold.removeFirst();
                                    String assignedRep = availableReps.removeFirst();

                                // System.out.println("assigning " + assignedRep + " to " + c.name);

                                    activeChats.addLast(new ChatSesh(c.name, assignedRep));
                                
                                    int wait = parseTime(curTimestamp) - parseTime(c.requestTime);
                                    if(wait > maxWaitTimeMins) 
                                        maxWaitTimeMins = wait;

                                    outputBuffer[3].addLast("RepAssignment " + c.name + " " + assignedRep + " " + curTimestamp);
                                }
                            }
/* assign a representative immediately or wait */
                            else if (p == 2) { 
                                String customer = parts[2];
                                String waitOrLater = parts[3];
                               
                                outputBuffer[2].addLast("ChatRequest " + curTimestamp + " " + customer + " " + waitOrLater);
    
                                if (!availableReps.isEmpty()) {
                                    String rep = availableReps.removeFirst();
                                    activeChats.addLast(new ChatSesh(customer, rep));
                                    outputBuffer[3].addLast("RepAssignment " + customer + " " + rep + " " + curTimestamp);
                                }else {
                                    if (waitOrLater.equals("wait")) {
                                        customerHold.addLast(new Customer(customer, curTimestamp));
                                        outputBuffer[4].addLast("PutOnHold " + customer + " " + curTimestamp);
                                    } else {
                                        outputBuffer[5].addLast("TryLater " + customer + " " + curTimestamp);
                                    }
                                }
                            }  
                            
                            else if (p == 6) { 
                                String customer = parts[2];
                                Customer c = removeholdcustomer(customerHold, customer);
                                if (c != null) {
                                    int wait = parseTime(curTimestamp) - parseTime(c.requestTime);
                                    if (wait > maxWaitTimeMins) maxWaitTimeMins = wait;
                                }
                                outputBuffer[6].addLast("QuitOnHold " + curTimestamp + " " + customer);
                            } 
                            /* print the representatives that are on queue for a chat */
                            else if (p == 7) { 
                                String listStr = "";
                                int repCount = availableReps.size();
                                for (int j = 0; j < repCount; j++) {
                                    String r = availableReps.removeFirst();
                                    listStr += " " + r;
                                    availableReps.addLast(r);
                                }
                                outputBuffer[7].addLast("AvailableRepList " + curTimestamp + listStr);
                            } 
                            /* format/print the longest recorded wait */
                            else if (p == 8) {
                                outputBuffer[8].addLast("MaxWaitTime " + curTimestamp + " " + formatTime(maxWaitTimeMins));
                            }
                        } else {
                            /* re que if the priority level hasn't been reached yet */
                            curEvents.addLast(event);
                    
                        }
                    }
                }
/* Print the sorted output buffers to the screen for the current timestamp */
                for (int p = 1; p <= 8; p++) {
                    while (!outputBuffer[p].isEmpty()) {
                        System.out.println(outputBuffer[p].removeFirst());
                    }
                }
            }

            
        }

        catch (FileNotFoundException e) {
            System.out.println("Input file not found: " + args[0]);
        }
    }

/*
      searches for a customer in the hold queue by cycling through a temporary list,
      then removes without a built in delete method. 
      parameters are the list of the customer hold queue and
      the name of the customer to remove. 
      it returns the customer object if found, otherwise null.
    */
    public static Customer removeholdcustomer(SinglyLinkedList<Customer> list, String name) {
        SinglyLinkedList<Customer> temp = new SinglyLinkedList<>();
        Customer found = null;
        while (!list.isEmpty()) {
            Customer c = list.removeFirst();
            if (c.name.equals(name)) {
                found = c;

                // System.out.println("pulling out " + name + "");
            } else {
                temp.addLast(c);
            }
        }
        while (!temp.isEmpty()) {
            list.addLast(temp.removeFirst());
        }
        return found;
    }
    /*
      removes a finished chat session from the active list by popping the end elements to
      a temporary list then pushes them back, 
      which passes by the target customer. 
      parameters are the list of
      the active chat queue, and customer name, the customer ending the chat.
    */

    public static void removeActChat
    (SinglyLinkedList<ChatSesh> list, String customerName) {
        SinglyLinkedList<ChatSesh> temp = new SinglyLinkedList<>();
        while (!list.isEmpty()) {
            ChatSesh s = list.removeFirst();
            if (!s.customerName.equals(customerName)) {
                temp.addLast(s);
            }
        }
        while (!temp.isEmpty()) {
            list.addLast(temp.removeFirst());
        }
    }

/*
    description of the exactTime method:  
    only grabs specific time of the event that has occured
    parses a raw input string to determine the timestamp of the event, 
      the parameters are the lines/raw strings
      from the input file, and returns the HMMM timestamp as a string.
    */
    public static String extractTime(String line) {
        String[] parts = line.split(" ");
        if (parts[0].equals("ChatEnded")) return parts[3];
        return parts[1];
    }
/*
      description of getEventPriority method:
      assigns a number priority to an event based on the output of the line
      order requered when events are sharing the exact same timestamp.
      the parameters are the line/raw event string
      returns an integer from 1-9 representing processing priority.
    */
    public static int getEventPriority(String line) {
        if (line.startsWith("ChatEnded")) return 1;
        if (line.startsWith("ChatRequest")) return 2;
        if (line.startsWith("QuitOnHold")) return 6;
        if (line.startsWith("PrintAvailableRepList")) return 7;
        if (line.startsWith("PrintMaxWaitTime")) return 8;
        return 9;
    }
    /*
      description of parseTime function:
      converts an HHMM string representation into integer minutes (total)
      to make ofr easy wait time subtraction and calculation
      the parameters are t, which is the string representation of time in HHMM
      returns an integer total of minutes since 00:))
    */

    public static int parseTime(String t) {
        return Integer.parseInt(t.substring(0, 2)) * 60 + Integer.parseInt(t.substring(2, 4));
    }
    /*
      description of the formatTime function:

      converts total integer minutes back into a standard HHMM formatted string
      with leading zeroes if necessary for time.
      the parameters are the minutes - the total minutes to format
      and returns a string formatted as HHMM.
    */

    public static String formatTime(int mins) {
        return String.format("%02d%02d", mins / 60, mins % 60);
    }

}