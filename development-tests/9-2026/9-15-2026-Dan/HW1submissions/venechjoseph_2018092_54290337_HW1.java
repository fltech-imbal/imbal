/*

  Author: Joseph Venech
  Email: jvenech2025@my.fit.edu
  Course: CSE 2010
  Section: 1
  Description of this file: The purpose of this file is to manage chat requests. 
  
  I did use the Java API documentation for help understanding java.io.File and Comparable
  
  

 */
import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class HW1
{
    //convert timestamp from string to int
    static int HHMM (String timeStr) {
        int hhmm = Integer.parseInt(timeStr);
        int hours = hhmm / 100;
        int minutes = hhmm % 100;
        return hours * 60 + minutes;
    }
    
    //Store customer put on hold
    static class PutOnHold {
        String name;
        String timeStr;
        int time;
        
        PutOnHold(String name, String timeStr, int time) {
            this.name = name;
            this.timeStr = timeStr;
            this.time = time;
        }
    }
    
    //Stores Chat Sessions   
    static class ChatSession {
        String customer;
        String rep;
        
        ChatSession(String customer, String rep) {
            this.customer = customer;
            this.rep = rep;
        }
    }
    
    //Store events for ordering
    static class Event implements Comparable<Event>{
        int timeMinutes;
        int priority; //order according assignment is ChatEnded 1 , ChatRequest 2 , Rep Assignment 3 , PutOnHold 4 , TryLater 5 , QuitonHold 6 , AvailableRepList 7 , MaxWaitTime 8
        int inputOrder;
        String outputLine;
        
        Event(int timeMinutes, int priority, int inputOrder, String outputLine) {
            this.timeMinutes = timeMinutes;
            this.priority = priority;
            this.inputOrder = inputOrder;
            this.outputLine = outputLine;
        }
        
        @Override
        public int compareTo(Event other) {
            if (this.timeMinutes != other.timeMinutes) { //Checks if event occur at same time
               return Integer.compare(this.timeMinutes, other.timeMinutes);
            }
            if (this.priority != other.priority) { //Checks if event has same priority (refer to field for order)
                return Integer.compare(this.priority, other.priority);
            }
            return Integer.compare(this.inputOrder, other.inputOrder); //Compares by order they were inputted if occured at same time and same priority
        }
    }

    public static void main(String[] args)
    {
    //Create available rep list   
    SinglyLinkedList<String> AvailableRepList = new SinglyLinkedList<>();
    //Adds Reps to availability list
    AvailableRepList.addLast("Alice");
    AvailableRepList.addLast("Bob");
    AvailableRepList.addLast("Carol");
    AvailableRepList.addLast("David");
    AvailableRepList.addLast("Emily");
    
    //Create list for customers on hold
    SinglyLinkedList<PutOnHold> customersOnHold = new SinglyLinkedList<>();
    //Create list for currently active chat sessions
    SinglyLinkedList<ChatSession> activeChats = new SinglyLinkedList<>();
    //Create list to store Events for output
    SinglyLinkedList<Event> outputEvents = new SinglyLinkedList<>();
    

    
    int maxWait = 0; // Current Max wait for customers
    int inputCounter = 0; //Used to ensure events are correctly ordered
    
    
    
    try (Scanner scanner = new Scanner(new File(args[0]))) {
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine().trim();
            if (line.isEmpty()) continue; //continue added to prevent ArrayIndexOutOfBounds errors with empty line
            
            String[] parts = line.split("\\s+"); //Splits line into 4 parts: command [0], time [1], name [2], action [3]
            String command = parts[0];
            
            switch (command) {
                case "ChatRequest": {
                    String timeStr = parts[1];
                    String customer = parts[2];
                    String waitOrLater = parts[3];
                    int time = HHMM(timeStr); //convert time from string to int
                    
                    outputEvents.addLast(new Event(time, 2, inputCounter++, "ChatRequest " + timeStr + " " + customer + " " + waitOrLater));
                    
                    if (!AvailableRepList.isEmpty()){ //There is a rep available
                        String assignedRep = AvailableRepList.removeFirst(); //removes first rep available for rep list
                        activeChats.addLast(new ChatSession(customer, assignedRep)); //add this chat to chat session list
                        outputEvents.addLast(new Event(time, 3, inputCounter++, "RepAssignment " + customer + " " + assignedRep + " " + timeStr)); //add event to event list
                    } else { // no rep available
                        if (waitOrLater.equals("wait")) {
                            //put customer on hold list
                            customersOnHold.addLast(new PutOnHold(customer, timeStr, time));
                            //add event to event list
                            outputEvents.addLast(new Event(time , 4, inputCounter++, "PutOnHold " + customer + " " + timeStr));
                        } else if (waitOrLater.equals("later")){
                            //add event to event list
                            outputEvents.addLast(new Event(time, 5, inputCounter++, "TryLater " + customer + " " + timeStr));
                        }
                    }
                    break;
                }
                case "ChatEnded": {
                    String customer = parts[1];
                    String rep = parts[2];
                    String endTime = parts[3];
                    int time = HHMM(endTime); //Change from string to int
                    
                    //remove from active chats
                    removeChat(activeChats, customer, rep);
                    //Add event to event list
                    outputEvents.addLast(new Event(time, 1, inputCounter++, "ChatEnded " + customer + " " + rep + " " + endTime));
                    //Add rep back to available list
                    AvailableRepList.addLast(rep);
                    
                    if (!customersOnHold.isEmpty()) { //Put on next customer on hold
                        PutOnHold next = customersOnHold.removeFirst(); //takes the next person in hold
                        String nextRep = AvailableRepList.removeFirst(); //removes first rep available for rep list
                        activeChats.addLast(new ChatSession(next.name, nextRep)); //add this chat to chat session list
                        outputEvents.addLast(new Event(time, 3, inputCounter++, "RepAssignment " + next.name + " " + nextRep + " " + time)); //add event to event list

                        int waitTime = time - next.time; //Calculates wait time
                        if(waitTime > maxWait) { //Checks if current wait time is greater than max wait time
                            maxWait = waitTime; //make current wait time to the new max wait
                        }
                        
                    }
                    break;
                }
                case "QuitOnHold": {
                    String endTime = parts[1];
                    String customer = parts[2];
                    int time = HHMM(endTime); //Change from String to int
                    
                    PutOnHold quitter = removeCustomerOnHold(customersOnHold, customer); //Remove customer from hold
                    if (quitter != null) { //in case remove wasn't possible
                    outputEvents.addLast(new Event(time, 6, inputCounter++, "QuitOnHold " + endTime + " " + customer)); //Add event to event list
                    int waitTime = time - quitter.time;
                    if (waitTime > maxWait) {
                        maxWait = waitTime;
                    }
                }
                    break;
                }
                case "PrintAvailableRepList": {
                    String printTime = parts[1];
                    int time = HHMM(printTime);
                    
                    String repList = "";
                    for (int i = 0; i < AvailableRepList.size(); i++) { //Cycles through available rep list
                        String rep = AvailableRepList.removeFirst(); //takes rep first in line
                        repList = repList + rep + " "; //add first rep to list
                        AvailableRepList.addLast(rep); //add first rep to back
                    }
                    
                    outputEvents.addLast(new Event(time, 7, inputCounter++, "AvailableRepList " + printTime + " " + repList)); //add event to event list
                    break;
                }
                case "PrintMaxWaitTime": {
                    String printTime = parts[1];
                    int time = HHMM(printTime);
                    
                    outputEvents.addLast(new Event(time, 8, inputCounter++, "MaxWaitTime " + printTime + " " + maxWait));
                                        
                    break;
                }
            }
            
        }
    } catch (FileNotFoundException e) {
         System.err.println("File not found: " + args[0]);
         return;
    }
    
    outputEvents = sortEvents(outputEvents); //sort events
    
    while (!outputEvents.isEmpty()) { //Output Events
        System.out.println(outputEvents.removeFirst().outputLine);
    }
    
    } // main
    
    //locate and remove active chat session
    private static void removeChat(SinglyLinkedList<ChatSession> list, String customer, String rep) {
        
        for (int i = 0; i < list.size(); i++) { //Loops to find the right chat session to remove
            ChatSession session = list.removeFirst(); //remove from the front
            if (session.customer.equals(customer) && session.rep.equals(rep)) {
             break; //Correct node found
            } else {
                list.addLast(session); //Wasn't the right session and added to the back of the list
            }
        }
    }
    
    //locate and remove customer from hold
    private static PutOnHold removeCustomerOnHold(SinglyLinkedList<PutOnHold> list, String customerName) {
      PutOnHold removed = null;
      int n = list.size();
      for (int i = 0; i < n; i++) { //Loop to find right customer
          PutOnHold item = list.removeFirst(); //Remove from the front
          if (removed == null && item.name.equals(customerName)) {
              removed = item; //correct customer
            } else {
                list.addLast(item); //not correct and adds customer to back
            }
      }
      return removed;
    }
    
    //sort with insertion sort
    private static SinglyLinkedList<Event> sortEvents(SinglyLinkedList<Event> list) {
        SinglyLinkedList<Event> sorted = new SinglyLinkedList<>();
        while(!list.isEmpty()) {
            Event current = list.removeFirst();
            insertInOrder(sorted, current);
        }
        return sorted;
    }
    //used with sort method to sort them in order
    private static void insertInOrder(SinglyLinkedList<Event> list, Event event) {
        boolean inserted = false; //use to see if event been inserted in order
        
        for (int i = 0; i < list.size(); i++ ){
            Event head = list.removeFirst();
            if(!inserted && event.compareTo(head) < 0) {
                list.addLast(event);
                inserted = true;
            }
            list.addLast(head);
        }
        
        if (!inserted) {
            list.addLast(event);
        }
    }
}
