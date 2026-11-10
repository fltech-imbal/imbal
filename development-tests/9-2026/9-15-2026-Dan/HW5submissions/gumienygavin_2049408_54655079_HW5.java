/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
  This file uses a skip list file to manage a timeline of events given from an input file.
  It reads a list of possible events to add or delete given the specified input action, and
  it can display certain events between a certain time frame, or up to or beyond a specific
  date. The entire skip list or events can also be displayed. One input file name is taken 
  as a command-line argument and each input line is read within the file.
  
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

public class HW5
{
    /** Prints out the list of events found within a range.
     *  @param action  the action to be printed out
     *  @param events  the list of events found within a range
     */
    private static void printRangeResult(String action, List<SkipList.Node> events) {
        // Print action first.
        System.out.print(action);
        
        // If there are no events in the list print out none.
        if (events.isEmpty()) {
            System.out.print(" none");
        }
        // Otherwise for each node entry in the events list we print out the date and event.
        else {
            for (SkipList.Node entry : events) {
                System.out.print(" " + entry.getDate() + ":" + entry.getEvent());
            }
        }
        System.out.println();
    }
    
    /** The main method (entry point) for the program.
     * 
     *  @param args  an array of Strings passed into the program (in this case a single input file containing the list of actions.).
     *  @throws FileNotFoundException  if the attempt to access the file through args fails.
     */
    public static void main(String[] args)
    {
        try {
            // Checks if there were any arguments passed into the program
            if (args.length < 1) {
                throw new FileNotFoundException("Incorrect file entry! Enter the actions file.");
            }
            
            // Scanner variable to read the actions file.
            Scanner s = new Scanner(new File(args[0]));
            
            // Declare new skip list
            SkipList skipList = new SkipList();
            
            // Reads each line from the input file.
            while (s.hasNextLine()) {
                // Stores and entire line in a string variable.
                String line = s.nextLine();
                
                // Reads the text stored in each line.
                Scanner lineScanner = new Scanner(line);
                
                // If a line exists, the first word in the line, the action, is read and stored.
                if (lineScanner.hasNext()) {
                    String action = lineScanner.next();
                    
                    // If the action is "DisplayEvent," the parameter date is read and stored.
                    if (action.equals("DisplayEvent")) {
                        String date = lineScanner.next();
                        String event = skipList.get(date);   // get the event with the associated date
                        
                        // Print event with date if we found an event.
                        if (event != null) {
                            System.out.println("DisplayEvent " + date + " " + event);
                        }
                        // Print none with date if we did not find an event.
                        else {
                            System.out.println("DisplayEvent " + date + " none");
                        }
                    }
                    // If the action is "AddEvent," the parameters date and event are read and stored.
                    else if (action.equals("AddEvent")) {
                        String date = lineScanner.next();
                        String event = lineScanner.next();
                        skipList.put(date, event);          // add to skip list
                        // Output that we added the event
                        System.out.println("AddEvent " + date + " " + event + " success");
                    }
                    // If the action is "DeleteEvent," the parameter date is read and stored.
                    else if (action.equals("DeleteEvent")) {
                        String date = lineScanner.next();
                        String removed = skipList.remove(date);  // remove from skip list
                        // Output success if removed is not null
                        if (removed != null) {
                            System.out.println("DeleteEvent " + date + " success");
                        }
                        // Otherwise output error
                        else {
                            System.out.println("DeleteEvent " + date + " noDateError");
                        }
                    }
                    // If the action is "DisplayEventsBetweenDates," the parameters startDate and endDate are read and stored.
                    else if (action.equals("DisplayEventsBetweenDates")) {
                        String startDate = lineScanner.next();
                        String endDate = lineScanner.next();
                        
                        // List to store the events between the dates
                        List<SkipList.Node> events = skipList.subMap(startDate, endDate);
                        printRangeResult("DisplayEventsBetweenDates " + startDate + " " + endDate, events);     // Prints out the results
                    }
                    // If the action is "DisplayEventsFromStartDate," the parameter startDate is read and stored.
                    else if (action.equals("DisplayEventsFromStartDate")) {
                        String startDate = lineScanner.next();
                        
                        // List to store the events from the start date.
                        List<SkipList.Node> events = skipList.subMap(startDate, SkipList.POS_INF);
                        printRangeResult("DisplayEventsFromStartDate " + startDate, events);        // Prints out the results
                    }
                    // If the action is "DisplayEventsToEndDate," the parameter endDate is read and stored.
                    else if (action.equals("DisplayEventsToEndDate")) {
                        String endDate = lineScanner.next();
                        
                        // List to store the events to the end date.
                        List<SkipList.Node> events = skipList.subMap(SkipList.NEG_INF, endDate);
                        printRangeResult("DisplayEventsToEndDate " + endDate, events);              // Prints out the results
                    }
                    // If the action is "DisplayAllEvents," .
                    else if (action.equals("DisplayAllEvents")) {
                        // List to store all of the events in the skip list.
                        List<SkipList.Node> events = skipList.subMap(SkipList.NEG_INF, SkipList.POS_INF);
                        printRangeResult("DisplayAllEvents", events);           // Prints out the results
                    }
                    // If the action is "PrintSkipList," printSkipList is called.
                    else if (action.equals("PrintSkipList")) {
                        System.out.println("PrintSkipList");
                        skipList.printSkipList();               // print the skip list
                    }
                }
            }
        }
        catch (FileNotFoundException ex){
            System.err.println("Error: File not found. " + ex.getMessage());
        }
    }
}
