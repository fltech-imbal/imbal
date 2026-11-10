/*
 * Author: Jackson Young
 * Email: young2025@my.fit.edu
 * Course: CSE 2010
 * Section: 1
 * Description of this file: Driver class for reading input files and implementing the SkipList class
*/
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
public class HW5 {

    // main method; used to read input files used in command-line arguments
    // then uses the input to build the SkipList
    public static void main(String[] args) {
        // First implement a Scanner objects for the input file
        // using a try-catch block to terminate the program if an error occurs during file reading
        Scanner scanner; // Declare scanner variable
        try {
            // Attempt to construct File and Scanner objects using input
            scanner = new Scanner(new File(args[0]));
        }
        // If the file cannot be read, safely terminate the program
        catch (FileNotFoundException e) {
            System.out.println("ERROR: FileNotFoundException.\n" + e.getMessage() +"\nTerminating Program.");
            return; // End program
        }

        // Setip skip list
        SkipList<String> skipList = new SkipList<>();

        // Process the input file using Scanner to read each token
        // one at a time to determine the action needed
        while (scanner.hasNext()) { // Keep going untill all input is processed
            String action = scanner.next(); // The first word in each line is the action

            // Determine the name of the action
            if (action.equals("DisplayEvent")) {
                // Reads an event from the skip list and prints it
                String date = scanner.next();
                String event = skipList.get(Integer.valueOf(date));
                System.out.print("DisplayEvent " + date + " ");
                if (event != null) System.out.println(event);
                else System.out.println("none");

            } else if (action.equals("AddEvent")) {
                // Adds an event to the skip list
                String date = scanner.next();
                String event = scanner.next();
                skipList.put(Integer.parseInt(date), event);
                System.out.println("AddEvent " + date + " " + event + " success");

            } else if (action.equals("DeleteEvent")) {
                // Remove an event and return its value
                String date = scanner.next();
                String removed = skipList.remove(Integer.parseInt(date));
                System.out.print("DeleteEvent " + date + " ");
                if (removed != null) System.out.println("success");
                else System.out.println("noDateError");

            } else if (action.equals("DisplayEventsBetweenDates")) {
                // Print the submap between two events
                String startDate = scanner.next();
                String endDate = scanner.next();
                System.out.print("DisplayEventsBetweenDates " + startDate + " " + endDate);
                ArrayList<String> events = skipList.subMap(Integer.parseInt(startDate), Integer.parseInt(endDate));
                if (events.isEmpty()) {
                    System.out.println(" none");
                } else {
                    for (String event : events) {
                        System.out.print(" " + event);
                    }
                    System.out.println();
                }

            } else if (action.equals("DisplayEventsFromStartDate")) {
                // Print the submap of start event to max_int
                String startDate = scanner.next();
                System.out.print("DisplayEventsFromStartDate " + startDate);
                ArrayList<String> events = skipList.subMap(Integer.parseInt(startDate), Integer.MAX_VALUE);
                if (events.isEmpty()) {
                    System.out.println(" none");
                } else {
                    for (String event : events) {
                        System.out.print(" " + event);
                    }
                    System.out.println();
                }

            } else if (action.equals("DisplayEventsToEndDate")) {
                // Print the submap of min_int event to end event
                String endDate = scanner.next();
                System.out.print("DisplayEventsToEndDate " + endDate);
                ArrayList<String> events = skipList.subMap(Integer.MIN_VALUE, Integer.parseInt(endDate));
                if (events.isEmpty()) {
                    System.out.println(" none");
                } else {
                    for (String event : events) {
                        System.out.print(" " + event);
                    }
                    System.out.println();
                }

            } else if (action.equals("DisplayAllEvents")) {
                // Print the submap of min_int to max_int
                System.out.print("DisplayAllEvents");
                ArrayList<String> events = skipList.subMap(Integer.MIN_VALUE, Integer.MAX_VALUE);
                if (events.isEmpty()) {
                    System.out.println(" none");
                } else {
                    for (String event : events) {
                        System.out.print(" " + event);
                    }
                    System.out.println();
                }

            } else if (action.equals("PrintSkipList")) {
                // Prints each layer of the skip list and every entry within
                System.out.println("PrintSkipList");
                skipList.print();
                
            }
        }
    }
}
