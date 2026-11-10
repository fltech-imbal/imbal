// Author: Adrian Santana
// Email: santanaa2025@my.fit.edu
// Section: 1
// Description: The program shown below contains the main method which will evaluate the input file containing the actions
// and uses methods from SkipList.java to output the messages for each case.
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW5 {
    public static void main(String[] args) throws FileNotFoundException {
        // Create a file and scanner object to read the input file
        File inputFile = new File(args[0]);
        Scanner scanner = new Scanner(inputFile);
        SkipList events = new SkipList();

        // Create a while loop that reads each line of the scanner
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            // Use the split tool to break each of the input file to be used independently
            String[] inputParts = line.split(" ");
            String action = inputParts[0];
            // If the requested action is "AddEvent", use put() from SkipList.java to add the event to the skip list
            // and print the message
            if (action.equals("AddEvent")) {
                int eventDate = Integer.parseInt(inputParts[1]);
                String event = inputParts[2];
                events.put(eventDate, event);
                System.out.println(line + " success");
                // If the requested action is "DisplayEvent", use get() from SkipList.java to find the event
                // and print the message
            } else if (action.equals("DisplayEvent")) {
                int eventDate = Integer.parseInt(inputParts[1]);
                String event = events.get(eventDate);
                // If event is null, print the required message with "none" following it
                if (event == null) {
                    System.out.println(line + " none");
                } else {
                    System.out.println(line + " " + event);
                }
                // If the action is "DeleteEvent", use the remove() method from SkipList.java to remove the event
                // from the skip list and print the required message
            } else if (action.equals("DeleteEvent")) {
                int eventDate = Integer.parseInt(inputParts[1]);
                String event = events.remove(eventDate);
                // If event is null, print the message with "noDateError" following it
                if (event == null) {
                    System.out.println(line + " noDateError");
                } else {
                    System.out.println(line + " success");
                }
                // If the requested action is "DisplayEventsBetweenDates", use subMap() from SkipList.java to find
                // the events between the dates and print the required message
            } else if (action.equals("DisplayEventsBetweenDates")) {
                int startDate = Integer.parseInt(inputParts[1]);
                int endDate = Integer.parseInt(inputParts[2]);
                String result = events.subMap(startDate, endDate);
                System.out.println(line + " " + result);
                // If the requested action is "DisplayEventsFromStartDate", use subMap() to find the events from a specific date
                // to the end of the year and print the message
            } else if (action.equals("DisplayEventsFromStartDate")) {
                int startDate = Integer.parseInt(inputParts[1]);
                String result = events.subMap(startDate, 1231);
                System.out.println(line + " " + result);
                // If the action is to "DisplayEventsToEndDate", use subMap() to find the events from the beginning of
                // the year to the specified date and print the events
            } else if (action.equals("DisplayEventsToEndDate")) {
                int endDate = Integer.parseInt(inputParts[1]);
                String result = events.subMap(101, endDate);
                System.out.println(line + " " + result);
                // If the requested action is "DisplayAllEvents", use subMap() to find all the events throughout the year
                // and print them
            } else if (action.equals("DisplayAllEvents")) {
                String result = events.subMap(101, 1231);
                System.out.println(line + " " + result);
                // If the requested action is to "PrintSkipList", use the toString() method to print the entire skip list
            } else if (action.equals("PrintSkipList")) {
                System.out.println(line);
                System.out.println(events.toString());
            }
        }

    }
}