/*

  Author: Joseph Venech
  Email: jvenech2025@my.fit.edu
  Course: CSE2010
  Section: 1
  Description of this file: Manages events using a skip list 

 */

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class HW5
{
    //Stores events using their dates as keys and events as values
    private SkipListMap<Integer, String> events = new SkipListMap<>();
    /*
      Description of each method, including parameters 
    */
    public static void main(String[] args)
    {
        
        HW5 program = new HW5();
        
        //Get input file
        String eventsFile = args[0];
        
        try (Scanner eventsScanner = new Scanner(new File(eventsFile))) {
        
            //Read events
            while (eventsScanner.hasNextLine()) {
                String line = eventsScanner.nextLine().trim();
            
                //Skip empty lines
                if (line.length() == 0) {
                    continue;
                }
            
                //Do action base off input
                program.processAction(line);
            
            }
        } catch (FileNotFoundException e) {
            System.err.println("Cannot find input file: " + args[0]);
        }
        }
        
    //Process each action
    private void processAction(String line) {
        Scanner scanner = new Scanner(line);
        
        String action = scanner.next();            
        try {
            switch (action) {
                case "DisplayEvent": {
                //Read date and seach for event
                int date = Integer.parseInt(scanner.next());
                String value = events.get(date);
                
                //print event if found otherwise none
                System.out.println(action + " " + formatDate(date) + " " + (value == null ? "none" : value));
                break;
                }

                case "AddEvent": {
                    //Read the date and complete event
                    int date = Integer.parseInt(scanner.next());
                    String value = scanner.hasNextLine() ? scanner.nextLine().trim() : "";
                    
                    //Insert event
                    events.put(date,value);

                    System.out.println(action + " " + formatDate(date) + " " + value + " success");
                    break;
                }

                case "DeleteEvent": {
                    //Read date and attempt to remove
                    int date = Integer.parseInt(scanner.next());
                    String old = events.remove(date);

                    //Prints action or error if does not exist
                    System.out.println(action + " " + formatDate(date) + " " + (old == null ? "noDateError" : "success"));
                    break;
                }

                case "DisplayEventsBetweenDates": {
                    //Read the endpoint dates
                    int start = Integer.parseInt(scanner.next());
                    int end = Integer.parseInt(scanner.next());
                    
                    //Retrieve and print events within range
                    System.out.println(action + " " + formatDate(start) + " " + formatDate(end) + " " + formatEntries(events.subMap(start, end)));
                    break;
                }

                case "DisplayEventsFromStartDate": {
                    //read date
                    int start = Integer.parseInt(scanner.next());
                        
                    //Displays events from this date through end of year
                    System.out.println(action + " " + formatDate(start) + " " + formatEntries(events.subMap(start, 1232))); // one more day than the last day of the year (MAX)
                    break;
                }

                case "DisplayEventsToEndDate": {
                    //read date
                    int end = Integer.parseInt(scanner.next());

                    //Displays events from the start to ending date
                    System.out.println(action + " " + formatDate(end) + " " + formatEntries(events.subMap(0, end)));
                    break;
                }

                case "DisplayAllEvents": {
                    //Displays every event in date order
                    System.out.println(action + " " + formatEntries(events.subMap(0, 1232)));
                    break;
                }

                case "PrintSkipList": {
                    //print skip list, including levels
                    System.out.println(action);
                    events.printSkipList();
                    break;
                }

                default:
                    System.err.println("Unknown action: " + action);
            }
        } catch (Exception e) {
            System.err.println("Error cannot process: " + line);
        }
        scanner.close();
        }
    
    //Converts date int to string
    private static String formatDate(int date) {
        return String.format("%04d", date);
    }
    
    //Formats entries as Date:event pairs separated by spaces
    private static String formatEntries(Iterable<SkipListMap.Entry<Integer, String>> entries) {

        StringBuilder result = new StringBuilder();

        for (SkipListMap.Entry<Integer, String> entry : entries) {
            // Separate entries with a space, but not before the first one.
            if (result.length() > 0) {
                result.append(" ");
            }

            // Append the formatted date and its event description.
            result.append(formatDate(entry.getKey())).append(":").append(entry.getValue());
        }

        // Return none when the collection is empty.
        return result.length() == 0 ? "none" : result.toString();
    }
}

