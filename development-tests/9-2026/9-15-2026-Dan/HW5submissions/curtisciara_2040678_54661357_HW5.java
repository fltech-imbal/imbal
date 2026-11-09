/* Author: Ciara Curtis
   Email: ccurtis2025@my.fit.edu
   Course: CSE2010
   Section: 4
   Description of file: Homework 5 submission that uses the SkipList
	class to create a traversable linked list that holds dates as
	keys and events as values. This class takes a file of specific
        requests that will be handled using the skip list. */

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class HW5 {

  public static void main(String[] args) throws FileNotFoundException{
    //scanner to read the commands and the data/events
    Scanner entries = new Scanner(new File(args[0]));

    //SkipList to hold the dates and events
    SkipList eventList = new SkipList();

    //while there are still lines to read
    while(entries.hasNext()) {
      String command = entries.next();
      if(command.equals("AddEvent")) {
        int date = entries.nextInt();
	String event = entries.next();
        eventList.put(date, event);
        System.out.printf("%s %04d %s success\n", command, date, event);
      }

      else if(command.equals("DisplayEvent")) {
	int date = entries.nextInt();
        String event = eventList.get(date);
        if(event == null) { event = "none"; }
        System.out.printf("%s %04d %s\n", command, date, event);
      }

      else if(command.equals("DeleteEvent")) {
	int date = entries.nextInt();
	String success = eventList.remove(date);
        if(success == null) success = "noDateError";
        else success = "success";
        System.out.printf("%s %04d %s\n", command, date, success);
      }

      else if(command.equals("PrintSkipList")) {
        System.out.println(command);
        eventList.toPrint();
      }

      else if(command.equals("DisplayEventsBetweenDates")) {
	int date1 = entries.nextInt();
        int date2 = entries.nextInt();
        System.out.printf("%s %04d %04d ", command, date1, date2);
        eventList.subMap(date1, date2);
      }

      else if(command.equals("DisplayEventsFromStartDate")) {
	int startDate = entries.nextInt();
        System.out.printf("%s %04d ", command, startDate);
        eventList.subMap(startDate, 4999);
      }

      else if(command.equals("DisplayEventsToEndDate")) {
	int endDate = entries.nextInt();
	System.out.printf("%s %04d ", command, endDate);
        eventList.subMap(1, endDate);
      }

      else if(command.equals("DisplayAllEvents")) {
        System.out.print(command + " ");
        eventList.printBottom();

      }
    } // end of while loop

  }
}
