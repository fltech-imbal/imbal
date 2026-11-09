/*

  Author: Keidgiler Claimon
  Email: kclaimon2025@fit.edu
  Course: CSE2010
  Section: 9:30 am
  Description of this file: A skipList made to input and retrieve events and dates.

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW5
{

    /*
      Description of each method, including parameters 
    */
    public static void main(String[] args) {
	/* description of variables */  
      Scanner scanner;
      try { // a try catch exception for file use
          scanner = new Scanner(new File(args[0])); 
        } 
      catch(FileNotFoundException e) {
          System.out.println("File was not found"); 
          return;
      }   

      //MySkipList "lifeStory" is initialized.
      MySkipList lifeStory = new MySkipList();

      while (scanner.hasNext()) { 
        //int date, String event, and String action are initialized.
        int date = 0;
        String event = null; 
        String action = scanner.next(); 

        switch(action) { 
          //DisplayEvent obtains the event by using get(date)-- 
          //returns "none" if there is no event on that date..
          case "DisplayEvent":  
            date = scanner.nextInt(); 
            if (lifeStory.get(date) != null) {
              event = lifeStory.getToString(date); 
              System.out.println("DisplayEvent " + event);
            } else {
              System.out.println("DisplayEvent " + date + "none"); 
            }
             
            
          break; 
          //AddEvent adds a specific day into the event. It will print "success" 
          // once it finishes.
          case "AddEvent": 
            date = scanner.nextInt(); 
            event = scanner.next(); 
            lifeStory.put(date, event); 
            
            System.out.println("AddEvent " + date + " " + event + " success"); 
          break;  

          //DeleteEvent deletes the event by first checking if the event is there,
          //Then removes the event. using remove(date).
          case "DeleteEvent": 
            date = scanner.nextInt(); 
            if (lifeStory.get(date) != null) {
              lifeStory.remove(date); 
              System.out.println("DeleteEvent " + date + "success");
            } else {
              System.out.println("DeleteEvent" + date + "noDateError");
            }
            
            
          break; 
          
          //DisplayAllEvents uses displayNodes.
          case "DisplayAllEvents": 
            event = lifeStory.displayNodes(); 
            System.out.println("DisplayAllEvents " + event); 
          break; 
          
          //DisplayAllEventsBetweenDates uses date1 and date2 as a 
          // floor and ceiling for the subMap to use.
          case "DisplayAllEventsBetweenDates": 
            int date1 = scanner.nextInt(); 
            int date2 = scanner.nextInt();
            String betweenList = lifeStory.subMap(date1, date2); 

            System.out.println("DisplayAllEventsBetweenDates " + betweenList); 
          break; 
          
          //DisplayAllEventsFromStartDate uses date and Integer.MAX_VALUE as a 
          //floor and ceiling respectively.
          case "DisplayAllEventsFromStartDate": 
            date = scanner.nextInt(); 
            event = lifeStory.subMap(date, Integer.MAX_VALUE); 
            
            System.out.println("DisplayAllEventsFromStartDate " + event);

            
          break;  


          //DisplayEventsToEndDate displays events from Integer.MIN_VALUE to date.
          case "DisplayEventsToEndDate": 
            date = scanner.nextInt(); 
            event = lifeStory.subMap(Integer.MIN_VALUE, date); 

            System.out.println("DisplayEventsToEndDate " + event); 
          break; 

          case "PrintSkipList": 
            System.out.println(lifeStory.toString()); 
          break;
            

        }

      }

      

	
      scanner.close();
    }

}
