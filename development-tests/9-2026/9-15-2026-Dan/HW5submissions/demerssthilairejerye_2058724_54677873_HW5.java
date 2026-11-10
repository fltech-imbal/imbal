/*
Author: Jerye Demers-St.Hilaire
Email: Jdemerssthil2025@my.fit.edu
Course: CSE 2010
Section: 1
Description: Manages a timeline of events using a skip list.
*/

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;

public class HW5 {
 public static void main(String[] args) throws FileNotFoundException {
 
  if (args.length == 0) {
    System.out.println("Please provide an input filename.");
    return;
   }
  
  // Creating a new skip list and scanner             
  SkipList timeline = new SkipList();
  Scanner input = new Scanner(new File(args[0]));
    
    
  //Looping through the while statement as long as the scanner has next line
  // Splitting the input and storing it    
   while (input.hasNextLine()) {
    String line = input.nextLine();
    String[] lineSplit = line.split(" ", 3);
    String action = lineSplit[0];  
    
    // Checking if the action is Display event
    // Storing the date, using .get to find the event
    // Outputting the event if there is one, outputtinbg none if there is none
    if (action.equals("DisplayEvent")) {
     String date = lineSplit[1];
     String event = timeline.get(date);
     
     if (event == null) {
      System.out.println("DisplayEvent " + date + " none");
     }
     else {
      System.out.println("DisplayEvent " + date + " " + event);
     } 
    
    } 
    
    
    // Checking if the action is AddEvent
    // Storing the date and the event
    // Using out .put method to add the event to the skip list 
    // Outputting thast we were able to sucessfully add the event 
    if (action.equals("AddEvent")) {
     String date = lineSplit[1];
     String event = lineSplit[2];
     timeline.put(date, event);
     System.out.println("AddEvent " + date + " " + event + " success");
    }
    
    
    // Checking if the action is DeleteEvent
    // Storing the date, removing the event at the date and storing it 
    if (action.equals("DeleteEvent")) {
     String date = lineSplit[1];
     String event = timeline.remove(date);
     
      if (event == null) {
       System.out.println("DeleteEvent " + date + " noDateError");
      }
      else { 
       System.out.println("DeleteEvent " + date + " success");
      } 
    }


    // Checking if the action is Displaying events between dates
    // Storing the start and end date 
    // Outputting the events betweeen the dates
    if (action.equals("DisplayEventsBetweenDates")) {
     String startDate = lineSplit[1];
     String endDate = lineSplit[2];
     System.out.print("DisplayEventsBetweenDates " + startDate + " " + endDate + " ");
     printEntries(timeline.subMap(startDate, endDate));
    }
    
    
    // Checking if the action equals display events from start date
    // Storing the start date
    // Outputting the events using the print entries method
    if (action.equals("DisplayEventsFromStartDate")) {
     String startDate = lineSplit[1];
     System.out.print("DisplayEventsFromStartDate " + startDate + " ");
     printEntries(timeline.subMap(startDate, "9999"));
    }
    
    
    // Checking if action is display events to end date 
    // Storing the end date 
    // using our print entries method to output the events 
    if (action.equals("DisplayEventsToEndDate")) {
     String endDate = lineSplit[1];
     System.out.print("DisplayEventsToEndDate " + endDate + " ");
     printEntries(timeline.subMap("0000", endDate));
    }
    
    
    // Checking if the action is display all events
    // Using our print entries method to outpur all events
    if (action.equals("DisplayAllEvents")) {
     System.out.print("DisplayAllEvents ");
     printEntries(timeline.subMap("0000", "9999"));
    }
    
    
    // Checking if the action is print skip list and outputting it using out print skip list method
    if (action.equals("PrintSkipList")) {
     timeline.printSkipList();
    }
   } 
}    


// Creating a method to print entries 
public static void printEntries(ArrayList<Entry> entries) {
 
 
 // Outputing none if there are no entries 
 if (entries.isEmpty()) {
  System.out.println("none");
  return;
 }
 
 // Looping through the array list of entreis and outputting them
 for (int i = 0; i < entries.size(); i++) {
  if (i > 0) {
   System.out.print(" ");
  }
  System.out.print(entries.get(i).key + ":" + entries.get(i).value);
 }

 System.out.println();

}


// Creating an entry class
static class Entry {
 String key;
 String value;

 // Storing the key and value 
 public Entry(String key, String value) {
  this.key = key;
  this.value = value;
 }
}

// Creating a skiplist class to manage timeline events
static class SkipList {
 private static int maxHeight = 10;
 
 // Creating a Node class to store each event 
 private class Node {
  String key;
  String value;
  Node[] next;
  int height;
  
  // Storing the key, value and height of the node
  public Node(String key, String value, int height) {
  this.key = key;
  this.value = value;
  this.height = height;
  next = new Node[height + 1];
  }
 }

 private Node head;
 private int currentHeight;
 private FakeRandHeight randHeight;
 
 
 // Initialising the skip list with a head node and the given height randomizer
 public SkipList() {
 head = new Node(null, null, maxHeight);
 currentHeight = 0;
 randHeight = new FakeRandHeight();
 }
 // Creating a method to search for an event using date
 public String get(String key) {
  Node current = head;
  
  //Looping through the events 
  for (int i = currentHeight; i >= 0; i--) {
  
  //While the next date is before the given date, go to next event
   while (current.next[i] != null && current.next[i].key.compareTo(key) < 0)
    current = current.next[i];
   }

  // Moving to the next node
  current = current.next[0];
  
  // Returning event if date matches
  if (current != null && current.key.equals(key)) {
   return current.value;
  }

  // if nothing is found return null
  return null;
}

// Creating a method to add or replace an event 
public String put(String key, String value) {
 Node[] update = new Node[maxHeight + 1];
 Node current = head;
 
 // Looping through the levels 
 for (int i = currentHeight; i >= 0; i--) {
   
   //While the next date is before the given date, go to next event
   while (current.next[i] != null && current.next[i].key.compareTo(key) < 0) {
    current = current.next[i];
   } 
   
    update[i] = current;
 } 
  
  current = current.next[0];
  
  if (current != null && current.key.equals(key)) {
   String oldValue = current.value;
   current.value = value;
   return oldValue;
  }
  
  int height = randHeight.get();
  
  if(height > currentHeight) {
    for (int i = currentHeight + 1; i <= height; i++) {
     update[i] = head;
    }
     currentHeight = height;
  }
 
 Node newest = new Node(key, value, height);
 
 for(int i =0; i <= height; i++) {
  newest.next[i] = update[i].next[i];
  update[i].next[i] = newest;
 }
 
 return null;

}

public String remove(String key) {
 Node[] update = new Node[maxHeight + 1];
 Node current = head;
  
  for (int i = currentHeight; i >= 0; i--) { 
    while (current.next[i] != null && current.next[i].key.compareTo(key) < 0) {
      current = current.next[i];
    }
     update[i] = current;
  }

  current = current.next[0];
  
  if (current == null || !current.key.equals(key)) {
    return null;
  }

  for (int i = 0; i <= currentHeight; i++) {
    if (update[i].next[i] != current) {
     break;
    }
     update[i].next[i] = current.next[i];
   }

  while (currentHeight > 0 && head.next[currentHeight] == null) {
    currentHeight--;
  }

 return current.value;

}

public Entry floorEntry(String key) {
 Node current = head;
  
  for (int i = currentHeight; i >= 0; i--) {
    while (current.next[i] != null && current.next[i].key.compareTo(key) <= 0) {
     current = current.next[i];
    }
  }

  if (current == head) {
   return null;
  }
 
 return new Entry(current.key, current.value);

}


public Entry ceilingEntry(String key) {
 Node current = head;

  for (int i = currentHeight; i >= 0; i--) {
    while (current.next[i] != null && current.next[i].key.compareTo(key) < 0) {
      current = current.next[i];
    }
  }

  current = current.next[0];

  if (current == null) {
   return null;
  }

  return new Entry(current.key, current.value);

}

public ArrayList<Entry> subMap(String key1, String key2) {
  ArrayList<Entry> entries = new ArrayList<Entry>();
   
   if (key1.compareTo(key2) > 0) {
    return entries;
   }

   Node current = head;
   
   for (int i = currentHeight; i >= 0; i--) {
     while (current.next[i] != null && current.next[i].key.compareTo(key1) < 0) {
      current = current.next[i];
     }
   }

   current = current.next[0];
   
   while (current != null && current.key.compareTo(key2) <= 0) {
    entries.add(new Entry(current.key, current.value));
    current = current.next[0];
   }

  return entries;

}


public void printSkipList() {
 System.out.println("PrintSkipList");
 for (int i = currentHeight+1; i >= 0; i--) {
   System.out.print("(S" + i + ")");
   Node current = head.next[i];
    if (current == null) {
      System.out.println(" empty");
    }
    else {
     while (current != null) {
      System.out.print(" " + current.key + ":" + current.value);
      current = current.next[i];
     }
    System.out.println();
    }
 }
}
}
}

