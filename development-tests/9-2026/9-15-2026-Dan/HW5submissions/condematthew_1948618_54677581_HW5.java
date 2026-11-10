/*

  Author: Matthew Conde
  Email:  mconde2022@my.fit.edu
  Course: CSE 2010
  Section: 2
  Description of this file:  file creates and interacts with a skip list for managing events/dates
			     uses various methods that traverse the skip list to display events
			     at different levels, add events with a pseudo-random height
			     remove events, and display events between, before, or after dates
			     utilizes custom skip list class and node class
 */

import java.io.File;
import java.util.Scanner;

public class HW5
{
	// nested helper class, represents a single node in skip list
	// stores key (MMDD), value, next node at height i

    private static class SkipListNode {
        String key;
	String value;
	SkipListNode[] next;


	// constructor for node, takes same values mentioned above

	public SkipListNode(String key, String value, int height){
	    this.key = key;
	    this.value = value;
	    this.next = new SkipListNode[height];	// node has as many next nodes as the height
	}

    }

	// inner class to manage all nodes/operations 

    private static class SkipList {
	private static final int MAX_HEIGHT = 10;	// fixed max height
	private final SkipListNode head;		// define head node
	private final FakeRandHeight randHeight;	// pull random height values
	
	// create new instance of skiplist
	public SkipList() {
	    this.head = new SkipListNode(null, null, MAX_HEIGHT); // new skiplist node, so all levels have a head pointer
	    this.randHeight = new FakeRandHeight(); 
	}
	
	// helper method, locate predecessor nodes at all levels, move down through levels of skip list
	// at each level, move forward until  no node left or we find a node with larger key, 	
	// record the path taken to reach chosen key (or the floor closest to it)

	private SkipListNode[] findPredNodes(String key) {
	    SkipListNode[] found = new SkipListNode[MAX_HEIGHT];
	    SkipListNode curr = head;

	    for (int i = MAX_HEIGHT - 1; i>=0; i--){
		while (curr.next[i] != null && curr.next[i].key.compareTo(key) < 0){
		    curr = curr.next[i];
		}
		found[i] = curr;
	    }
	    return found;
	}
    

	// return value if the key exists, otherwise return null
	// uses previous method to find path to a given node key
	// make sure key is identical and not null, return its value
	// otherwise return null

	public String get(String key){
	    SkipListNode[] update = findPredNodes(key);
	    SkipListNode target = update[0].next[0];

	    if (target != null && target.key.equals(key)){
		return target.value;
	    }
	    return null;
	}

	// check if a value exists at a given key, if so return the old value and place the new value
	// otherwise generate a height for the key and cycle through levels updating node pointers	
	// then return null

	public String put(String key, String value) {
	    SkipListNode[] update = findPredNodes(key);
	    SkipListNode target = update[0].next[0];

	    // see if value already exists at exact key, if so return old value, place new value
	    if (target != null && target.key.equals(key)){
		String oldV = target.value;
		target.value = value;

		return oldV;
	     }
	    
	     int nodeHeight = randHeight.get() + 1;	// find randheight of a key, add a level
	     SkipListNode newNode = new SkipListNode(key, value, nodeHeight); // create new node

		// cycle through levels and update pointers for each node
	     for (int i = 0; i< nodeHeight; i++){		
		newNode.next[i] = update[i].next[i];
		update[i].next[i] = newNode;

	     }
		
	     return null;
	}

	// use helper to find all nodes leading to a target key, if target doesnt exist then return null
	// otherwise loop through path to target to remove all pointers to target, return target value

	public String remove(String key) {
	    SkipListNode[] update = findPredNodes(key);
	    SkipListNode target = update[0].next[0];

		// if the target doesn't exist (null or not exact key) then return null
	    if (target == null || !target.key.equals(key)) {
		return null;
	    }	

		// otherwise loop through path removing all pointers to target, return value
	    for (int i = 0; i<target.next.length; i++) {
		if (update[i].next[i] == target) {
		    update[i].next[i] = target.next[i];
		}
	    }
	
	    return target.value;
	}

	// find predecessors, if we find exact target match, return it
	// if no exact match find direct predecessor at level 0, otherwise return null

	public SkipListNode floorEntry(String key) {
	    SkipListNode[] update = findPredNodes(key);
	    SkipListNode target = update[0].next[0];

	    if (target != null && target.key.equals(key)){
		return target;
	    }
	   
	    if (update[0] != head) {
		return update[0];
	    }
		
	    return null;
	
	}

	// search for key in list, either find exact match, or smallest key larger than target
	// otherwise return null (built into predecessor nodes method), make sure lowest level

	public SkipListNode ceilingEntry(String key) {
	    SkipListNode[] update = findPredNodes(key);
	    SkipListNode target = update[0].next[0];

	    return target;
	}

	// search list between a given start and end key, count entries, initiate skiplist of count size
	// then loop back through and add each node to the list

	public SkipListNode[] subMap(String k1, String k2) {
	    int ct = 0;
	    SkipListNode curr = ceilingEntry(k1);

	    while (curr!= null && curr.key.compareTo(k2) <= 0) {
		ct++;
		curr = curr.next[0];
	    }

	    SkipListNode[] result = new SkipListNode[ct];
	    
	    curr = ceilingEntry(k1);

	    for (int i = 0; i< ct; i++){
		result[i] = curr;
	 	curr = curr.next[0];
	    }
	    return result;

	}



	// start at highest level and work down until find first node not empty, at each level
	// work forward through nodes, repeat for each level until last level

	public void printSkipList() {
	    int maxLevel = MAX_HEIGHT - 1;

	    while (maxLevel > 0 && head.next[maxLevel] == null) {
		maxLevel--;	// find highest level with at least one node
	    }
	    
   	    System.out.println("PrintSkipList");
 
		// if we are at max height, dont print empty level, otherwise empty level is maxlevel +1  
	    if (head.next[maxLevel] != null && maxLevel + 1 < MAX_HEIGHT) {
        	System.out.println("(S" + (maxLevel + 1) + ") empty");
    	    }
	
		// move down levels, cycle forward at each level to print entire list
	    for (int level = maxLevel; level >= 0; level --){
		System.out.print("(S" + level + ")");
		SkipListNode curr = head.next[level];
		
		if (curr == null) {
		    System.out.println(" empty");
		} else {				// move forward in a level
		    while (curr!= null) {
			System.out.print(" " + curr.key + ":" + curr.value);
			curr = curr.next[level];
		    }
		    System.out.println();
		}
	    }

	}
    }

    public static void main(String[] args) throws Exception {
	
	// pull input file from command line, create scanner and skiplist
	File inputFile = new File(args[0]);
	Scanner scnr = new Scanner(inputFile);
	SkipList list = new SkipList();
	
	// initialize each relevant variable so we dont have to declare in each case
	String date, startDate, endDate, event, removed, val;
	SkipListNode[] entries; 

	while (scnr.hasNextLine()) {
	
		// pull next line from scanner
	    String line = scnr.nextLine();
	    if (line.isEmpty()) continue;

	    Scanner sc = new Scanner(line);
	
		// switch over keyword of each line
	    switch (sc.next()) {

		// pull date and event, print output and add to skiplist with .put
		case "AddEvent": {
		    date = sc.next();
		    event = sc.next();
		
		    while(sc.hasNext()){
			event = event + " " + sc.next();
		    }

		    list.put(date, event);
		    System.out.println("AddEvent " + date + " " + event + " success");
		    break;
		}
		
		// pull date, remove from skip list, if not found then output error
		case "DeleteEvent": {
		    date = sc.next();
		    removed = list.remove(date);
		    
		    if (removed != null) {
			System.out.println("DeleteEvent " + date + " success");
		    } else {
			System.out.println("DeleteEvent " + date + " noDateError");
		    }
		    break;
		}
	
		case "PrintSkipList": {
		    list.printSkipList();	// call print skip list
		    break;
		}

		case "DisplayEvent": {		// gets value at key(date), output value or none if no value
		    date = sc.next();
		    val = list.get(date);

		    if (val != null) {
			System.out.println("DisplayEvent " + date + " " + val);
		    } else {
			System.out.println("DisplayEvent " + date + " none");
		    }
		    break;
		}

		// start at ceiling of startDate, loop through all nodes after until next date
		// is larger than target endDate, at lowest level of list (all in .subMap)
	
		case "DisplayEventsBetweenDates": {
		    startDate = sc.next();
		    endDate = sc.next();

		    System.out.print("DisplayEventsBetweenDates " + startDate + " " + endDate);

		    entries = list.subMap(startDate, endDate);
		    if (entries.length == 0) {
		  	System.out.println(" none");	
		    } else {
			for (SkipListNode node : entries) {
				System.out.print(" " + node.key + ":" + node.value);
			}
			System.out.println();
		    }
		    break;
		}

		// ceiling of startDate, loop through all nodes after until we reach end of year (1231)
		// at lowest level of list (all done in submap)
		case "DisplayEventsFromStartDate": {
		    startDate = sc.next();
		    endDate = "1231";

		    System.out.print("DisplayEventsFromStartDate " + startDate);
		
                    entries = list.subMap(startDate, endDate);
                    if (entries.length == 0) { 
                        System.out.println(" none");
                    } else {
                        for (SkipListNode node : entries) { 
                                System.out.print(" " + node.key + ":" + node.value);
                        }
                        System.out.println();
                    }
		    break;
		}

		// ceiling of startDate (beginning of year, 0101), loop through all nodes after until end date
		// at lowest level of list (all done in submap)
		case "DisplayEventsToEndDate": {
		    endDate = sc.next();
		    startDate = "0101";
		    System.out.print("DisplayEventsToEndDate " + endDate);

                    entries = list.subMap(startDate, endDate);
                    if (entries.length == 0) { 
                        System.out.println(" none");
                    } else {
                        for (SkipListNode node : entries) { 
                                System.out.print(" " + node.key + ":" + node.value);
                        }
                        System.out.println();
                    }
		    break;
		}
		
		// ceiling of start date (0101), loop through all nodes after until end date (1231)
		// at lowest level of list (all done in submap)
		case "DisplayAllEvents": {
		    System.out.print("DisplayAllEvents");
		
		    startDate = "0101";
		    endDate = "1231";

                    entries = list.subMap(startDate, endDate);
                    if (entries.length == 0) {
                        System.out.println(" none");
                    } else {
                        for (SkipListNode node : entries) {
                                System.out.print(" " + node.key + ":" + node.value);
                        }
                        System.out.println();
                    }
		    break;
		}
	    }
	}
    }
}
























