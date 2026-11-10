//Thank you for the help daniel! Sorry i dint get to come back in to show my code a bunch of stuff happened that took priority
//but I took the info you gave me and it helped a ton

/* Richie Kowalski
   Rkowalski2023@my.fit.edu
   CSE 2010
   HW5:
   take a list of events and dates and store them in a skip list. Then be able to add, remove, and display events based on the date.
   */

//resources used: https://www.geeksforgeeks.org/dsa/skip-list-set-2-insertion/
    //used for the skip list implementation to store the events and dates in a priority queue
//https://www.baeldung.com/java-skiplist
    //same use as above, but also used to help with the submap method
//Provided Java Files on canvas Specifically HW5.java and FakeRandHeight.java



import java.io.*;
import java.util.*;

public class HW5 {

//test FAKERANDHEIGHT
//change to import as FakeRandHeightMethod and compile together as in doc
    
/* 
public class FakeRandHeight
{
static int[] height = {0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, //
sequence of height
0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0};
int count; // number of times get() has been called
public FakeRandHeight()
{
count = 0;
}
// return the next number in the sequence, recycle the sequence if needed
public int get()
{
return height[count++ % 31];
}
}
*/





    static class Node {
        String date;
        String event;
        Node[] next;

        Node(String date, String event, int height) {
            this.date = date;
            this.event = event;
            next = new Node[height + 1];
        }
    }
    //date event and height. height set to 9 so that the skip list can have a maximum of 10 levels. The next array is used to store the next nodes at each level.
    static class SkipList {
        FakeRandHeight randHeight = new FakeRandHeight();
        Node head = new Node("", "", 9);
        int topLevel = 0;



        // Find the node immediately before the given date.
        //get the head loop for each level. while the next node isnt null and the next node's date is less than the given date, move to the next node. return the current node.
        Node findBefore(String date) {
            Node current = head;

            for (int i = topLevel; i >= 0; i--) {
                while (current.next[i] != null && current.next[i].date.compareTo(date) < 0) {
                    current = current.next[i];
                }
            }
            return current;
        }

        // get evewnt
        // get the node with given date through the get before. then check if the node is not null and if the date matches. If it does, return the event. Otherwise, return null.
        String get(String date) {
            Node node = findBefore(date).next[0];

            if (node != null && node.date.equals(date))
                return node.event;

            return null;
        }

        // thank you again daniel!
        // put method to instert new event
        // create array of nodes to track  each level 
        
        String put(String date, String event) {
            Node[] update = new Node[10];
            Node current = head;

            //for loop to traverse the skip list from the top level down to level 0. At each level
            // it moves forward through the nodes until it finds the correct position for the new node based on the date.
            // The update array keeps track of the nodes that need to be updated when inserting the new node.
            for (int i = topLevel; i >= 0; i--) {
                while (current.next[i] != null &&
                       current.next[i].date.compareTo(date) < 0) {
                    current = current.next[i];
                }
                update[i] = current;
            }
            // check if the node with the given date already exists. If it does, update its event and return the old event.
            // Otherwise, create a new node with a random height and insert it into the skip list by updating the next pointers of the nodes in the update array.
            Node found = current.next[0];

            if (found != null && found.date.equals(date)) {
                String old = found.event;
                found.event = event;
                return old;
            }

            // Generate a random height for the new node.
            // EDIT TO USE HIS FILE
            int height = randHeight.get();

            // If the new node's height is greater than the current top level of the skip list, update the update array for the new levels and set the new top level.
            if (height > topLevel) {
                for (int i = topLevel + 1; i <= height; i++)
                    update[i] = head;

                topLevel = height;
            }

            //finallly create a new node with the given date, event, and height. Update the next pointers of the nodes in the update array to insert the new node into the skip list.
            Node node = new Node(date, event, height);

            for (int i = 0; i <= height; i++) {
                node.next[i] = update[i].next[i];
                update[i].next[i] = node;
            }

            return null;
        }

        // remove
        String remove(String date) {
            Node[] update = new Node[10];
            Node current = head;

            // go through the skip list to find the node to be removed and update the update array with the nodes that need to be updated.
            for (int i = topLevel; i >= 0; i--) {
                while (current.next[i] != null && current.next[i].date.compareTo(date) < 0) {
                    current = current.next[i];
                }
                update[i] = current;
            }

            // then check if the node to be removed exists.
            // If it does, update the next pointers of the nodes in the update array to remove the node from the skip list.
            // Finally, adjust the top level of the skip list if necessary and return the event of the removed node.
            Node found = current.next[0];

            if (found == null || !found.date.equals(date))
                return null;

            for (int i = 0; i <= topLevel; i++) {
                if (update[i].next[i] == found)
                    update[i].next[i] = found.next[i];
            }

            while (topLevel > 0 && head.next[topLevel] == null)
                topLevel--;

            return found.event;
        }


        // simple method to find the node with the largest date
        // less than or equal to the given date. It first finds the node immediately before the given date using the findBefore method. 
        // Then, it checks if the next node exists and if its date matches the given date. If it does, it returns that node.
        //  If not, it checks if the before node is the head of the skip list. If it is, it means there is no node with a date less than or equal to the given date,
        //  so it returns null. Otherwise, it returns the before node, which is the largest node with a date less than or equal to the given date.
        Node floorEntry(String date) {
            Node before = findBefore(date);
            Node next = before.next[0];

            if (next != null && next.date.equals(date))
                return next;

            if (before == head)
                return null;

            return before;
        }


        // celing is just return before next at 0 level
        //so cuurent node at lowest level
        Node ceilingEntry(String date) {
            return findBefore(date).next[0];
        }


        // submap method to return a string representation of the events between two given dates
        // check if the start date is greater than the end date.
        //  If it is, return "none". Otherwise, find the node with the smallest date greater than or equal to the start date using the ceilingEntry method.
        //  Then, iterate through the nodes at lvl 0 until reaching a node with a date greater than the end date.
        // For each node in this range, append its date and event to the result string. If no events are found, return "none". Otherwise, return the result string.
        String subMap(String start, String end) {
            if (start.compareTo(end) > 0) {
                return "none";
            }

            Node current = ceilingEntry(start);
            String result = "";

            while (current != null && current.date.compareTo(end) <= 0) {
                if (result.length() > 0) {
                    result += " ";
                }
                result += current.date + ":" + current.event;
                current = current.next[0];
            }

            if (result.length() == 0) {
                return "none";
            }

            return result;
            }


        // all events method to return a string representation of all events in the skip list
        // It starts from the head node and iterates through the nodes at level 0, appending each node's date and event to the result string.
        // If no events are found, it returns "none". Otherwise, it returns the result string.
        String allEvents() {
            Node current = head.next[0];
            String result = "";

            while (current != null) {
                if (result.length() > 0) {
                    result += " ";
                }
                result += current.date + ":" + current.event;

                current = current.next[0];
            }

            if (result.length() == 0) {
                return "none";
            }
            else {
                return result;
            }
        }

        // print skip list method to print the contents of the skip list at each level
        // It starts from the highest level and iterates down to level 0. For each level
        // it prints the level number and then iterates through the nodes at that level, printing each node's date and event.
        void printSkipList() {
            System.out.println("PrintSkipList");

            int highest = Math.min(9, topLevel + 1);

            for (int i = highest; i >= 0; i--) {
                System.out.print("(S" + i + ")");

                Node current = head.next[i];

                if (current == null) {
                    System.out.println(" empty");
                    continue;
                }

                while (current != null) {
                    System.out.print(" " + current.date + ":" + current.event);
                    current = current.next[i];
                }

                System.out.println();
            }
        }
    }


    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            return;
        }
        Scanner input = new Scanner(new File(args[0]));
        SkipList list = new SkipList();



        while (input.hasNextLine()) {
            String line = input.nextLine().trim();
            if (line.length() == 0) {
                continue;
            }
            String[] parts = line.split("\\s+", 3);
            String action = parts[0];




            //list of diff actions for skip list
            if (action.equals("AddEvent")) {
                if (parts.length >= 3) {
                    list.put(parts[1], parts[2]);
                    System.out.println("AddEvent " + parts[1] + " " + parts[2] + " success");
                }
            }
            else if (action.equals("DisplayEvent")) {
                String event = list.get(parts[1]);

                if (event == null) {
                event = "none";
                }

                System.out.println("DisplayEvent " + parts[1] + " " + event);
            }

            else if (action.equals("DeleteEvent")) {
                String removed = list.remove(parts[1]);

                if (removed == null) {
                    System.out.println("DeleteEvent " + parts[1] + " noDateError");
                }
                else {
                    System.out.println("DeleteEvent " + parts[1] + " success");
                }
            }

            else if (action.equals("DisplayEventsBetweenDates")) {
                String[] dates = line.split("\\s+");

                System.out.println("DisplayEventsBetweenDates " + dates[1] + " " + dates[2] + " " + list.subMap(dates[1], dates[2]));
            }

            else if (action.equals("DisplayEventsFromStartDate")) {
                System.out.println("DisplayEventsFromStartDate " + parts[1] + " " + list.subMap(parts[1], "1231"));
            }

            else if (action.equals("DisplayEventsToEndDate")) {
                System.out.println("DisplayEventsToEndDate " + parts[1] + " " + list.subMap("0101", parts[1]));
            }

            else if (action.equals("DisplayAllEvents")) {
                System.out.println("DisplayAllEvents " + list.allEvents());
            }

            else if (action.equals("PrintSkipList")) {
                list.printSkipList();
            }
        }

        input.close();
    }
}
