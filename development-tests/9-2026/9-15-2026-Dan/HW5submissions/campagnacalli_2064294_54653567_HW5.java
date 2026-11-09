import java.util.Scanner;
import java.io.File; 
import java.io.FileNotFoundException;
/*
 * Author: Calli Campagna 
 * Email:ccampagna2025@code01.fit.edu
 * Course:CSE2010
 * Section:1-4
 * Description of this file: 
 * this program uses a skip list to store and manage events by date. It allows the user to add
 * output, delete, and search for events.It can output events within date ranges or output 
 * all events.It can also print the different levels of the skip list.
*/
public class HW5{
  public static void main(String[] args) throws FileNotFoundException {
     /*
      * Description: The main method reads commands from an input file and performs certain outputs 
      * on the skip list based on command statement. Creates the HW5 object and the skip list
      * used to store events.Scanner is created and file is opened 
      * Parameters: args - contains the input filename at index 0.
      */
     HW5 homework = new HW5();
     SkipList list = homework.new SkipList();
     Scanner sc = new Scanner(new File(args[0]));
     /*
      * store the current line and create a scanner to read each part of the line. Then the first command 
      * is heldthat determines which operation to perform. 
      */  
     while(sc.hasNextLine()){
         String line = sc.nextLine();
         Scanner scanner = new Scanner(line);
         
         String command = scanner.next();
         /*
          * if the command line is AddEvent it reads a date and event description, then inserts
          * them into the skip list using the put method.Then outputs the results 
          */
         if(command.equals("AddEvent")){
             String data = scanner.next();
             String event = scanner.next();
             list.put(data,event);
             System.out.println("AddEvent " + data + " " + event + " success");
             /*
              * if the command line is DispalyEvent it searches for the specified date. If the
              * date exists its event description is printed. Otherwise the program prints none.
              */
         }else if (command.equals("DisplayEvent")){
             String data = scanner.next();
             String event = list.get(data);
             
             if (event == null){
                System.out.println("DisplayEvent " + data + " none");
             }else{
                 System.out.println("DisplayEvent " + data + " " + event);
             }
             /*
              * if the command line is DeleteEvent it removes the event associated with the specified date.
              * The remove method returns the old event description if successful. If the date does 
              * not exist the program prints an error message.
              */
         }else if(command.equals("DeleteEvent")){
             String data = scanner.next();
             String event = list.remove(data);
             
             if (event == null){
                System.out.println("DeleteEvent " + data + " noDateError");
             }else{
                 System.out.println("DeleteEvent " + data + " success");
             }
             /*
              * if the command line is DisplayEventsBetweenDates it reads a starting and ending date. 
              * it sends those dates to the specific method that gets the dates within this range.
              * It displays all events whose dates fall within that range including events on the starting 
              * and ending dates.
              */
        }else if(command.equals("DisplayEventsBetweenDates")){
            String startDate = scanner.next();
            String endDate = scanner.next();
            String events = list.displayEventsBetweenDates(startDate, endDate);
            
            if (events.isEmpty()){
                System.out.println("DisplayEventsBetweenDates " + startDate + " " + endDate + " none");
            }else{
                System.out.println("DisplayEventsBetweenDates " + startDate + " " + endDate + " " + events);
            }
            /*
             * if the command line is DisplayEventsFromStartDate it reads a starting. 
              * it sends that dates to the specific method that gets the dates within this range.
              * It displays all events whose dates fall within that range including the event on the starting 
              * date.
             */
        }else if (command.equals("DisplayEventsFromStartDate")){
            String startDate = scanner.next();
            String events = list.displayEventsFromStartDate(startDate);
            
                        if (events.isEmpty()){
                System.out.println("DisplayEventsFromStartDate " + startDate + " none");
            }else{
                System.out.println("DisplayEventsFromStartDate " + startDate + " "  + events);
            }
            /*
             * if the command line is DisplayEventsToEndDate it reads an ending date. 
              * it sends those dates to the specific method that gets the dates within this range.
              * It displays all events whose dates fall within that range including the ending date.
             */
        }else if (command.equals("DisplayEventsToEndDate")){
            String endDate = scanner.next();
            String events = list.displayEventsToEndDate(endDate);
            
            if (events.isEmpty()){
                System.out.println("DisplayEventsToEndDate " + endDate + " none");
            }else{
                System.out.println("DisplayEventsToEndDate " + endDate + " "  + events);
            }
            /*
             * if the command line is DisplayAllEvents it displays all events and the events lsit is empty 
             * it outputs none 
             */
        }else if (command.equals("DisplayAllEvents")){
            String events = list.displayAllEvents();
            
            if (events.isEmpty()){
                System.out.println("DisplayAllEvents "  + " none");
            }else{
                System.out.println("DisplayAllEvents " + events);
            }
            /*
             * if the command line is PrintSkipList it prints the skip list showing the nodes 
             * connected at each level from the highest level to level 0.
             */
        }else if (command.equals("PrintSkipList")){
             System.out.println("PrintSkipList");
             System.out.print(list.printSkipList());
         }
     }
     
     
  }
  /*
   * Node represents one event in the skip list. Each node stores a date
   * an event description and an array of references to other nodes.
   * Variables: key - the date associated with the event. 
   * value - the event description.
   * next - an array containing the next node at each level.
   */
    class Node {

    String key;
    String value;
    Node[] next;
    /*

     * Creates a node containing a date, event description, and links for all levels up to the specified height.
     * Parameters: key - the event date.
     * value - the event description.
     * height - the highest level occupied by this node.
     */
    
    public Node(String key, String value, int height) {
        this.key = key;
        this.value = value;
        this.next = new Node[height + 1];
    }
  }

  /*
   *SkipList stores events in sorted order using multiple linked levels. Higher levels allow the program 
   *to skip over nodes during searches.
   *Variables: maxHeight - the maximum level of the skip list.
   *head - the starting node for all levels. 
   *size - the number of stored events.
   *andHeight - generates the height for each new node.
   */

   class SkipList {

    int maxHeight = 10;

    Node head;
    int size;
    FakeRandHeight randHeight;

    /*
     * Initializes an empty skip list, creates the head node, sets the size to zero, and initializes 
     * the height generator.
     */
    
    public SkipList() {
        
        head = new Node(null, null, maxHeight);
        size = 0;
        randHeight = new FakeRandHeight();
    }


    /*
     * Searches for a date in the skip list and returns its event description if the date exists. The for 
     * loop searches from the highest level down to level 0. The while loop moves forward while the next data 
     * is smaller than the target 
     * Parameters: key - the date to search for.
     * Returns: The event description if found or null if not
     */
    public String get(String key) {
        Node current = head;
        
        for(int i = maxHeight; i >= 0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(key) < 0){
             current = current.next[i];
        }
       }
       /* 
        * checks the next node at level 0 for the requested data 
        */
       current = current.next[0];
       if(current != null && current.key.equals(key)){
           return current.value;
       }
       return null; 
    }

    /*
     * Inserts a new date and event into the skip list. If the date already exists, its event description is updated.
     * Parameters:key - the date to insert or update.
     * value - the event description.
     * Returns:mThe previous event description if updated or null if not.
     */
    
    public String put(String key, String value) {
        /*
         * store the node immediately before the insertion position at each level. The for loop searches from the 
         * highest level down to level 0. The while loop moves forward while the next data is smaller than the target 
         */
        Node[] update = new Node[maxHeight + 1];
        Node current = head;
        
        for (int i = maxHeight; i >=0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(key) < 0){
                current = current.next[i];
            }
            /*
             * Save current as the  previous node at this level so we can reconnect the links later
             */
            update[i] = current;
        }
        /*
         * Check whether the date already exists at level 0
         */
        current = current.next[0];
        
        if (current!= null && current.key.equals(key)){
            String oldValue = current.value;
            current.value = value;
            return oldValue;
        }
        /*
         * Generate a height and create the new event node. Connect the new node to the list at each level it occupies.
         * increase the number of stored events 
         */
        int nodeHeight = randHeight.get();
        Node newNode = new Node(key,value,nodeHeight);
        
        for(int i =0; i <= nodeHeight; i++){
            newNode.next[i] = update[i].next[i];
            update[i].next[i] = newNode;
        }
        size++;
        return null; 
    }


    /*
     *Removes the node associated with the specified date and returns its event description.
     *Parameters: key - the date to remove.
     *Returns:The removed event description if found or null if not
     */
    public String remove(String key) {
       /*
        * store the node immediately before the target at each level
        */
       Node[] update = new Node[maxHeight + 1];
       Node current = head;  
       /*
        * store the node immediately before the insertion position at each level. The for loop searches from the 
        * highest level down to level 0. The while loop moves forward while the next data is smaller than the target 
        */
       for(int i = maxHeight; i >= 0; i--){
           while(current.next[i] != null && current.next[i].key.compareTo(key) <0){
               current = current.next[i];
           }
           /*
            * Save current as the  previous node at this level so we can reconnect the links later
            */
           update[i] = current; 
       }
       /*
        * Check the next node at level 0
        */
       current = current.next[0];
       if (current == null || !current.key.equals(key)){
           return null;
        }
        /*
         * Save the description before removing the node. For loop bypass the node at every level in which it appears
         */
        String oldValue = current.value;
        
       for (int i = 0; i < current.next.length; i++){
           update[i].next[i] = current.next[i];
       }
       size--;
       return oldValue;
       
     }

    /*
     * Finds the node with the greatest key less than or equal to the specified key.
     * Parameters: key - the date used as the search target.
     * Returns: The matching floor node or null if no suitable node exists.
     */
    
    public Node floorEntry(String key) {
        
        Node current = head;
        /*
         * Search for an exact match or the position before the target.The for loop searches from the 
         * highest level down to level 0. The while loop moves forward while the next data is smaller than the target 
         */
        for(int i = maxHeight; i >= 0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(key) < 0){
                current = current.next[i];
            }
        }
        /*
         *  Check the next node at level 0
         */
        current = current.next[0];
        /*
         * Return immediately if the exact date exists
         */
        if (current != null && current.key.equals(key)){
            return current;
        }
        /*
         * Restart to find the greatest date less than the target. The for loop searches from the 
         * highest level down to level 0. The while loop moves forward while the next data is smaller than the target 
         */
        current = head;
        for(int i = maxHeight; i >= 0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(key) <= 0){
                current = current.next[i];
            }
        }
        if(current == head){
            return null;
        }
        return current;
    }

    /*
     * Finds the node with the smallest key greater than or equal to the specified key.
     * Parameters: key - the date used as the search target. 
     * Returns: The matching ceiling node or null if no suitable node exists.
     */
    
    public Node ceilingEntry(String key) {
        
        Node current = head;
        /*
         * The for loop searches from the highest level down to level 0. The while loop moves forward while the next data is smaller 
         * than the target 
         */
        for(int i = maxHeight; i >= 0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(key) < 0){
                current = current.next[i];
            }
        }
        /*
         * The next node is the first date greater than or equal to key
         */
        return current.next[0];
    }


    /*
     * Returns all events whose keys fall between key1 and key2, including both endpoints.
     * Parameters:key1 - the starting date. 
     * key2 - the ending date.
     * Returns: A string containing the matching dates and event descriptions.
     */
    public String subMap(String key1, String key2) {
        
        String result = "";
        Node current = head;
        /*
         * Find the position immediately before key1. The for loop searches from the highest level down to level 0. The while loop moves forward while the next data is smaller 
         * than the target 
         */
        for(int i = maxHeight; i >= 0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(key1) < 0){
                current = current.next[i];
            }
        }  
        current = current.next[0];
        /*
         * Continue going through the skip list at level 0 while there is another node to visit and its date is less than or equal to key2.
         * The loop stops when there are no more nodes or the date is after key2.
        */
        while(current != null && current.key.compareTo(key2) <= 0){
            /*
             * Check whether an event has already been added to the result string. If the result is not empty, add a space before the next event
             * so that the events are separated in the output. Add the current event to the result string. Store the date first, followed
             * by a colon and the event description
             */
            if(!result.equals("")){
                result += " ";
            }
            result += current.key + ":" + current.value;
            current = current.next[0];
        }
        return result;
    }

    /*
     * Returns every event stored in the skip list, regardless of its date. The events are returned in ascending date order
     * with each event formatted as date:eventDescription.
     * Returns:A string containing every event, with a space separating each event.
     */
    
    public String displayAllEvents() {
        /*
         * Start with an empty string and begin at the first actual node in the skip list
         */
        String result = "";
        Node current = head.next[0];
        /*
         * while loop continues visiting nodes until current becomes null. A null reference means there are no more events to process.
         * if statement checks whether an event has already been added to result
         */
        while(current != null){
            if (!result.equals("")){
                result += " ";
            }
            /*
             * Add the current event to the output string. current.key stores the event's date. current.value stores the event's description.
             */
            result += current.key + ":" + current.value;
            current = current.next[0];
        }
        return result;
    }


    /*
     * Returns every event whose date is greater than or equal to the specified starting date. Events are returned in ascending
     * date order, including an event that occurs exactly on startDate.
     * Parameters: startDate - the earliest date to include in the results.
     * Returns: A string containing all events on or after startDate with a space separating each event.
     */
    public String displayEventsFromStartDate(String startDate) {
        String result = "";
        Node current = head;
        
        /*
         * The for loop searches from the highest level down to level 0. The while loop moves forward while the next data is smaller 
         * than the target 
         */
        for(int i = maxHeight; i >= 0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(startDate) < 0){
                current = current.next[i];
            }
        } 
        /*
         * Collect every remaining event. No ending date is specified, so continue until there are no more nodes in the skip list.
         */
        current = current.next[0];
        while(current != null){
            if(!result.equals("")){
                result += " ";
            }
            result += current.key + ":" + current.value;
            current = current.next[0];
        }
        /*
         * Return the string containing all events on or after startDate. If no events meet the condition, the returned string is empty
         */
        return result;
    }
    /*
     * Returns all events whose dates fall between startDate and endDate, including events that occur exactly on either boundary date.
     * The events are returned in ascending date order.
     * Parameters: startDate - the earliest date to include.
     * endDate - the latest date to include.
     * Returns:A string containing all events in the specified date range, with a space separating each event.
     */
    public String displayEventsBetweenDates(String startDate, String endDate) {
        String result = "";
        Node current = head;
        
        /*
         * The for loop searches from the highest level down to level 0. The while loop moves forward while the next data is smaller 
         * than the target 
         */
        for(int i = maxHeight; i >= 0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(startDate) < 0){
                current = current.next[i];
            }
        }  
        
        current = current.next[0];
        /* collect events while there is a node to process and its date is less than or equal to endDate. The loop stops when it reaches
         * a date after endDate.
         */
        while(current != null && current.key.compareTo(endDate) <= 0){
            /*
             * If at least one event has already been collected add a space before appending the next event.
             */
            if(!result.equals("")){
                result += " ";
            }
            result += current.key + ":" + current.value;
            current = current.next[0];
        }
        /*
         * Return all events between the two dates, including both boundary dates. If no events match, return an empty string.
         */
        return result;
    }

    /*
     * Returns all events whose dates are less than or equal to the specified ending date. Events are returned in ascending
     * date order, beginning with the earliest event in the skip list.
     * Parameters: endDate - the latest date to include in the results.
     * Returns:A string containing every event on or before endDate with a space separating each event.
     */    
    public String displayEventsToEndDate(String endDate) {
        String result = "";
        Node current = head.next[0];
        /*
         * Continue while a node exists and its date is before or equal to endDate. Because events are ordered by date, once a date
         * exceeds endDate, no later event can meet the condition.
         */
        while(current != null && current.key.compareTo(endDate) <= 0){
            if(!result.equals("")){
                result += " ";
            }
            result += current.key + ":" + current.value;
            current = current.next[0];
        }
        /*
         * Return the collected events. If the first event is after endDate, this returns an empty string.
         */
        return result;
    }


    /*
     * Creates a formatted string showing the events at each level of the skip list, beginning with the level above the highest occupied
     * level when possible and ending at level 0. Each level displays only the nodes connected at that particular level.
     * Returns: A formatted string containing the skip list's levels and events. Each level is displayed on a separate line.
     */
    public String printSkipList() {
        
        String result = "";
        int highestLevel = 0;
        /*
         *  Search from the maximum possible level down to level 0. The goal is to find the highest level that contains at least
         *  one actual event node connected to the head.
         */
        for(int i = maxHeight; i >= 0; i--){
            if(head.next[i] != null){
                highestLevel = i;
                break;
            }
        }
        /*
         * rint the levels from highest to lowest. The loop begins at startLevel and decreases i by one each time, ending after 
         * level 0 has been processed.
         */
        int startLevel = Math.min(highestLevel + 1, maxHeight);
        for(int i = startLevel; i >= 0; i--){
            result += "(S" + i + ")";
            
            Node current = head.next[i];
            
            if (current == null){
                result += " empty";
            }else{
                /*
                 * Visit every node connected at the current level. A higher level may contain fewer nodes than level 0
                 * because not every event has a height reaching this level.
                 */
                while(current != null){
                    result += " " + current.key + ":" + current.value;
                    current = current.next[i];
                }
            }
            result += "\n";
        }
        return result;
    }

    /*
     * Finds the predecessor node for the specified key at every level of the skip list. A predecessor is the last node before the target
     * key at that level. The resulting array is useful when inserting or removing a node because it identifies which links must change.
     * Parameters: key - the date whose position is being searched for.
     * Returns: An array of previous nodes, where update[i] stores the previous node at level i.
     */
    
    private Node[] findPredecessors(String key) {
        
        Node[] update = new Node[maxHeight + 1];
        Node current = head;
        /*
         * The for loop searches from the highest level down to level 0. The while loop moves forward while the next data is smaller 
         * than the target 
         */        
        for(int i = maxHeight; i >=0; i--){
            while(current.next[i] != null && current.next[i].key.compareTo(key) < 0){
                current = current.next[i];
            }
            update[i] = current;
        }
        /*
         * Return the completed predecessor array
         */
        return update;
    }
 
     
   }
  } 