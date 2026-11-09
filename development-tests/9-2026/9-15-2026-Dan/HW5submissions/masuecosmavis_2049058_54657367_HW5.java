/*
  Author Clarence Masuecos
  Email mmasuecos2025@my.fit.edu
  Course CSE 2010
  Section: 3
  Description of this file: 
  implements Skip List data structure from scratch to manage timeline of events.
   uses 4-way linked node system to achieve O(log n) insertions, deletions, 
  and lookups. range queries supported to display events between date
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;

public class HW5 {

    /* description of variables */
    // skipnode represents single entry in  skip list grid.
    // 'key' stores date (eg "0405")and 'value' stores  event.
    // up, down, left, and right  pointers to navigate  2D grid.
    static class skipnode {
        String key;
        String value;
        skipnode up, down, left, right;

        public skipnode(String key, String value) {
            this.key = key;
            this.value = value;
        }
    }

    /* description of variables */
    // skipList manages entry points of the grid
    // 'head' is  top-left negative infinity sentinel.
    // 'tail' is  top-right positive infinity sentinel.
    // 'height' tracks the current maximum level.
    static class skipList {
        skipnode head;
        skipnode tail;
        int height;
        FakeRandHeight randomizer;

        /* description of each block (around 5-10 lines) of instructions */
        //  constructor initializes base level (S0) of the list.
        // creates the -INF and +INF sentinel nodes to act as left and 
        // right guard rails. the height starts at 0 and randomizer is loaded.
        public skipList() {
            head = new skipnode("-INF", null);
            tail = new skipnode("+INF", null);
            head.right = tail;
            tail.left = head;
            height = 0;
            randomizer = new FakeRandHeight();
        }

        /* description of each block (around 5-10 lines) of instructions */
        // search() drops from top-left head down  the bottom level (S0).
        // at each level, scans right as long as the next nodes key is less 
        // than or equal to the target. returns the closest node at S0.
        private skipnode search(String key) {
            skipnode curr = head;
            while (curr.down != null) {
                curr = curr.down;
                while (!curr.right.key.equals("+INF") && curr.right.key.compareTo(key) <= 0) {
                    curr = curr.right;
                }
            }
            while (!curr.right.key.equals("+INF") && curr.right.key.compareTo(key) <= 0) {
                curr = curr.right;
            }
            return curr; 
        }

        /* description of each block (around 5-10 lines) of instructions */
        // get() uses the search method to locate node at S0
        // if  returned node exactly matches requested key, it returns 
        // the event string. otherwise the event does not exist returning null
        public String get(String key) {
            skipnode p = search(key);
            if (p.key.equals(key)) {
                return p.value;
            }
            return null;
        }

        /* description of each block (around 5-10 lines) of instructions */
        // put() first checks if date is already in the list. If it is,  
        // overwrite  event name on all vertical levels of that node. 
        // if its a new date, creates  new bottom-level node then links it
        public void put(String key, String value) {
            skipnode p = search(key);
            
            if (p.key.equals(key)) {
                while (p != null) {
                    p.value = value;
                    p = p.up;
                }
                return;
            }

            skipnode q = new skipnode(key, value);
            q.left = p;
            q.right = p.right;
            p.right.left = q;
            p.right = q;

            /* description of each block (around 5-10 lines) of instructions */
            //  get a random height to determine how tall node's tower is
            // if  new height requires more levels than  current have, 
            //  dynamically add new empty top levels to skip list until 
            // we have exactly one completely empty level on top.
            int nodeLevel = randomizer.get();
            while (height <= nodeLevel) {
                skipnode newHead = new skipnode("-INF", null);
                skipnode newTail = new skipnode("+INF", null);
                newHead.right = newTail;
                newTail.left = newHead;
                
                newHead.down = head;
                head.up = newHead;
                newTail.down = tail;
                tail.up = newTail;
                
                head = newHead;
                tail = newTail;
                height++;
            }

            /* description of each block (around 5-10 lines) of instructions */
            // build the tower for the new node from  bottom up. 
            //  scan left until we find  node that goes up, move up a level, 
            // and link new tower node horizontally + vertically.
            int currentLevel = 0;
            while (currentLevel < nodeLevel) {
                while (p.up == null) {
                    p = p.left;
                }
                p = p.up;

                skipnode towerNode = new skipnode(key, value);
                towerNode.left = p;
                towerNode.right = p.right;
                p.right.left = towerNode;
                p.right = towerNode;

                towerNode.down = q;
                q.up = towerNode;

                q = towerNode;
                currentLevel++;
            }
        }

        /* description of each block (around 5-10 lines) of instructions */
        // remove()locates  node at S0. If found, unlinks the node 
        // horizontally on every level the node exists through bypassing it.
        // then returns value of  successfully removed event
        public String remove(String key) {
            skipnode p = search(key);
            if (!p.key.equals(key)) {
                return null;
            }
            
            String removedValue = p.value;
            while (p != null) {
                p.left.right = p.right;
                p.right.left = p.left;
                p = p.up;
            }
            
            /* description of each block (around 5-10 lines) of instructions */
            // cleanup: checks if  top level and level just below it are 
            // both completely empty. If so, removes the extra top level by 
            // shifting  head and tail pointers down keeping one empty top.
            while (height > 0 && head.right.key.equals("+INF") && head.down.right.key.equals("+INF")) {
                head = head.down;
                tail = tail.down;
                head.up = null;
                tail.up = null;
                height--;
            }
            return removedValue;
        }

        /* description of each block (around 5-10 lines) of instructions */
        // ceilingentry() returns  first node that is greater than / equal 
        // to  given start date. If = exact date doesn't exist,  shifts 
        // one step right. used to efficiently start range queries
        public skipnode ceilingentry(String key) {
            skipnode p = search(key);
            if (p.key.equals(key)) return p;
            
            p = p.right;
            if (p.key.equals("+INF")) return null;
            return p;
        }

        /* description of each block (around 5-10 lines) of instructions */
        // Submap() gathers all events between two dates (inclusive). 
        //  jumps to the ceiling entry and walks right across  bottom level 
        // S0, adding items to  ArrayList until end date is passed.
        public ArrayList<skipnode> Submap(String key1, String key2) {
            ArrayList<skipnode> result = new ArrayList<>();
            skipnode start = ceilingentry(key1);
            
            if (start == null) return result;
            
            skipnode curr = start;
            while (!curr.key.equals("+INF") && curr.key.compareTo(key2) <= 0) {
                result.add(curr);
                curr = curr.right;
            }
            return result;
        }

        /* description of each block (around 5-10 lines) of instructions */
        // printtheList() loops from the top level down to S0, printing every node 
        // on that level to match the required format exactly. It identifies 
        // completely empty levels and formats the key:value string outputs.
        public void printtheList() {
            skipnode levelStart = head;
            int h = height;
            
            while (levelStart != null) {
                System.out.print("(S" + h + ")");
                skipnode curr = levelStart.right;
                
                if (curr.key.equals("+INF")) {
                    System.out.println(" empty");
                } else {
                    while (!curr.key.equals("+INF")) {
                        System.out.print(" " + curr.key + ":" + curr.value);
                        curr = curr.right;
                    }
                    System.out.println();
                }
                levelStart = levelStart.down;
                h--;
            }
        }
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Please provide the action file.");
            return;
        }

        skipList timeline = new skipList();

        /* description of each block (around 5-10 lines) of instructions */
        //  main method reads the actions text file line by line.  splits 
        // the line by spaces to determine command + arguments, then 
        // delegates to appropriate skipList method then formats the output
        try {
            Scanner scan = new Scanner(new File(args[0]));
            while (scan.hasNextLine()) {
                String line = scan.nextLine().trim();
                if (line.isEmpty()) continue;
                
                String[] parts = line.split(" ");
                String action = parts[0];

                /* description of each block (around 5-10 lines) of instructions */
                // handles adding / deleting events. uses put() and remove() 
                // methods of custom skipList. validates whether removal 
                // was successful to output the proper 'success' /'noDateError' tag
                if (action.equals("AddEvent")) {
                    timeline.put(parts[1], parts[2]);
                    System.out.println(line + " success");
                } 
                else if (action.equals("DeleteEvent")) {
                    String removed = timeline.remove(parts[1]);
                    if (removed != null) {
                        System.out.println(line + " success");
                    } else {
                        System.out.println(line + " noDateError");
                    }
                } 
                /* description of each block (around 5-10 lines) of instructions */
                // handles singular event display. calls O(log N) get() method 
                // to fetch event value. if event is null, cleanly 
                // outputs 'none'
                else if (action.equals("DisplayEvent")) {
                    String event = timeline.get(parts[1]);
                    if (event != null) {
                        System.out.println(line + " " + event);
                    } else {
                        System.out.println(line + " none");
                    }
                } 
                /* description of each block (around 5-10 lines) of instructions */
                // handles full timeline / range displays. uses the submap() 
                // method with artificial boundary dates ("0000" and "9999") for 
                // unbounded queries, going through  returned ArrayList
                else if (action.equals("DisplayAllEvents")) {
                    System.out.print(action);
                    ArrayList<skipnode> all = timeline.Submap("0000", "9999");
                    if (all.isEmpty()) System.out.println(" none");
                    else {
                        for (skipnode n : all) System.out.print(" " + n.key + ":" + n.value);
                        System.out.println();
                    }
                }
                else if (action.equals("DisplayEventsBetweenDates")) {
                    System.out.print(line);
                    ArrayList<skipnode> results = timeline.Submap(parts[1], parts[2]);
                    if (results.isEmpty()) System.out.println(" none");
                    else {
                        for (skipnode n : results) System.out.print(" " + n.key + ":" + n.value);
                        System.out.println();
                    }
                }
                else if (action.equals("DisplayEventsFromStartDate")) {
                    System.out.print(line);
                    ArrayList<skipnode> results = timeline.Submap(parts[1], "9999");
                    if (results.isEmpty()) System.out.println(" none");
                    else {
                        for (skipnode n : results) System.out.print(" " + n.key + ":" + n.value);
                        System.out.println();
                    }
                }
                else if (action.equals("DisplayEventsToEndDate")) {
                    System.out.print(line);
                    ArrayList<skipnode> results = timeline.Submap("0000", parts[1]);
                    if (results.isEmpty()) System.out.println(" none");
                    else {
                        for (skipnode n : results) System.out.print(" " + n.key + ":" + n.value);
                        System.out.println();
                    }
                }
                else if (action.equals("PrintSkipList")) {
                    System.out.println("PrintSkipList");
                    timeline.printtheList();
                }
            }
            scan.close();
        } catch (FileNotFoundException e) {
            System.out.println("Action file not found.");
        }
    }
}