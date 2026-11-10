 /*
Author: Olivia Cicerrella
Email: ocicerrella2025@fit.edu
Course: Algor and data Struc
Section: Section 1
Description of this file: this progrsm manages timelines by implementing a 
custom skip list structure for sreaching, reads user command from command line 

*/
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("unchecked")
public class HW5 {

    // // data container class to hol timeline
    public static class SkipListEntry {
        private String key;   // / sgtorezs data string in mmdd form
        private String value; // sotres the envet decription tect string

        // constructors for the new data even
        public SkipListEntry(String key, String value) {
            this.key = key;
            this.value = value;
        }

        public String getKey() { return key; } // accessor to retrive the enrty date
        public String getValue() { return value; } // accessor to get the event description
    }

    // node configuation reprents positions in the grid
   public static class SkipListNode {
        SkipListEntry entry; 
        // horizontal navigation is handled by DoublyLinkedList 
        //track vertical pillars manually with this
       SkipListNode up;// verticle pointer licked the nodes above
       SkipListNode down; // verticle pointer linking the nodes bwloew

        // node constructor for the timeline map entery
       public SkipListNode(SkipListEntry entry) {
           this.entry = entry;
       }

       // overrides the to srting method that is given to make it fit this proffram
        @Override
       public String toString() {
           return entry.key + ":" + (entry.value != null ? entry.value : "null");
       }
   }

    public static class SkipListMap {
        private static final String NEG_INF = "-Infinity"; // Left sentinell\
        private static final String POS_INF = "+Infinity"; // Right sentinel 

        // array of layers  where each layer is  the DoublyLinkedList
        private ArrayList<DoublyLinkedList<SkipListNode>> layers;
        private FakeRandHeight randHeight; // determins random sequence simulator component
        private int maxLevel;              // counter to track highest row lane 

        // map constructors
        public SkipListMap() {
            randHeight = new FakeRandHeight();
            layers = new ArrayList<>();
            clear();
        }

        // resets list back to emptry state
        private void clear() {
            layers.clear();
            maxLevel = 0;
            addEmptyLayer();
        }

        //horuizontal line genoraotr establishing positions across the whole row
        private void addEmptyLayer() {
            DoublyLinkedList<SkipListNode> newLayer = new DoublyLinkedList<>();
            
            SkipListNode negSentinel = new SkipListNode(new SkipListEntry(NEG_INF, null));
            SkipListNode posSentinel = new SkipListNode(new SkipListEntry(POS_INF, null));
            
            // push track bounds into custom structures
            newLayer.addFirst(negSentinel);
            //links them horizontally
            newLayer.addLast(posSentinel); 
            
            layers.add(newLayer); // registers row layer to collectoions
        }

        // helper to find the bottom node
        private SkipListNode search(String key) {
            //sarart at the top left sentinel position
            int currentLevel = layers.size() - 1;
            
            // manually go through your list to traverse the grid
            SkipListNode p = layers.get(currentLevel).first(); 
            
            //skip List topdown search
            while (currentLevel >= 0) {
                // to look forward horixzontaly, simulate a right shift scanloop
                while (p != null && !p.entry.key.equals(POS_INF)) {
                    // look forward if possible
                    SkipListNode nextNode = getNextNodeInLayer(currentLevel, p);
                    if (nextNode != null && !nextNode.entry.key.equals(POS_INF) && nextNode.entry.key.compareTo(key) <= 0) {
                        p = nextNode; // key is bnounded avdance forward rightward
                    } else {
                        break; // right bound condition reachs for this layer
                    }
                }
                //drop down one tear verticle if lower row is there
                if (currentLevel > 0) {
                    p = p.down;
                    currentLevel--;
                } else {
                    break; // scan t3erminated and floor
                }
            }
            return p; // retuens bottom level node
        }

        // simulates next stepping within boundaries
        private SkipListNode getNextNodeInLayer(int level, SkipListNode current) {
            DoublyLinkedList<SkipListNode> targetLayer = layers.get(level);
            int size = targetLayer.size();
            ArrayList<SkipListNode> snapshot = new ArrayList<>();
            
            // unload the row level into a temp array frame structure
            for (int i = 0; i < size; i++) {
                snapshot.add(targetLayer.removeFirst());
            }
            
            SkipListNode result = null;
            // reload the list while identifying the right hand neighbor part
            for (int i = 0; i < snapshot.size(); i++) {
                SkipListNode item = snapshot.get(i);
                targetLayer.addLast(item);
                if (item == current && i < snapshot.size() - 1) {
                    result = snapshot.get(i + 1); // slecets the node item matching sequcen
                }
            }
            return result; 
        }

        // public look up interface checks if even exits for a date
        public String get(String key) {
            SkipListNode p = search(key);
            if (p != null && p.entry.key.equals(key)) {
                return p.entry.value; // data matched pass back the descirption
            }
            return null; // target recrod miss from databacde
        }

        //inserts a new element or changes vals if data already here in maps
        public String put(String key, String value) {
            SkipListNode p = search(key);
            
            // id exact date already exists update towers vertically
            if (p != null && p.entry.key.equals(key)) {
                String oldValue = p.entry.value;
                while (p != null) {
                    p.entry.value = value;
                    p = p.up; // clibms up coloums replcing strings
                }
                return oldValue; 
            }

            int targetHeight = randHeight.get(); // pull the sequential height 
            
            // adds new empty horizontal rows if target height breaks max
            while (maxLevel < targetHeight) {
                addEmptyLayer();
                maxLevel++;
                
                // connects vertical updown anchors between the old and new layers
                int topIdx = layers.size() - 1;
                SkipListNode oldHead = (SkipListNode) layers.get(topIdx - 1).first();
                SkipListNode newHead = (SkipListNode) layers.get(topIdx).first();
                newHead.down = oldHead;
                oldHead.up = newHead;
            }

            // insert new nodes sorted horizontally across list layers
            SkipListNode lastCreatedNode = null;
            for (int lvl = 0; lvl <= targetHeight; lvl++) {
                SkipListNode newNode = new SkipListNode(new SkipListEntry(key, value));
                
                // cast the layer directly to bypass type erasure confusion
                DoublyLinkedList layer = (DoublyLinkedList) layers.get(lvl);
                ArrayList temp = new ArrayList();
                int size = layer.size();
                
                // unloads all items from layer into a temp array
                for (int i = 0; i < size; i++) {
                    temp.add(layer.removeFirst());
                }
                
                boolean inserted = false;
                // travers array and insert new nodes into the sorted locations
                for (int i = 0; i < temp.size(); i++) {
                    SkipListNode node = (SkipListNode) temp.get(i);
                    if (!inserted && !node.entry.key.equals(NEG_INF) && (node.entry.key.equals(POS_INF) || node.entry.key.compareTo(newNode.entry.key) > 0)) {
                        layer.addLast(newNode);
                        inserted = true;
                    }
                    layer.addLast(node); // reloads the left over items back into order
                }
                
                // interlock vertical pillar structures together 
                if (lastCreatedNode != null) {
                    newNode.down = lastCreatedNode;
                    lastCreatedNode.up = newNode;
                }
                lastCreatedNode = newNode; // save reference for a connection ofr later
            }
            return null; 
        }

        // structural deletion method removing elements across all tiers
        public String remove(String key) {
            SkipListNode p = search(key);
            if (p == null || !p.entry.key.equals(key)) {
                return null; //  if the key doesn't exist return error flag
            }
            
            String removedValue = p.entry.value;
            // iterate up the columns purgers matches out ofthe lists
            while (p != null) {
                for (int lvl = 0; lvl < layers.size(); lvl++) {
                    DoublyLinkedList<SkipListNode> layer = layers.get(lvl);
                    int size = layer.size();
                    // filters the lists by skipping over the targeted node 
                    ArrayList<SkipListNode> temp = new ArrayList<>();
                    for (int i = 0; i < size; i++) {
                        SkipListNode item = layer.removeFirst();
                        if (!item.entry.key.equals(p.entry.key)) {
                            temp.add(item);
                        }
                    }
                    // repopulats the list layer 
                    for (SkipListNode node : temp) {
                        layer.addLast(node);
                    }
                }
                p = p.up; // advances up to next node position
            }
            return removedValue; 
        }

        // isolates clost structuresal elements of equal metics
        public SkipListEntry floorEntry(String key) {

            SkipListNode p = search(key);
            if (p == null || p.entry.key.equals(NEG_INF)) {
                return null; // no valid calender fits perameters
            }
            return p.entry;
        }
        // isolates  clostes structs eements for equal metics again
        public SkipListEntry ceilingEntry(String key) {
            SkipListNode p = search(key);
            if (p != null && p.entry.key.equals(key)) {
                return p.entry; // direct match found retrusn entry 
            }
            if (p != null) {
                SkipListNode nextNode = getNextNodeInLayer(0, p);
                if (nextNode != null && !nextNode.entry.key.equals(POS_INF)) {
                    return nextNode.entry;
                }
            }
            return null;
        }
        // bottom range query aggregation pakcin recors between timeline
        public List subMap(String key1, String key2) {
            List list = new ArrayList<>();
            SkipListNode p = search(key1);
            if (p != null && !p.entry.key.equals(key1)) {
                p = getNextNodeInLayer(0, p);
            }
            // gather consecutive items across base row layer
            while (p != null && p.entry.key != POS_INF && p.entry.key.compareTo(key2) <= 0) {
                list.add(p.entry);
                p = getNextNodeInLayer(0, p); // shifts forward right way across base row
            }
            return list;
        }
        // visual layout formmatter wriing curreent list 
        public void printSkipList() {
            // loops descending  from top layout levels down to row lane so
            for (int lvl = layers.size() - 1; lvl >= 0; lvl--) {
                System.out.print("(S" + lvl + ")");
                DoublyLinkedList layer = layers.get(lvl);
                int size = layer.size();
                ArrayList temp = new ArrayList<>();
                // read and rotate items without corrupting the sequence inside the class
                for (int i = 0; i < size; i++) {
                    temp.add(layer.removeFirst());
                }
                int printCount = 0;
                // reload items back and output them to the console stream
                for (int i = 0; i < temp.size(); i++) {
                    HW5.SkipListNode node = (HW5.SkipListNode) temp.get(i);
                    layer.addLast(node);
                    if (!node.entry.key.equals(NEG_INF) && !node.entry.key.equals(POS_INF)) {
                        System.out.print(" " + node.entry.key + ":" + node.entry.value);
                        printCount++; // increments count for real data tracking 
                    }
                }
                if (printCount == 0) {
                    System.out.print(" empty"); // level contains only sentienals mark as empty
                }
                System.out.println(); // terminate visual row line with a new line
            }
        }
    }
        // the m ain method for this class that contains the cases for the program
        public static void main(String[] args) {
            if (args.length < 1) {
                System.err.println("Usage: java HW5 ");
                return;
            }
            String filename = args[0]; // capture command line input filename 
            SkipListMap map = new SkipListMap();
            // sstream execution tracking file reader routines
            try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
                String line;
                while ((line = br.readLine()) != null) {
                    line = line.trim();
                    if (line.isEmpty()) continue;
                        // parse input lines into maximum 3 tokens to preserve event multiword names
                        String[] tokens = line.split("\\s+", 3);
                        String action = tokens[0];
                        switch (action) {
                            case "DisplayEvent": {
                                String date = tokens[1];
                                String val = map.get(date);
                                if (val != null) {
                                    System.out.println("DisplayEvent " + date + " " + val);
                                } else {
                                    System.out.println("DisplayEvent " + date + " none");
                                }
                                break;
                            }
                            case "AddEvent": {
                                String date = tokens[1];
                                String event = (tokens.length > 2) ? tokens[2] : "";
                                map.put(date, event); // insters data nd text descriptions into grid
                                System.out.println("AddEvent " + date + " " + event + " success");
                                break;
                            }
                            case "DeleteEvent": {
                            String date = tokens[1];
                            String val = map.remove(date);
                            if (val != null) {
                                System.out.println("DeleteEvent " + date + " success");
                            } else {
                                System.out.println("DeleteEvent " + date + " noDateError");
                            }
                            break;
                            }
                            case "DisplayEventsBetweenDates": {
                                String startDate = tokens[1];
                                String endDate = tokens[2];
                                List sub = map.subMap(startDate, endDate);
                                System.out.print("DisplayEventsBetweenDates " + startDate + " " + endDate);
                                printEntries(sub);// prints coloected submap arrays side by side
                                break;
                            }
                            case "DisplayEventsFromStartDate": {
                                String startDate = tokens[1];
                                List sub = map.subMap(startDate, "1231"); // query to abslujte end of year 
                                System.out.print("DisplayEventsFromStartDate " + startDate);
                                printEntries(sub);
                                break;
                            }
                            case "DisplayEventsToEndDate": {
                                String endDate = tokens[1];
                                List sub = map.subMap("0101", endDate); // from begining of year
                                System.out.print("DisplayEventsToEndDate " + endDate);
                                printEntries(sub);
                                break;
                            }
                            case "DisplayAllEvents": {
                                List sub = map.subMap("0101", "1231");
                                System.out.print("DisplayAllEvents");
                                printEntries(sub);// pass tracking array down fto formatter
                                break;
                            }
                            case "PrintSkipList": {
                                System.out.println("PrintSkipList");
                                map.printSkipList();// trigger visual grid generation
                                break;
                            }
                            default:
                                break;
                    }
                }
            }
            catch (IOException e) {
                System.err.println("Error reading file: " + e.getMessage());
            }
        }
        // static print method to process timeline item arrays into termina view
        private static void printEntries(List<HW5.SkipListEntry> list) {
            if (list.isEmpty()) {
                System.out.println(" none");
            } else {
                for (int i = 0; i < list.size(); i++) {
                    HW5.SkipListEntry entry = list.get(i);
                    System.out.print(" " + entry.getKey() + ":" + entry.getValue());
                }
                System.out.println(); // commit line feed ending the exicution layout 
            }
        }
}
