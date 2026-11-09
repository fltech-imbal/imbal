
//Author: Keidgiler Claimon 
//Course: CSE2010 
//Lab: 9:30 am 
//Description: a SkipList named"MySkipList". It is an original skipList and does not use the doublyLinkedList<Node>.

public class MySkipList {
    //variables in the class that must be stated at the top. Will be described lower in the program.
    Node start; 
    int height; 
    FakeRandHeight randHeight = new FakeRandHeight();
    
    private static class Node{  
        //Creates an Entry.
        private Entry entry;

        //Creates a true Node.
        public Node(int key, String value){ 
            //entry will take the nodes "key" and "value" to access the entry's operations
           entry = new Entry(key, value); 
           
           

        }  

        public String toString() { 
            //Use StringBuilder to store nodes.
            StringBuilder nodeWord = new StringBuilder(); 
            //Appends in specific String.format("%04d"), which allows the integer to be formatted with 4 numbers.
            nodeWord.append(String.format("%04d", entry.getKey()) + ":" + entry.getValue()); 


            return nodeWord.toString();
        }

        //These nodes are used as directions, and will be used later. 
        // They do not have a set or get method since everything is stored in a 
        //singular class.
        Node above; 
        Node below; 
        Node before; 
        Node after;   

        

         

    } 

    //createNewLayer creates a layer that is placed on top of the last list. 
    // This is used in many of the methods, but specifically for the variable "start".
    private Node createNewLayer() { 
        //minKey takes the smallest value and stores it as "key".
        Node minKey = new Node(Integer.MIN_VALUE, null); 
        //maxKey takes the largest value and stores it as "key". 
        Node maxKey = new Node(Integer.MAX_VALUE, null);  

        //minKey and maxKey need a way to traverse. 
        // This is why minKey's after points to maxKey and maxKey's before 
        // points to minKey.
        
        minKey.after = maxKey; 

        maxKey.before = minKey; 

        //minKey must be the value that is returned, because 
        // it is the top left node that will be the start position.
        return minKey;
    } 

    //MySkipList() creates a skipList. 
    MySkipList() { 
        //height will be explained properly in put, but right now it is at 0.
        height = 0;  
        //"start", as mentioned before, is equal to "createNewLayer()".
        start = createNewLayer(); 

    }  


    //public void put(int key, String value) first implements a random height, 
    // then (if it is greater than the original height), becomes the height while 
    // only one new layer is made.
    public void put(int key, String value) {
        int heightInsert = randHeight.get(); 
        System.out.println(); 
        System.out.println("Operation: put()");

        if (heightInsert >= height) {
            heightInsert = height; 
            Node newLayerMinKey = createNewLayer();  

            //Node newLayerMinKey's below is tethered to start's above.  
            //This goes four ways, so the nodes will be able to go 
            // up and down from each layer. This is important for 
            // traversing the skipList.
            start.above = newLayerMinKey; 
            newLayerMinKey.below = start; 

            newLayerMinKey.after.below = start.after; 
            start.after.above = newLayerMinKey.after;  

            //When the nodes are tethered, start can safely becoome newLayerMinKey without
            //losing any layers.
            
            start = newLayerMinKey;  
            //height increases by 1.
            height++;
        }  

        //Then we begin the second task, where 
        // Node current, Node lastInsertedNode, and int currentHeight are 
        // initialized.

        Node current = start; 
        //System.out.println(current.entry.getKey()); 
        int currentHeight = height; 
        Node lastInsertedNode = null; 

        while (current != null) { 
            //current's key must check if it is lesser than key to scan forward.
            if (current.after.entry.getKey() <= key) {
                current = current.after; 
                
            } else {  
                
                if (currentHeight <= heightInsert) {
                    //inside this if-statement, a new node is 
                    // initialized to hold "key" and "value".
                    Node newNode = new Node(key, value); 
                    //After this, newNode is tethered between current and current's after.
                    current.after.before = newNode; 
                    newNode.after = current.after;
                    
                    current.after = newNode; 
                    newNode.before = current; 
                    
                    
                    if (lastInsertedNode != null) { 
                        //lastInsertedNode is here to make sure the node can safely traverse 
                        //to the bottom without losing the "key".
                        lastInsertedNode.below = newNode; 
                        newNode.above = lastInsertedNode;
                    } 
                    //When this is done, lastInsertedNode becomes newNode.
                    lastInsertedNode = newNode;  
                    


                } 
                //current goes down, and current's heightlevel decreases.
                current = current.below; 
                currentHeight--;
            }
        } 

    }

        //public String toString has two nodes. 
        // One node goes from top to bottom, while the other goes from left to right. 
        //The left to right node will read the nodes and add 
        // them to the StringBuilder, acting as a "current". 
    public String toString() {
        //create a StringBuilder  
        System.out.println();
        System.out.println("Operation: toString()");

        StringBuilder fullWord = new StringBuilder(); 
            
        //Node that goes from the top layer to the bottom layer 
        System.out.println("This is in the second layer: " + start.below.after.entry.getKey());
        Node heightUpDown = start; 
        System.out.println("This is heightUpDown: " + heightUpDown.below.after.entry.getKey());
        //create a Node that goes from left to right. 
        Node leftToRight;  
        int counter = height; 

        
        //build what the nodes do in between all of this. 
        while (heightUpDown != null) { 
           //leftToRight must equal height up down. 
           leftToRight = heightUpDown;  
           //fullWord will start the list by appending.
           fullWord.append("(S"+counter+")"); 
           //wordCounter will be used as a tracker for how many nodes were 
           //counted.
           int wordCounter = 0; 
           
           
           while (leftToRight.after != null) { 
                //While leftToRight is not next to null, 
                // it will continue to scan forward.
                leftToRight = leftToRight.after;  
                //The if-statement has to make sure that the fullWord does not append the maxKey.
                if (leftToRight.entry.getKey() != Integer.MAX_VALUE) {
                    //Like before, String.format("%04d") is used to format the node correctly.
                    fullWord.append(" " + String.format("%04d", leftToRight.entry.getKey()) + ":" + leftToRight.entry.getValue()); 
                    //after fullWord appends, the counter goes up.
                    wordCounter++;
                    
                } 
           }   
           //After this while loop, the if-statement asks if wordCounter was raised.
           if (wordCounter == 0) {
                fullWord.append(" empty");
           }
           //It will also append so that the layer can end.
           fullWord.append("\n");  
           //height goes down and counter decreases.
           heightUpDown = heightUpDown.below; 
           counter--;
        }

        String result = fullWord.toString();  
        

        return result; 


    } 


    //public Node get(int key) starts below the 
    // actual start to find and retrieve the node.
    public Node get(int key) { 
        
        
        Node current = start.below; 
        

        while (current != null) { 
            //checks if current's key is lesser than key and scans forward. 
            if (current.after.entry.getKey() <= key) { 
                current = current.after; 
               //checks if current's key is equal to key and returns it 
                if (current.entry.getKey() == key) { 
                    
                    return current;
                } 
                //"else if"checks if there is anything below.
            } else if (current.below != null) {
                current = current.below; 
                //"else" returns null if nothing could be found.
            } else { 
                
                return null;
            }
            
        } 

        
        return current;
    }  

    //getToString is used after the get method in case any node needed to be 
    // returned as a string without breaking the program.
    public String getToString(int key) {
         
        Node nodeWord = get(key); 

        return nodeWord.toString();
    }

    //remove(key) also uses get(key), but for removal instead.
    public Node remove(int key) {
        
        
        Node current = get(key);  
        
        while (current != null) { 
            System.out.println("Goodbye, " + current.entry.getKey() + "!"); 
            //node current's before and after equal each other, cutting them off.
            current.before.after = current.after; 
            current.after.before = current.before; 
            if (current.below != null) { 
                //and it continues to go down and repeat the process.
                current = current.below;
            } else { 
                
                return current;
            }
        } 
        //This method actually never hits null. 
        // Despite this, the program works?
        return current;
    } 

    //floorEntry(int key) isn't actually used in the main function, 
    // since subMap was used instead. However, floorEntry still returns
    //the smallest node greater than it. 
    public int floorEntry(int key) {
        int floor = 0;  
        //floor is equal to 0, the least integer of positive integers. 

        //All the nodes will be initialized the same way as toString().

        Node heightUpDown = start; 
        Node leftToRight; 

        while (heightUpDown != null) {
            
            leftToRight = heightUpDown; 

            
            while(leftToRight.after != null) {  
                if (leftToRight.entry.getKey() != Integer.MAX_VALUE) { 
                    //The change happens here: the if statement checks if the key fulfills the 
                    //conditions, and becomes the newest floor. 
                    //It repeats this process until every node is checked in 
                    //the skipList.
                    if(leftToRight.entry.getKey() > floor && leftToRight.entry.getKey() <= key) {
                        floor = leftToRight.entry.getKey();
                    }
                    leftToRight = leftToRight.after;  
                }
                    

            } 

            heightUpDown = heightUpDown.below;
            
        }

        
        return floor;
    } 
    
    //ceilingEntry(int key) was also never used, and operates
    //the same way as floorEntry(key), just under 
    //different conditions.
    public int ceilingEntry(int key) { 
        
        //ceiling takes the max value, the largest positive integer.
        int ceiling = Integer.MAX_VALUE;  
        System.out.println();
        System.out.println("Operation: ceilingEntry()");

        Node heightUpDown = start; 
        Node leftToRight; 

        while (heightUpDown != null) {
            
            leftToRight = heightUpDown; 

            
            while(leftToRight.after != null) {  
                if (leftToRight.entry.getKey() != Integer.MAX_VALUE) { 
                    //This if statement checks current key is greater than or equal to 
                    //the desired "key".
                    if(leftToRight.entry.getKey() >= key) {  
                        //Then it checks if it is less than ceiling.
                       if(leftToRight.entry.getKey() < ceiling) { 
                            //This satisfies both conditions 
                            ceiling = leftToRight.entry.getKey();
                       } 
                       //This will repeat until every node has been read.
                    }
                    leftToRight = leftToRight.after;  
                }
                    

            } 
            heightUpDown = heightUpDown.below; 
            
        }

        
        return ceiling;
    }   


    //displayNodes is used as a way to display all the nodes. 
    // It is not in order by number, unfourtunately. 
    public String displayNodes() {
        //StringBuilder nodeWord will be used.
        StringBuilder nodeWord = new StringBuilder(); 
        //Node holder will also be initialized. 
        Node holder; 
        
        //this will start the same way as toString()
        Node leftToRight; 
        Node heightUpDown = start; 

        while (heightUpDown != null) {
            leftToRight = heightUpDown; 

            while(leftToRight.after != null) {
                if(leftToRight.entry.getKey() != Integer.MIN_VALUE && leftToRight.entry.getKey() != Integer.MAX_VALUE) {
                    //The change happens here: holder equals leftToRight, and can
                    //be appended if it does not match any other word in the string.
                    holder = leftToRight; 
                    if (!nodeWord.toString().contains(holder.toString())) {
                        nodeWord.append(holder.toString() + " "); 
                        

                    }  

                    
                } 
                leftToRight = leftToRight.after; 
                
            } 

            heightUpDown = heightUpDown.below;
        } 

        return nodeWord.toString();
    } 


    //subMap is used to print out every key in between key1 and key2.
    public String subMap(int key1, int key2) { 
        //floor and ceiling are intialized because it is 
        // easier to remember by words.
        int floor = key1; 
        int ceiling = key2;   

        

        StringBuilder fullWord = new StringBuilder(); 
        Node holder; 
       
        //Starts the same way as toString()
        Node leftToRight; 
        Node heightUpDown = start; 

        while (heightUpDown != null) {
            leftToRight = heightUpDown; 
            
            while(leftToRight.after != null) {
                if(leftToRight.entry.getKey() != Integer.MIN_VALUE && leftToRight.entry.getKey() != Integer.MAX_VALUE) {
                    holder = leftToRight; 
                    //The change happens here: this if-statement checks if holder
                    //is greater than floor and lesser than ceiling.
                    if (floor < holder.entry.getKey() && holder.entry.getKey() < ceiling) {
                        if (!fullWord.toString().contains(holder.toString())) {
                            fullWord.append(holder.toString() + " "); 
                            
                        }  
                    } 

                    
                } 
                leftToRight = leftToRight.after; 
                
            } 

            heightUpDown = heightUpDown.below;

            
        } 

        return fullWord.toString();


    } 

    





  //Thus, the skipList is finished.          
}


