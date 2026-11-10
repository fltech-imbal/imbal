
import java.util.ArrayList;
public class SkipList{

    /*
    SKIPLIST
    A skiplist consists of a collection of rows (1-n), starting
    from the ground up
    Each of these rows is a simple double-linked list
    Each one has a positive max and a negative max
    Numbers are added to the 1st row and to subsequent rows
    */
    public class Node{
        public Entry element;//Stores the element itself
        public Node up;
        public Node down;
        public Node prev;
        public Node next;


        public Node(Entry ele){//For an element with undefined connections
            this.element = ele;
            this.up = null;
            this.down = null;
            this.prev = null;
            this.next = null;
        }

        public Entry getElement(){
            return this.element;
        }
        public Node getUp(){
            return this.up;
        }
        public Node getDown(){
            return this.down;
        }
        public Node getPrev(){
            return this.prev;
        }
        public Node getNext(){
            return this.next;
        }

        public void setElement(Entry ele){
            this.element = ele;
        }
        public void setUp(Node newEle){
            this.up = newEle;
        }
        public void setDown(Node newEle){
            this.down = newEle;
        }
        public void setPrev(Node newEle){
            this.prev = newEle;
        }
        public void setNext(Node newEle){
            this.next = newEle;
        }
    }

    public Node head;
    public int curLevels;

    public SkipList(){
        //Populates an empty node with 2 levels, like so:
        /*
        negInf - inf (always empty)
        negInf - inf */
        this.head = new Node(new Entry("", Integer.MIN_VALUE));//Creates Upper Left
        this.head.down = new Node(new Entry("", Integer.MIN_VALUE));//Creates Lower Left
        this.head.down.setUp(this.head);
        this.head.next = new Node(new Entry("", Integer.MAX_VALUE));//Creates Upper Right
        this.head.next.setPrev(this.head);
        this.head.down.next = new Node(new Entry("", Integer.MAX_VALUE));//Creates Lower Right
        this.head.down.next.setPrev(this.head.down);
        this.head.next.setDown(this.head.down.next);
        this.head.down.next.setUp(this.head.next);



        this.curLevels = 1;//Starts at 0, includes the topmost level :)
        /*
        1: ---
        0: --- */
    }

    public String get(int thisKey){
        /*
        ----------------------get----------------------------------
        goal: return the value of the entry with the associated key
        method:  if the key is SMALLER than the next one, go down.
                if the key is LARGER than the next one, go next
        */
        Node curNode = this.head;
        while(curNode != null){
            //Checks if the current Node holds the element w. the key
            if(curNode.getElement().getKey() == thisKey){
                return curNode.getElement().getText();
            }
            else{
                //if the key is SMALLER than the next one, go down.
                if(thisKey < curNode.getNext().getElement().getKey()){
                    curNode = curNode.getDown();
                }
                //if the key is LARGER than the next one, go next
                else{
                    curNode = curNode.getNext();
                }
            }
        }
        return null;
    }
    public String put(int thisKey, String thisValue){
        /*
        -------------------put------------------------
        creates a new Entry with the given values
        afterwards, finds the next sufficient location for it
        the sufficient location is just prior to the first key which
        is larger than thisKey
        */
        Entry newEntry = new Entry(thisValue, thisKey);//NOT A NODE
        Node curNode = this.head;
        //Goes to the bottom of the node
        while(curNode.getDown() != null){
            curNode = curNode.getDown();
        }
        //Current objective:
        //While in the bottom of the list, we're trying to find the first node which is larger than newEntry
        while(curNode.getNext() != null){
            //The instance in which the current node is getting replaced
            if(curNode.getNext().getElement().getKey() == thisKey){
                String oldValue = curNode.getNext().getElement().getText();
                curNode.getNext().setElement(newEntry);
                return oldValue;
            }
            else if(curNode.getNext().getElement().getKey() > thisKey){//A satisfactory location has been found
                // curNode - newEntry - newNext;
                Node newNode = new Node(newEntry);
                Node newNext = curNode.getNext();

                curNode.setNext(newNode);
                newNode.setPrev(curNode);
                newNode.setNext(newNext);
                newNext.setPrev(newNode);
                //Determines how many upper rows there will be
                int numUp = FakeRandHeight.get();//Returns a value between 0-4, 0 meaning just the bottom layer
                //System.out.println("Entry: "+thisValue);
                //System.out.println("numUp: " + numUp);

                /*
                Current objective:
                Using the result of numUp, populate the skiplist upwards
                */
                if(numUp > this.curLevels- 1){//Eg: curLevels is 0, numUp is 1
                    int remaining = numUp;
                    int newCurLevels = this.curLevels;
                    while(remaining > this.curLevels - 1){
                        Node oldHead = this.head;
                        Node oldTail = this.head.next;
                        while(oldTail.getNext() != null){
                            oldTail = oldTail.getNext();//Gets the rightmost end of the furthermost node
                        }
                        oldHead.setUp(new Node(new Entry("", Integer.MIN_VALUE)));
                        Node newHead = oldHead.getUp();
                        this.head = newHead;
                        oldTail.setUp(new Node(new Entry("", Integer.MAX_VALUE)));
                        Node newTail = oldTail.getUp();
                        newHead.setNext(newTail);
                        newTail.setPrev(newHead);
                        newHead.setDown(oldHead);
                        newTail.setDown(oldTail);
                        newCurLevels++;
                        remaining--;
                    }
                    this.curLevels = newCurLevels;

                }
                Node newNodeUp = newNode;
                for(int i = 0; i < numUp; i++){
                    //POPULATES THE UPPER  BITS OF THE SKIPLIST
                    newNodeUp.setUp(new Node(newEntry));//here she is :)
                    newNodeUp.getUp().setDown(newNodeUp);//Attatches her to her 'child'
                    Node checkPrev = newNodeUp.getPrev();
                    Node checkNext = newNodeUp.getNext();
                    //If the prev/next of the prior are not null, sets them on the upper level
                    //Next objective: attatching the new one to the prev/next's uppers... if they exist
                    

                    while(checkPrev.getUp() == null && checkPrev.getPrev() != null){
                        checkPrev = checkPrev.getPrev();
                    }

                    while(checkNext.getUp() == null && checkNext.getNext() != null){
                        checkNext = checkNext.getNext();
                    }

                    newNodeUp.getUp().setPrev(checkPrev.getUp());
                    newNodeUp.getUp().setNext(checkNext.getUp());
                    checkPrev.getUp().setNext(newNodeUp.getUp());
                    checkNext.getUp().setPrev(newNodeUp.getUp());
                    //Repeats for the next uppermost level
                    newNodeUp = newNodeUp.getUp();
                }
                return null;
            }
            curNode = curNode.getNext();
        }
        return null;
    }
    public void print(){
        /*
        Format:
        S(h) empty
        S(h-1) ...
        .
        .
        .
        S(0) ...
        */
        Node curNode = this.head;
        for(int i = curLevels; i >= 0; i--){
            Node curHead = curNode;//Saves the head for the sake of future iterations
            System.out.print(" S(" +i+ ") ");
            while(curNode != null){
                if(curNode.getElement().getKey() == Integer.MIN_VALUE && curNode.getNext().getElement().getKey() == Integer.MAX_VALUE){
                    System.out.print("empty");
                    break;
                }
                if(curNode.getElement().getKey() == Integer.MAX_VALUE || curNode.getElement().getKey() == Integer.MIN_VALUE){
                    curNode = curNode.getNext();
                    continue;
                }
                if(curNode.getElement().getKey() <= 999){
                    System.out.print("0"+curNode.getElement().getKey() +":"+ curNode.getElement().getText()+" ");
                }
                else{
                    System.out.print(curNode.getElement().getKey() +":"+ curNode.getElement().getText()+" ");
                }
                curNode = curNode.getNext();
            }
            System.out.println();
            curNode = curHead.getDown();
            
        }
    }
    public void printAll(){
        Node currNode = this.head;
        while(currNode.getDown() != null){//Gets to the bottom of the list
            currNode = currNode.getDown();
        }
        currNode = currNode.getNext();
        while(currNode.getElement().getKey() != Integer.MAX_VALUE){
            if(currNode.getElement().getKey() >= 999){
                System.out.print("0"+currNode.getElement().getKey()+":"+currNode.getElement().getText()+" ");
            }
            else{
                System.out.print(currNode.getElement().getKey()+":"+currNode.getElement().getText()+" ");
            }
            currNode = currNode.getNext();
        }
    }
    public String remove(int thisKey){
            /*
            ----------------------remove----------------------------------
            the exact same objective as get, however removes the node and all of it's upper nodes
            */
            if(thisKey == Integer.MIN_VALUE || thisKey == Integer.MAX_VALUE){
                return null;
            }
            Node curNode = this.head;
            while(curNode != null){
                //Checks if the current Node holds the element w. the key
                if(curNode.getElement().getKey() == thisKey){//Found the node to be removed
                    String returnString = curNode.getElement().getText();// Will help us later
                    while(curNode != null){
                        Node lowerNode = curNode.getDown();
                        curNode.getPrev().setNext(curNode.getNext());
                        curNode.getNext().setPrev(curNode.getPrev());
                        curNode = lowerNode;
                        this.removeExtra();
                    }
                    //this.removeExtra();
                    return returnString;
                }
                else{
                    //if the key is SMALLER than the next one, go down.
                    if(thisKey < curNode.getNext().getElement().getKey()){
                        curNode = curNode.getDown();
                    }
                    //if the key is LARGER than the next one, go next
                    else{
                        curNode = curNode.getNext();
                    }
                }
            }
            return null;
        }
    public Entry floorEntry(int thisKey){
            /*
            --------------floorEntry------------
            The objective this method is to return the entry *directly before* the key value,
            particularly useful if the actual key does not exist. 
            */
            Node currNode = this.head;
            while(currNode.getDown() != null){
                if(currNode.getElement().getKey() == thisKey){
                    return currNode.getElement();//This is awkward
                }
                else{
                    //if the key is SMALLER than the next one, go down.
                    if(thisKey < currNode.getNext().getElement().getKey()){
                        currNode = currNode.getDown();
                    }
                    //if the key is LARGER than the next one, go next
                    else{
                        currNode = currNode.getNext();
                    }
                }                
            }
            return currNode.getElement();
        }
    public Entry ceilingEntry(int thisKey){
            /*
            --------------ceilingEntry------------
            The objective this method is to return the entry *directly after* the key value,
            particularly useful if the actual key does not exist. 
            */
            Node currNode = this.head;
            while(currNode.getDown() != null){
                if(currNode.getElement().getKey() == thisKey){
                    return currNode.getElement();//This is awkward
                }
                else{
                    //if the key is SMALLER than the next one, go down.
                    if(thisKey < currNode.getNext().getElement().getKey()){
                        currNode = currNode.getDown();
                    }
                    //if the key is LARGER than the next one, go next
                    else{
                        currNode = currNode.getNext();
                    }
                }                
            }
            return currNode.getNext().getElement();
        }
    public ArrayList<Entry> subMap(int key1, int key2){

        if(key1 > key2){//Impossible
            return null;
        }
        Node currNode = this.head;
        while(currNode.getDown() != null){
            //Goes to the bottom of the list
            currNode = currNode.getDown();
        }
        while(currNode.getNext().getElement().getKey() < key1){
            //Continues along until the first end of the node sublist is met
            currNode = currNode.getNext();
        }
        currNode = currNode.getNext();
        //At this point currNode should be greater than or equal to key1 (inclusive!)
        ArrayList<Entry> subList = new ArrayList<>();

        while(currNode.getElement().getKey() != Integer.MAX_VALUE && currNode.getElement().getKey() <= key2){
            subList.add(currNode.getElement());
            currNode = currNode.getNext();
        }
        return subList;
    }
    public void removeExtra(){
        /*
        ----removeExtra-----
        Checks if the SkipList has 2 empty rows,
        if so, removes the topmost one
        */
        if(this.head.getDown().getNext().getElement().getKey() == Integer.MAX_VALUE){//If we have two 'empty' nodes in a row
            Node newHead = this.head.getDown();
            Node newTail = newHead.getNext();
            //Removes all connections to the topmost level - the rest will be dealt with via garbage collection :)
            newHead.setUp(null);
            newTail.setUp(null);
            this.head = newHead;
            this.curLevels--;//Decrements the number of levels :)
        }
    }
}
