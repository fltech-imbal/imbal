/*
Author:Israel Caballero
Email:icaballeroes2025@my.fit.edu
Course:CSE2010-Algorithms and Data Structures
Section:01
Description of this file: Program that stores a timeline of events in a skiplist and displays events based on a time frame
*/
import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
public class HW5{
//Skip list class
public static class SkipList{
//Node class for each node in the skip list
public static class Node{
    // A node stores a key,element, and pointers to the previous,next,above and below nodes
    private int key;
    private String element;

    private Node prev;
    private Node next;
    private Node above;
    private Node below;

    public Node(int k, String e, Node p, Node n, Node a, Node b){
        key = k;
        element = e;
        prev = p;
        next = n;
        above = a;
        below = b;
    }
    //returns the key,element,previous,next,above and below elements in the node
    public int getKey(){return key;}
    public String getElement(){return element;}
    public Node getNext(){return next;}
    public Node getPrev(){return prev;}
    public Node getAbove(){return above;}
    public Node getBelow(){return below;}

    //sets the different elements in the node
    public void setElement(String e){element = e;}
    public void setNext(Node n){next = n;}
    public void setPrev(Node p){prev = p;}
    public void setAbove(Node a){above = a;}
    public void setBelow(Node b){below = b;}
}//End of Node class
    //used to generate random numbers that simulate how many heads are tossed
    private FakeRandHeight rand = new FakeRandHeight();
    //Head node, tail node,size and height of the skip list
    private Node head;
    private Node tail;
    private int size = 0;
    private int height = 0;

    //create a new skipList
    public SkipList(){
        head = new Node(Integer.MIN_VALUE,null,null,null,null,null);
        tail = new Node(Integer.MAX_VALUE,null,head,null,null,null);
        head.setNext(tail);
    }
    //returns the size and height of the skip list
    public int size(){return size;}
    public int height(){return height;}
    //checks if the skip list is empty
    public boolean isEmpty(){return size == 0;}
    //searches for a value k in the skiplist
    public Node skipSearch(int k){
        //set current position as the head node
        Node position = head;
        //runs until something is returned
        while(true){
            //if the key is bigger then the next nodes key move right
            while(k >= position.getNext().getKey()){
                position = position.getNext();
            }
            //if you can't drop down anymore return position, else move down
            if(position.getBelow() == null){return position;}
                else{
                    position = position.getBelow();
                }
        }
    }
    //puts a node into the skiplist
    public String put(int k,String e){
        //find where the key goes in the list
        Node p = skipSearch(k);
        //if the key already exists in the list, replace the value with the new key and element, and return the old value
        if(p.getKey() == k){
            String old = p.getElement();
            while(p != null){
                p.setElement(e);
                p = p.getAbove();
            }
            return old;
        }
        //picks the height of the tower
        int h = rand.get();
        //while the generated height is bigger or equal to the skip lists current height
        while(h >= height){
            //makes a new head/tail pair for the top of the tower and connects them to the pair below
            Node oldHead = head;
            Node oldTail = tail;
            head = new Node(Integer.MIN_VALUE,null,null,null,null,null);
            tail = new Node(Integer.MAX_VALUE,null,head,null,null,null);
            head.setNext(tail);

            head.setBelow(oldHead);
            oldHead.setAbove(head);
            tail.setBelow(oldTail);
            oldTail.setAbove(tail);
            height++;
            }
            //builds the tower from bottom to top
            Node q = null;
            for(int i = 0; i <= h; i++){
                //creates a new node with previous equal to p and next equals to p's next
                Node n = new Node(k,e,p,p.getNext(),null,q);
                p.getNext().setPrev(n);
                p.setNext(n);
                if(q != null){
                    q.setAbove(n);
                }
                q = n;
                //move p up one level for another pass
                if(i < h){
                    while(p.getAbove() == null){
                        p = p.getPrev();
                    }
                    p = p.getAbove();
                }
            }
            size++;
            return null;
        }
    //prints the skip list
    public void printSkipList(){
        //starts at the top head and moves down from height to 0
        Node levelsHead = head;
        for(int h = height; h >= 0; h--){
            //prints the height and makes node the node after the head
            System.out.print("(S" + h + ")");
            Node node = levelsHead.getNext();
            //if there are no items in the lists level print empty
            if(node.getKey() == Integer.MAX_VALUE){
                System.out.println(" empty");
            }
            //print each node until there are no nodes left
            else{
                while(node.getKey() != Integer.MAX_VALUE){
                    System.out.print(" " + String.format("%04d", node.getKey()) + ":" + node.getElement() );
                    node = node.getNext();
                }
                System.out.println();
            }
        //moves to the level below
        levelsHead = levelsHead.getBelow();
        }
    }
    //returns the element of a given key
    public String get(int k){
        Node p = skipSearch(k);
        if(p.getKey() == k) return p.getElement();
        else return null;
    }
    //returns the node with the largest key that is <= k
    public Node floorEntry(int k){
        Node p = skipSearch(k);
        if(p.getKey() == Integer.MIN_VALUE){
            return null;
        }
        return p;
    }
    //returns the node with the smallest key that is >= k
    public Node ceilingEntry(int k){
        Node p = skipSearch(k);
        if(p.getKey() == k) return p;
        else{
            if(p.getNext().getKey() == Integer.MAX_VALUE) return null;
            else return p.getNext();
        }
    }
    //Removes a entry from the skip list
    public String remove(int k){
        //find the key if in the list and set its element to old 
        Node p = skipSearch(k);
        if(p.getKey() != k) return null;
        String old = p.getElement();
        //while there are nodes to remove, unlink the towers from bottom to top
        while(p != null){
            Node up = p.getAbove();
            p.getPrev().setNext(p.getNext());
            p.getNext().setPrev(p.getPrev());
            p = up;
        }
        //If there are more than one empty lists at the top remove until there is only one
        while(height > 0 && head.getBelow().getNext().getKey() == Integer.MAX_VALUE){
            head = head.getBelow();
            tail = tail.getBelow();
            head.setAbove(null);
            tail.setAbove(null);
            height --;
        }
        size--;
        return old;
    }
    //returns a list of Nodes between two dates
    public ArrayList<Node> subMap(int k1, int k2){
        Node p = ceilingEntry(k1);
        ArrayList<Node> list = new ArrayList<Node>();

        while(p != null && p.getKey() <=k2){
            list.add(p);
            p = p.getNext();
        }
        return list;
    }
    
}//End of skip list class
public static void main(String[] args)throws Exception{
    //creates the scanner and skiplist
    Scanner input = new Scanner(new File(args[0]));
    SkipList list = new SkipList();
    //while there are lines in the file
    while(input.hasNextLine()){
        //split lines into tokens for each word
        String line = input.nextLine();
        String[] token = line.split("\\s+");
        //if the first word is "AddEvent"
        if(token[0].equals("AddEvent")){
            //call the put method with the date and event then print success
            int date = Integer.parseInt(token[1]);
            String event = token[2];
            list.put(date, event);
            System.out.println(line + " success");
        }
        //If first word is "DisplayEvent"
        if(token[0].equals("DisplayEvent")){
            //finds if a date has an event, print none if there isnt and print the event if there is
            int date = Integer.parseInt(token[1]);
            String event = list.get(date);
            if(event == null){
                System.out.println(line + " none");
            }
            else System.out.println(line + " " + event);
        }
        //First word is "PrintSkipList"
        if(token[0].equals("PrintSkipList")){
            //prints skip list through method
            System.out.println(line);
            list.printSkipList();
        }
        //First word is "DeleteEvent"
        if(token[0].equals("DeleteEvent")){
            //Delets event using remove method
            int date = Integer.parseInt(token[1]);
            list.remove(date);
            System.out.println(line + " success");
        }
        //First word is "DisplayAllEvents"
        if(token[0].equals("DisplayAllEvents")){
            //creates an ArrayList with events from 1 am to 12 am
            ArrayList<SkipList.Node> allEvents =list.subMap(101, 1231);
            System.out.print(line);
            //if list is empty print none, if not empty print each item in the list
            if (allEvents.isEmpty()) System.out.print(" none");
            for(int i = 0; i <allEvents.size();i++){
                System.out.print(" " + String.format("%04d", allEvents.get(i).getKey()) + ":" + allEvents.get(i).getElement());
            }
            System.out.println();
        }
        //First word is "DisplayEventsBetweenDates"
        if(token[0].equals("DisplayEventsBetweenDates")){
            //uses submap method to add events between two dates into a list and prints them if there are any
            int date1 = Integer.parseInt(token[1]);
            int date2 = Integer.parseInt(token[2]);
            System.out.print(line);
            ArrayList<SkipList.Node> allEvents = list.subMap(date1, date2);
            if (allEvents.isEmpty()) System.out.print(" none");
            for(int i = 0; i <allEvents.size();i++){
                System.out.print(" " + String.format("%04d", allEvents.get(i).getKey()) + ":" + allEvents.get(i).getElement());
            }
            System.out.println();
        }
        //First word is "DisplayEventsFromStartDate"
        if(token[0].equals("DisplayEventsFromStartDate")){
            //Inserts events into a list from given date to 12 am and prints
            int date = Integer.parseInt(token[1]);
            System.out.print(line);
            ArrayList<SkipList.Node> allEvents = list.subMap(date, 1231);
            if (allEvents.isEmpty()) System.out.print(" none");
            for(int i = 0; i <allEvents.size();i++){
                System.out.print(" " + String.format("%04d", allEvents.get(i).getKey()) + ":" + allEvents.get(i).getElement());
            }
            System.out.println();
        }
        //First word is "DisplayEventsToEndDate"
        if(token[0].equals("DisplayEventsToEndDate")){
            //Inserts events into a list from 1 am to a given date and prints them
            int date = Integer.parseInt(token[1]);
            System.out.print(line);
            ArrayList<SkipList.Node> allEvents = list.subMap(101, date);
            if (allEvents.isEmpty()) System.out.print(" none");
            for(int i = 0; i <allEvents.size();i++){
                System.out.print(" " + String.format("%04d", allEvents.get(i).getKey()) + ":" + allEvents.get(i).getElement());
            }
            System.out.println();
        }
    }  
}
}