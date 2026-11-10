 /*Author: sureena singh
  Email: sureena2025@my.fit.edu
  Description: The goal of this file is to manage the timeline and allow the
user to specify a time range to display/share his/her major
events*/
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
public class HW5 {
   public static class Entry{
    private int key;
    private String value;
    Entry(int key,String value){
        this.key = key;
        this.value= value;
    }
   }
    public static class Node{
        Entry entry;
        Node[] next;
        Node(Entry entry, int levels){
            this.entry = entry;
            this.next =  new Node[levels];
        }
   }
   /* SkipList stores the events in sorted order by date. Max_level is the number of levels available,
   head is the starting node, top is the highest level currently in use and rand supplies the heights for new nodes. */
   public static class SkipList{
    int Max_level =11;
    Node head = new Node(null, Max_level);
    int top =0;
    FakeRandHeight rand = new FakeRandHeight();
    /* findPredecessors starts at the top level and moves right while the key is smaller than the given key, then drops down a level. 
    Returns the last level-0 node with a key smaller than the given key.If update is not null, it also records the last node visited 
    at each level so put and remove know which pointers to change. */
    private Node findPredecessors(int key, Node[] update){
        Node current = head;
        for (int lvl = top; lvl >=0; lvl--){
            while (current.next[lvl]!= null && current.next[lvl].entry.key<key){
                current = current.next[lvl];
            }if (update != null) update[lvl]= current;
        }return current;
    }
    public String get(int key){
        Node n = findPredecessors(key, null).next[0];
        if (n!= null && n.entry.key==key) return n.entry.value;
        return null;
    }
    public String put(int key,String value){
        Node[] update = new Node[Max_level];
        Node pred = findPredecessors(key, update);
        Node n = pred.next[0];
        if (n!= null && n.entry.key ==key){ // if key exists then replace the value
            String old = n.entry.value;
            n.entry.value=value;
            return old;
        }
        int h = rand.get(); // called for new keys
        Node fresh = new Node(new Entry(key,value),h+1);
        for (int i =top+1; i<=h;i++)update[i] = head;
        if (h>top) top = h;
        for (int i=0; i<=h;i++){
            fresh.next[i] = update[i].next[i];
            update[i].next[i]= fresh;
        }return null;
    }
    public String remove(int key){
        Node[] update = new Node[Max_level];
        Node n = findPredecessors(key, update).next[0];
        if (n==null || n.entry.key != key) return null;
        for (int i =0; i<n.next.length;i++){
            if (update[i].next[i]==n) update[i].next[i]= n.next[i];
        }
        while (top>0 && head.next[top]==null) top--;
        return n.entry.value;
    }
    public Entry floorEntry(int key){
        Node pred = findPredecessors(key, null);
        Node n = pred.next[0];
        if (n!= null && n.entry.key == key) return n.entry;
        return (pred ==head) ? null : pred.entry;
    }
    public Entry ceilingEntry(int key){
        Node n= findPredecessors(key, null).next[0];
        return (n==null) ? null : n.entry;
    }
    public List<Entry> subMap(int key1,int key2){
        List<Entry> result = new ArrayList<>();
        Node n = findPredecessors(key1, null).next[0];
        while (n!= null && n.entry.key <= key2){
            result.add(n.entry);
            n = n.next[0];
        }
        return result;
    } 
    public void printList(){
        for(int lvl = top+1;lvl >=0;lvl--){
            System.out.print("(S"+lvl+")");
            if (lvl ==top+1|| head.next[lvl]==null){
                System.out.println(" empty");
                continue;
            }
            for (Node n = head.next[lvl]; n!= null; n = n.next[lvl]){
                System.out.print(" "+fmt(n.entry));
            }System.out.println();
        }
    }
   }
   /* fmt helps format an entry as date:event, with the date being formatted to 4 digits */
   static String fmt(Entry e){
    return String.format("%04d", e.key)+":"+e.value;
   }
   static void printEntries(String prefix, List<Entry> list){
    System.out.print(prefix);
    if(list.isEmpty()){
        System.out.println(" none");
        return;
    }
    for (Entry e: list) System.out.print(" "+fmt(e));
    System.out.println();
   }
   static SkipList list = new SkipList();
   /* d4 converts an int date to MMDD string */
   public static String d4(int date){
    return String.format("%04d",date);
   }
   public static void DisplayEvent(int date){
    String ev = list.get(date);
    System.out.println("DisplayEvent "+d4(date)+" "+(ev ==null ? "none":ev));
   }
   public static void AddEvent(int date, String event){
    list.put(date,event);
    System.out.println("AddEvent "+d4(date)+" "+event+" success");
   }
   public static void DeleteEvent (int date){
    String old = list.remove(date);
    System.out.println("DeleteEvent "+d4(date)+" "+(old ==null ? "noDateError": "success"));
   }
   public static void DisplayEventsBetweenDates(int startDate, int endDate){
    printEntries("DisplayEventsBetweenDates "+d4(startDate)+" "+d4(endDate),list.subMap(startDate,endDate));
   }
   public static void DisplayEventsFromStartDate(int startdate){
    printEntries("DisplayEventsFromStartDate "+d4(startdate),list.subMap(startdate,Integer.MAX_VALUE));
   }
   public static void DisplayEventsToEndDate (int endDate){
    printEntries("DisplayEventsToEndDate "+d4(endDate),list.subMap(0,endDate));
   }
   public static void DisplayAllEvents(){
    printEntries("DisplayAllEvents", list.subMap(0, Integer.MAX_VALUE));
   }
   public static void PrintSkipList(){
    System.out.println("PrintSkipList");
    list.printList();
   }
   /* main method reads the file containing the instructions line by line, splits each line into tokens and calls the method for that 
   action. */
   public static void main(String[] args) {
       if (args.length <1){
        System.out.println("provide the action file");
        return;
       }
       try {
           Scanner sc = new Scanner(new File(args[0]));
           while (sc.hasNextLine()){
            String line = sc.nextLine().trim();
            if (line.isEmpty()) continue;
            String[] t= line.split("\\s+",3); //to deal with spaces
            switch(t[0]){
                case "DisplayEvent":
                    DisplayEvent(Integer.parseInt(t[1]));
                    break;
                case "AddEvent":
                    AddEvent(Integer.parseInt(t[1]),t[2]);
                    break;
                case "DeleteEvent":
                    DeleteEvent(Integer.parseInt(t[1]));
                    break;
                case "DisplayEventsBetweenDates":
                    String[] u = line.split("\\s+");
                    DisplayEventsBetweenDates(Integer.parseInt(u[1]), Integer.parseInt(u[2]));
                    break;
                case "DisplayEventsFromStartDate":
                    DisplayEventsFromStartDate(Integer.parseInt(t[1]));
                    break;
                case "DisplayEventsToEndDate":
                    DisplayEventsToEndDate(Integer.parseInt(t[1]));
                    break;
                case "DisplayAllEvents":
                    DisplayAllEvents();
                    break;
                case "PrintSkipList":
                    PrintSkipList();
                    break;
                default:
                    break;

            }
           }
           sc.close();
       } catch (FileNotFoundException e) {
         System.out.println("File not found");
       }
   }




}
/*

  Author: Phil Chan
  Email: pkc@cs.fit.edu
  Description:  

  For easier debugging and testing, this class provides a known
  sequence of fake random height for HW5 in CSE2010.  The sequence
  simulates each successive higher level is half as likely to be used
  in SkipListMap.put().

  The sequence has 31 numbers: 0 occurs 16 times, 1 occurs 8 times, 2
  occurs 4 times, 3 occurs 2 times, and 4 occurs 1 time.  Each number
  is returned by get() in the following order and the numbers are
  recycled if get() is called more than 31 times.

                                1 1 1 1 1 1
  index:    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5

  sequence: 0 1 0 2 0 1 0 3 0 1 0 2 0 1 0 4

            1 1 1 1 2 2 2 2 2 2 2 2 2 2 3
  index:    6 7 8 9 0 1 2 3 4 5 6 7 8 9 0

  sequence: 0 1 0 2 0 1 0 3 0 1 0 2 0 1 0  


  Usage:

  FakeRandHeight randHeight = new FakeRandHeight();

  randHeight.get()  

 */

 class FakeRandHeight
{
    static int[] height = {0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, // sequence of height
                           0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0}; 
    int count;  // number of times get() has been called 

    public FakeRandHeight()
    {
	count = 0;
    }

    // return the next number in the sequence, recycle the sequence if needed
    public int get()
    {
	return height[count++ % 31];
    }

    // testing get()
    /*
    public static void main(String[] args)
    {
	FakeRandHeight randHeight = new FakeRandHeight();

	for (int i=0; i < 40; i++)
	    {
		System.out.println(i + " " + randHeight.get());
	    }
    }
    */
}
