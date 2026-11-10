/*

  Author: Avery Haughwout
  Email: ahaughwout2024@my.fit.edu
  Course: CSE2012
  Section: 3:30
  Description of this file: Creates a skipList to organize events based on a given 'key' date

 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class HW5
{
    public static void main(String[] args) throws FileNotFoundException{
        SkipList thisList = new SkipList();
        Scanner scnr1 = new Scanner(new File(args[0]));
        while(scnr1.hasNext()){
            //Reads each input
            String input = scnr1.nextLine();
            String[] inputs = input.split(" ");
            switch(inputs[0]){
                case "DisplayEvent":
                    int key = Integer.parseInt(inputs[1]);//Stores the key that will be used
                    if(thisList.get(key) == null){
                        if(key < 999){
                            System.out.println("DisplayEvent 0"+key+" none");
                        }
                        else{
                            System.out.println("DisplayEvent "+key+" none");
                        }
                    }
                    else{
                        if(key < 999){
                            System.out.println("DisplayEvent 0"+key+" "+thisList.get(key));
                        }
                        else{
                            System.out.println("DisplayEvent "+key+" "+thisList.get(key));
                        }
                    }
                    break;

                case "AddEvent":
                    //Gets the key
                    int newKey = Integer.parseInt(inputs[1]);
                    //Gets the Input
                    String event = inputs[2];
                    if(thisList.put(newKey, event) == null){
                        if(newKey < 999){
                            System.out.println("AddEvent 0"+newKey+" "+event+" success");
                        }
                        else{
                            System.out.println("AddEvent "+newKey+" "+event+" success");
                        }
                    }
                    else{
                        if(newKey < 999){
                            System.out.println("AddEvent 0"+newKey+" "+event+" fail");
                        }
                        else{
                            System.out.println("AddEvent "+newKey+" "+event+" fail");
                        }
                    }
                    break;

                case "DeleteEvent":
                    int removeKey = Integer.parseInt(inputs[1]);
                    String removeCheck = thisList.remove(removeKey);
                    if(removeCheck == null){
                        if(removeKey < 999){
                            System.out.println("DeleteEvent 0"+removeKey+" fail");
                        }
                        else{
                            System.out.println("DeleteEvent "+removeKey+" fail");
                        }
                    }
                    else{
                        if(removeKey < 999){
                            System.out.println("DeleteEvent 0"+removeKey+" success");
                        }
                        else{
                            System.out.println("DeleteEvent "+removeKey+" success");
                        }
                    }
                    break;

                case "DisplayEventsBetweenDates":
                    int key1 = Integer.parseInt(inputs[1]);
                    int key2 = Integer.parseInt(inputs[2]);
                    System.out.print("DisplayEventsBetweenDates ");
                    if(key1 > 1000){
                        System.out.print("0"+key1+" ");
                    }
                    else{
                        System.out.print(key1+" ");
                    }

                    if(key2 > 1000){
                        System.out.print("0"+key2+" ");
                    }
                    else{
                        System.out.print(key2+" ");
                    }
                    ArrayList<Entry> entries = thisList.subMap(key1, key2);
                    if(entries.size() == 0){
                            System.out.println("none");
                            break;
                    }
                    for(Entry entry : entries){
                        if(entry.getKey() < 999){
                            System.out.print("0"+entry.getKey()+":"+entry.getText()+" ");
                        }
                        else{
                            System.out.print(entry.getKey()+":"+entry.getText()+" ");
                        }
                    }
                    System.out.println();
                    break;

                case "DisplayEventsFromStartDate":
                    int startKey = Integer.parseInt(inputs[1]);
                    System.out.print("DisplayEventsFromStartDate: ");
                    if(startKey > 1000){
                        System.out.print("0"+startKey+" ");
                    }
                    else{
                        System.out.print(startKey+" ");
                    }
                    ArrayList<Entry> startEntries = thisList.subMap(startKey, Integer.MAX_VALUE);
                    if(startEntries.size() == 1){
                            System.out.println("none");
                            break;
                    }
                    for(Entry entry : startEntries){
                        if(entry.getKey() == Integer.MIN_VALUE){
                            continue;
                        }
                        if(entry.getKey() < 999){
                            System.out.print("0"+entry.getKey()+":"+entry.getText()+" ");
                        }
                        else{
                            System.out.print(entry.getKey()+":"+entry.getText()+" ");
                        }
                    }
                    System.out.println();
                    break;

                case "DisplayEventsToEndDate":
                    System.out.print("DisplayEventsToEndDate ");
                    int endKey = Integer.parseInt(inputs[1]);
                    if(endKey > 1000){
                        System.out.print("0"+endKey+" ");
                    }
                    else{
                        System.out.print(endKey+" ");
                    }
                    ArrayList<Entry> endEntries = thisList.subMap(Integer.MIN_VALUE, endKey);
                    if(endEntries.size() == 1){
                            System.out.println("none");
                            break;
                    }
                    for(Entry entry : endEntries){
                        if(entry.getKey() == Integer.MIN_VALUE){
                            continue;
                        }
                        if(entry.getKey() < 999){
                            System.out.print("0"+entry.getKey()+":"+entry.getText()+" ");
                        }
                        else{
                            System.out.print(entry.getKey()+":"+entry.getText()+" ");
                        }
                    }
                    System.out.println();
                    break;

                case "DisplayAllEvents":
                    System.out.print("DisplayAllEvents: ");
                    thisList.printAll();
                    System.out.println();
                    break;

                case "PrintSkipList":
                    System.out.println("PrintSkipList");
                    thisList.print();
                    break;
                    
                default:
                    break;
            }
        }
    }

}
