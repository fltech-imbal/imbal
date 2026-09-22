/*

  Author: Avery Haughwout
  Email: ahaughwout2024@my.fit.edu  
  Course: CSE2012
  Section: 01
  Description of this file: First Homework; 


*/
import java.util.Scanner;

public class HW1
{

    /*
      Description of each method, including parameters 
    */
    static SinglyLinkedList<String> availableRep = new SinglyLinkedList<String>();//Holds STRINGS of available representatives 
    static SinglyLinkedList<String[]> waitList = new SinglyLinkedList<String[]>(); // holds Request array (Time, Name)
    static SinglyLinkedList<String[]> chatList = new SinglyLinkedList<String[]>(); // holds chat array (Time, Name, Rep)
    public static Scanner scnr = new Scanner(System.in); // our scanner
    
    public static void main(String[] args){ 
        //Populating availableRep w. initial availability
        availableRep.addLast("Alice");
        availableRep.addLast("Bob");
        availableRep.addLast("Carol");
        availableRep.addLast("David");
        availableRep.addLast("Emily");//Declaring all used lists
        
        while(true){
            String input = scnr.nextLine();
            String[] inputChop = input.split(" ");//Reads and Divides Input
            switch(inputChop[0]){
                case "ChatRequest":
                    ChatRequest(inputChop[1], inputChop[2], inputChop[3]);
                    break;
                case "QuitOnHold":
                    QuitOnHold(inputChop[1], inputChop[2]);
                    break;
                case "ChatEnded":
                    ChatEnded(inputChop[3], inputChop[1], inputChop[2]); //pure evil
                    break;
                case "AvailableRepList":
                    AvailableRepList();
                    break;
                case "PrintMaxWaitTime":
                default:
                    break;
            }
            
        }
    }
    public static void ChatRequest(String Time, String Name, String waitOrLater){//WORKING
        /*
           ChatRequest
           Input: Time, Name, waitOrLater
           Output:
               - Constructs a string array "Request" which contains Name and Time of request
               - Assuming a representative is available, pulls the 1st available rep and makes a new string array for chatList which appends
               the representative to the prior request array
               - If no reps are available, either cancels the request OR puts the request method onto the waitList, depending on the user input 
               waitOrLater
           */
        String[] newReq = new String[2];
        newReq[0] = Time;
        newReq[1] = Name;
        if(availableRep.isEmpty() == false){
            String newRep = availableRep.removeFirst(); // gets the new representative
            String[] newChat = new String[3];
            newChat[0] = newReq[0]; 
            newChat[1] = newReq[1];
            newChat[2] = newRep;
            chatList.addLast(newChat);
            System.out.println("RepAssignment " + Name +" "+ newRep +" "+ Time);
        }
        else{ // what happens if no reps are available :(
            if(waitOrLater.equals("wait")){//checking if the person wants to wait
                waitList.addLast(newReq);
                System.out.println("PutOnHold "+Name+" "+Time);
            }
            else{

                System.out.println("TryLater "+Name+" "+Time);
            }
        }
        
    }
    public static void QuitOnHold(String Time, String Name){
        /*
           QuitOnHold
               Objective: Find the node of waitList which contains the name of the request (as declared in input)
               and removes it using the personally implemented findAndKeep method
               NOTE: This method and it's sister method chatEnded are both severely bugged due to my inability to work with the general <E>
               SinglyLinkedList method proficiently. 
           */
        String[] checkQuit = new String[2];
        checkQuit[0] = Time; checkQuit[1] = Name;
        waitList.findAndKeep(checkQuit); //I couldn't care less what happens to this String i just want the method to run
    }
    
    public static void ChatEnded(String Time, String Name, String Rep){
        String[] endChat = new String[3];
        endChat[0] = Time; endChat[1] = Name; endChat[2] = Rep; 
        availableRep.addLast(chatList.findAndKeep(endChat)[2]);
        if(waitList.isEmpty() == false && availableRep.isEmpty() == false){
            String[] nextServed = waitList.removeFirst();
            ChatRequest(nextServed[0], Time, "wait"); //I can only assume if they're on the wait list they want to continue waiting
        }
    }
    public static void AvailableRepList(){//WORKING
        //Prints all elements in availableRep with personally written method prettyPrint in SinglyLinkedList
        System.out.println(availableRep.prettyPrint());
    }
    public static void PrintMaxWaitTime(String Time){//maybe working who knows
        /*
           PrintMaxWaitTime
           NOTE: Ideally, this method would search through all elements of waitList and find the one with the highest Integer.parseInt(waitList(1)) value
           w the following implementation:
               Node checkNode = head;
               int returnNode = integer.parseInt(checkNode.getElement()[1]); //default return 
               while(checkNode != null){
                checkNext = integer.parseInt(returnNode.getElement()[1])
                if(checkNext > checkNode.getElement);
                    returnNode = checkNode.next;
                }
                checkNode = checkNode.next;
               }
                */
        System.out.println(waitList.last()[1]);
    }
    /* 
     
     SCRAP CODE
     //Display all options to user
            /* System.out.println("---------------------------------------------");
            System.out.println("Options:");
            System.out.println("* ChatRequest: Request a chat with a representative.");// Time, Name, waitOrLater
            System.out.println("* QuitOnHold: Quit being on hold if holding."); // Time, Name 
            System.out.println("* ChatEnded: Declare that your chat has ended.");// Name, Rep, Time
            System.out.println("* AvailableRepList: See all available representatives.");// time
            System.out.println("* PrintMaxWaitTime: See the longest wait time"); // Time
            System.out.print("Please input request: "); */
}
