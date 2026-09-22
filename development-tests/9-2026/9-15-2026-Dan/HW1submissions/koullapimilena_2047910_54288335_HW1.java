/*
  Author: Milena Koullapi
  Email: mkoullapi2025@fit.edu
  Course: CSE 2010
  Section: 01
  Description of this file: A simulation that represents an online chat service.
  Customers can request chats, get assigned to representatives, put on hols or try later. Also the maximum wait time is tracked.
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;

public class HW1
{

   /*
   The main method reads a filename from the command line, opens that file, and prints each line it contains to the screen.
   @param args - command-line arguments; args[0] is expected to be the path to the input file to read.
   */ 
    public static void main(String[] args)
    {
  
      /* 
      Filename store the name or path of the file that is going to be read.
      Thats information is received from args[0], which represents the first command-line argument 
      */

      String filename = args[0];
      
      
      /*
      It creates a File object using the filename variable and opens a Scanner to read what's in the file
      I used try(...) instead of try{} block because it automatically closes the Scanner
      when the block finishes executing, even if there was an error or exception and i dont have to manually close the scanner even
      */
      ArrayList<Request> requests = new ArrayList<>();

      SinglyLinkedList<String> availableRep = new SinglyLinkedList<>();
      availableRep.addLast("Alice");
      availableRep.addLast("Bob");
      availableRep.addLast("Carol");
      availableRep.addLast("David");
      availableRep.addLast("Emily");

      SinglyLinkedList<OnHoldCustomer> onHold = new SinglyLinkedList<>();
      SinglyLinkedList<ActiveChat> activeChat = new SinglyLinkedList<>();

      try ( Scanner scanner = new Scanner(new File(filename))){
       // Checks if there is another line of text left to read in the file. The loop will continue running as long as there is more lines.
        while(scanner.hasNextLine()){
       // Reads the next line of textand stores it in the variable called line.
          String line = scanner.nextLine();
       // Prints the text stored in the variable line and creates a new line.
          
          requests.add(takeRequest(line));
        }
        
        Collections.sort(requests);

        int maxWaitTime = 0;

        // Walk through every request in order (time first, then if its the same by the priority)
        //  simulate the appropriate behavior for each of the 5 event types
        for(Request req : requests) {

          // A customer requests a chat, a rep is assigned if available, otherwise
          // either put them on hold ("wait") or tell them to try later ("later")
          if (req.type.equals("ChatRequest")){

            System.out.println("ChatRequest " + formatTime(req.time) + " " + req.customer + " " + req.waitOrLater);
            
            if(!availableRep.isEmpty()){
              String rep = availableRep.removeFirst();
              System.out.println("RepAssignment " + req.customer + " " + rep + " " + formatTime(req.time));
            
            ActiveChat active = new ActiveChat();
            active.customer = req.customer;
            active.rep = rep;
            active.requestTime = req.time;
            activeChat.addLast(active);
            
            }
            else if(req.waitOrLater.equals("wait")){
              OnHoldCustomer hold = new OnHoldCustomer();
              hold.name = req.customer;
              hold.requestTime = req.time;
              onHold.addLast(hold);

              System.out.println("PutOnHold " + req.customer + " " + formatTime(req.time));
            }
            else{
              System.out.println("TryLater " + req.customer + " " + formatTime(req.time));
            }
          }
          
          //The chat between the customer and the rep ended. So the rep is assign to a new person on hold 
          // or they go back to the available list and wait for a customer
          else if (req.type.equals("ChatEnded")){

            System.out.println("ChatEnded " + req.customer + " " + req.rep + " " + formatTime(req.time));
          
            SinglyLinkedList<ActiveChat> temp = new SinglyLinkedList<>();
            
            while(!activeChat.isEmpty()){
              ActiveChat customer = activeChat.removeFirst();
              if((!customer.customer.equals(req.customer)) && (!customer.rep.equals(req.rep)) ){
                temp.addLast(customer);
              }
            }
            activeChat = temp;  

            if(!onHold.isEmpty()){
              OnHoldCustomer waiting = onHold.removeFirst();
              System.out.println("RepAssignment " + waiting.name + " " + req.rep + " " + formatTime(req.time));
              
              int waitTime = req.time - waiting.requestTime;
              if (waitTime > maxWaitTime) {
                maxWaitTime = waitTime;
              }

            ActiveChat active = new ActiveChat();
            active.customer = waiting.name;
            active.rep = req.rep;
            active.requestTime = waiting.requestTime;
            activeChat.addLast(active);

            }
            else{
              availableRep.addLast(req.rep);
            }
          }

          //A customer desites they want to no longer wait on hold, they are removed from the hold list.
          else if (req.type.equals("QuitOnHold")){
            System.out.println("QuitOnHold " + formatTime(req.time) + " " + req.customer);

            SinglyLinkedList<OnHoldCustomer> tempHold = new SinglyLinkedList<>();

            while(!onHold.isEmpty()){
              OnHoldCustomer customer = onHold.removeFirst();
              if(!customer.name.equals(req.customer)){
                tempHold.addLast(customer);
              }
              else{
                int waitTime = req.time - customer.requestTime;
               
                if (waitTime > maxWaitTime) {
                  maxWaitTime = waitTime;
                }
              }
            }
            onHold = tempHold;

          }

          //It prints all the available representatives at a specific time.
          else if (req.type.equals("PrintAvailableRepList")){
            SinglyLinkedList<String> temp = new SinglyLinkedList<>();
            
            System.out.print("AvailableRepList " + formatTime(req.time));

            while(!availableRep.isEmpty()){
              String name = availableRep.removeFirst();
              System.out.print(" " + name);
              temp.addLast(name);
            }

            availableRep = temp;

            System.out.println();
          }

          //it prints the longest wait time of a customer. It calculates their request time with the time they have been helped.
          //This is being tracked asnd updated in QuitOnHold and ChatEnded
          else if (req.type.equals("PrintMaxWaitTime")){
            
            System.out.println("MaxWaitTime " + formatTime(req.time) + " " + formatTime(maxWaitTime));

          }
        }
      
      }
      //This block only runs if the attempt to open the file specified by filename doesnt exists, cannot be opens or found in the specific path.
       catch (FileNotFoundException e) { 
        //It print "File not found" to the user
        System.out.println("File not found.");
        //it print the detailed technical details of the errors to help the developers understand where and why the error happened
        e.printStackTrace();
      }
    }


    /*
    The perpose of this class is to represent the different events (requests) from the input file.
    Each request has a different format. This class stores all the possible fields, but not every field is used by each request.
    */
    static class Request implements Comparable<Request> {

      String type; //Used by all Events - tells you which of the 5 kind of request it is.
      int time; //Used by all Event - tells you the time of when an action ended or started.
      String customer; // Used by ChatRequest, ChatEnded, QuitOnHold - It represents the name of the customer
      String rep; //Used only by ChatEnded - represent the name of the representative that the customer was talking to
      String waitOrLater; // Used by ChatRequest  - represents the choice of if they want to wait and be put on hold or end the call
    
      /*
      It compares two Requests so that they can be sorted in the correct order
      First it compares the time (the earlier event is first), if the time is the same it compares them by their type and its priority (using getPriority)
      */
      public int compareTo(Request other){

        if(this.time != other.time){
          return this.time - other.time; 
        }
        else{
          return getPriority(this.type) - getPriority(other.type);
        }
      }

    }

     /*
      takeRequest: parses one raw line of text from the input file into a Request object,
      then sorts each of the fields to its corresponding event types the line represents.
      @param line - one line of text read from the input file
      @return a Request object populated with that line's data
     */
    static Request takeRequest(String line){
      String[] place = line.split(" "); //Seperates the line into seperate words using " ". It is usefull because each word is a different type which can be accesed individually by index.
      Request req = new Request(); // Creates a new empty Request object 
      req.type = place[0]; //The first place (0) is always the type of Request/Event and it stored as that.

      //If the type of the Event is ChatRequest the following steps take place. If not it moves on to the next if.
      if(req.type.equals("ChatRequest")){
        req.time = Integer.parseInt(place[1]);
        req.customer = place[2];
        req.waitOrLater = place[3];
        return req;
      }
      //If the type of the Event is ChatEnded the following steps take place. If not it moves on to the next if.
      else if(req.type.equals("ChatEnded")){
        req.customer = place[1];
        req.rep = place[2];
        req.time = Integer.parseInt(place[3]);
        return req;
      }
      //If the type of the Event is QuitOnHold the following steps take place. If not it moves on to the next if.
      else if(req.type.equals("QuitOnHold")){
        req.time = Integer.parseInt(place[1]);
        req.customer = place[2];
        return req;
      }
      //If the type of the Event is PrintAvailableRepList the following steps take place. If not it moves on to the next if.
      else if(req.type.equals("PrintAvailableRepList")){
        req.time = Integer.parseInt(place[1]);
        return req;
      }
      //If the type of the Event is PrintMaxWaitTime the following steps take place. If not it moves on to the next if.
      else if(req.type.equals("PrintMaxWaitTime")){
        req.time = Integer.parseInt(place[1]);
        return req;
      }
      //If the Event is not any of the 5 that are expected this error message is printed out.
      else{
        System.out.println("Request doesnt exist. Please try again.");
      }
      //
      return req;


    }

    /*
    It is used to give priority to events in case they share the same request time. 
    So each event type has a corisponding number (lower numbers are printed first), which is then returned.
    It is used by copareTo method to compare the request time.
    @param type - the event type string (e.g. "ChatEnded", "ChatRequest")
    @return an int priority ranking, or -1 if the type is unrecognized
    */
   static int getPriority(String type){
    if(type.equals("ChatEnded")){
      return 0; 
    }
    else if(type.equals("ChatRequest")){
      return 1; 
    }
    else if(type.equals("QuitOnHold")){
      return 2; 
    }
    else if(type.equals("PrintAvailableRepList")){
      return 3; 
    }
    else if(type.equals("PrintMaxWaitTime")){
      return 4; 
    }
    else{
      return -1;
    }

   }

   /*
   Represents a customer that is on hold and is waiting for a representative.
   It stores their name and their original request time, so the wait time can be calculated later
   */ 
   static class OnHoldCustomer{
    String name; //the customer's name
    int requestTime; // customer's request time

   }

   /*
   Represents a ongoing chat session between a customer and a representative.
   It stores the customer's name, the rep's name, and the customer's original request time, so it can be used to calculate the wait time once the chat ends.
   */ 
   static class ActiveChat{
    String customer; // the customer's name 
    String rep; // the representative in the chat
    int requestTime; // customer's request time
   }

   /*
   It makes sure that the time value is printed  into HHMM output format.
   @param time - the time value to format
   @return the time formatted as a 4-digit String
   */ 
   static String formatTime(int time){
    return String.format("%04d", time);
   }

}
