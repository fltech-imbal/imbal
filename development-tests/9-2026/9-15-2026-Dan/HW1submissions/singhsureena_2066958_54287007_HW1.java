/*
author: Sureena Singh
email: sureena2025@my.fit.edu
course: CSE2010 section 3
description: Customer Service Chat Simulation that manages available representatives,
active chat sessions, and customer hold queues using singly linked lists. 

*/
import java.io.File;
import java.util.Scanner;
/*
Initialize global data structurea that help manage and track different helper methods.
reps- manages singly linked list of available representatives.
holds- tracks the customers waiting queue
sessions- records active chats and hold the customer name, representative and start time of the chat.
*/
public class HW1 {
     static AvailableRepList reps = new AvailableRepList();
     static HoldList holds = new HoldList();    
     static ChatSession sessions = new ChatSession();
    // available representatives list
    /* the list has the first in first out structure.
    There are helper methods such as printAvailableRepList(), isEmpty(), removeFirst() and addRep().
    The removeFirst() method was crucial for assigning the first available representative to the customer.*/
    private static class AvailableRepList {
        public static class RepNode {
            String name;
            RepNode next;

            RepNode(String name, RepNode next) {
                this.name = name;
                this.next = next;
            }
        }

        private RepNode head = null;
        private RepNode tail = null;

        public void printAvailableRepList(int printTime) {
            System.out.print("AvailableRepList " + String.format("%04d",printTime));
            RepNode current = head;
            while (current != null) {
                System.out.print(" " + current.name);
                current = current.next;
            }
            System.out.println();
        }

        public boolean isEmpty() {
            return head == null;
        }

        public String removeFirst() {
            if (head == null)
                return null;
            String name = head.name;
            head = head.next;
            if (head == null) {
                tail = null;
            }
            return name;
        }

        public void addRep(String name) {
            RepNode newNode = new RepNode(name, null);
            if (head == null) {
                head = newNode;
                tail = newNode;
            } else {
                tail.next = newNode;
                tail = newNode;
            }

        }
    }

    // Active Chat Sessions
    /* The ChatSession class stores two helper methods: ChatEnded() and addSession().
    The ChatEnded() method removes a chat from the active sessions list once it has ended.
    The addSession() method adds a chat session to the list once a representaive has been assigned to a customer.
    */
    public static class ChatSession {
        public static class ChatRecord {
            String customer;
            String rep;
            int startTime;

            ChatRecord(String c, String r, int t) {
                this.customer = c;
                this.rep = r;
                this.startTime = t;
            }
        }

        private static class SessionNode {
            ChatRecord element;
            SessionNode next;

            SessionNode(ChatRecord e, SessionNode n) {
                element = e;
                next = n;
            }
        }

        private SessionNode head = null;
        private SessionNode tail = null;

        public String ChatEnded(String cust, String repres, int endTime) {
            SessionNode current = head;
            SessionNode prev = null;
            while (current != null) {
                if (current.element.customer.equals(cust) && current.element.rep.equals(repres)) {
                    if (prev == null) {
                        head = current.next;
                    } else {
                        prev.next = current.next;
                    }
                    if (current == tail) {
                        tail = prev;
                    }
                    System.out.println("ChatEnded " + cust + " " + repres + " " +String.format("%04d",endTime));
                    return current.element.rep;
                }
                prev = current;
                current = current.next;
            }
            return null;
        }

        public void addSession(String name, String rep, int time) {
            ChatRecord data = new ChatRecord(name, rep, time);
            SessionNode newNode = new SessionNode(data, null);
            if (head == null) {
                head = newNode;
                tail = newNode;
            } else {
                tail.next = newNode;
                tail = newNode;
            }

        }
    }

    // Hold lists
    /*
    The HoldList class stores various helper methods such as updateMaxWait(),printMaxWaitTime(),quitOnHold() and addCustomer() method. 
    The Holdnode stores the customer name and the request time of the chat.
    printMaxWaitTime() uses the convertToMinutes() to convert the time from HHMM format to minutes to simplify the calculations.
    isEmpty() method is there to check if there are no customers on hold.
    updateMaxWait() updates the global variable that stores the max wait time.
    */

    public static class HoldList {
        private int maxWaitMinutesSoFar=0;
        public static class HoldData {
            public String cname;
            int requestTime;

            HoldData(String name, int time) {
                this.cname = name;
                this.requestTime = time;
            }
        }

        private static class HoldNode {
            HoldData element;
            HoldNode next;

            HoldNode(HoldData element, HoldNode next) {
                this.element = element;
                this.next = next;
            }
        }

        private HoldNode head = null;
        private HoldNode tail = null;
        public HoldData removeFirst(){
         if (head==null) return null;
         HoldData data = head.element; 
         head=head.next;
         if (head==null){
          tail =null;
         }
         return data; 
        }
        public void updateMaxWait(int waitMinutes){
         if (waitMinutes>maxWaitMinutesSoFar){
           maxWaitMinutesSoFar = waitMinutes;
         }
        }
        public void printMaxWaitTime(int printTime) {
            System.out.print("MaxWaitTime " + String.format("%04d",printTime));
            int diffHours = maxWaitMinutesSoFar/60;
                int diffMinsRemain =  maxWaitMinutesSoFar% 60;
                int waitFormatted = (diffHours * 100) + diffMinsRemain;
                System.out.print(" " + String.format("%04d", waitFormatted));
            System.out.println();
        }

        public static int convertToMinutes(int hhmm) {
            int hours = hhmm / 100;
            int minutes = hhmm % 100;
            return (hours * 60) + minutes;
        }

        public int quitOnHold(int quitTime, String customer) {
            HoldNode current = head;
            HoldNode prev = null;
            while (current != null) {
                if (current.element.cname.equals(customer)) {
                    if (prev == null) {
                        head = current.next;
                    } else {
                        prev.next = current.next;
                    }
                    if (current == tail) {
                        tail = prev;
                    }
                    int waitMinutes = convertToMinutes(quitTime)- convertToMinutes(current.element.requestTime);
                    updateMaxWait(waitMinutes);
                    System.out.println("QuitOnHold " + String.format("%04d", quitTime) + " " + customer);
                    return current.element.requestTime;
                }
                prev = current;
                current = current.next;
            }
            return -1;
        }

        public boolean isEmpty() {
            return head == null;
        }

        public void addCustomer(String name, int requestTime) {
            HoldData data = new HoldData(name, requestTime);
            HoldNode newNode = new HoldNode(data, null);
            if (head == null) {
                head = newNode;
                tail = newNode;
            } else {
                tail.next = newNode;
                tail = newNode;
            }

        }

    }
    /*chatRequest method uses nested conditional statements to function.
    If a representaive is available immediately then the representaive is assigned immediately.
    Otherwise depending on whether the customer chose to wait or try later,they are either added on the hold list or no function is performed.
    A try later statement is printed in case the later option is chosen*/
    public static void chatRequest(int time, String customer, String option) {
        System.out.println("ChatRequest " + String.format("%04d",time) + " " + customer + " " + option);
        if (!reps.isEmpty()) {
            String assignedRep = reps.removeFirst();
            System.out.println("RepAssignment " + customer + " " + assignedRep + " " + String.format("%04d",time));
            sessions.addSession(customer, assignedRep, time);
        } else {
            if (option.equals("wait")) {
                System.out.println("PutOnHold " + customer + " " + String.format("%04d",time));
                holds.addCustomer(customer, time);
            } else if (option.equals("later")) {
                System.out.println("TryLater " + customer + " " + String.format("%04d",time));
            }
        }
    }
    /*Accepts a valid command-line input argument file.
     Read the commands in the input file token-by-token.
     Add the elements in the available representative list.
     Use conditional statements to redirect the command and perform the required function by calling its respective method.
    */
    public static void main(String[] args) throws java.io.FileNotFoundException {
        if (args.length == 0) {
            System.out.println("Please provide a valid input file name.");
            return;
        }
        Scanner scanner = new Scanner(new File(args[0]));
   
        reps.addRep("Alice");
        reps.addRep("Bob");
        reps.addRep("Carol");
        reps.addRep("David");
        reps.addRep("Emily");
   
        while (scanner.hasNext()) {
            String command = scanner.next();
            if (command.equals("ChatRequest")) {
                int requestTime = scanner.nextInt();
                String customer = scanner.next();
                String waitOrLater = scanner.next();
                chatRequest(requestTime, customer, waitOrLater);
            } else if (command.equals("QuitOnHold")) {
                int quitOnHoldTime = scanner.nextInt();
                String customer = scanner.next();
                holds.quitOnHold(quitOnHoldTime, customer);
            } else if (command.equals("ChatEnded")) {
                String customer = scanner.next();
                String rep = scanner.next();
                int endTime = scanner.nextInt();
                String returnedRep = sessions.ChatEnded(customer,rep,endTime);
                if (returnedRep!=null){
                 if(!holds.isEmpty()){
                  HoldList.HoldData nextCustomer = holds.removeFirst();
                  if (nextCustomer!=null){ 
                   int waitMinutes = HoldList.convertToMinutes(endTime)- HoldList.convertToMinutes(nextCustomer.requestTime);
                   holds.updateMaxWait(waitMinutes);
                   System.out.println("RepAssignment "+nextCustomer.cname+" "+returnedRep+" "+String.format("%04d",endTime));
                   sessions.addSession(nextCustomer.cname, returnedRep,endTime);
                  }
                 
                 }else{
                 reps.addRep(returnedRep);
                 }
                }

            } else if (command.equals("PrintAvailableRepList")) {
                int printTime = scanner.nextInt();
                reps.printAvailableRepList(printTime);
            } else if (command.equals("PrintMaxWaitTime")) {
                int printTime = scanner.nextInt();
                holds.printMaxWaitTime(printTime);
            }
        }
        scanner.close();
    }
}

