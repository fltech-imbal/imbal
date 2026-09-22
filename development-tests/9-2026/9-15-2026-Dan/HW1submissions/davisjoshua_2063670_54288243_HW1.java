/**
Author: Joshua Davis
Email: joshuadavis2025@my.fit.edu
Course: CSE 2010 - Algorithms & Data Structures
Section: 4

Description:
This Program takes an input argument at Program run to simulate an online customer service chat.
It manages representatives, customers on hold, and active sessions with linked lists.
 */



import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.File;


public class HW1 {
    public static void main(String[] args) throws FileNotFoundException {
        StringNode availableReps;

        /*Here the required representatives and their names are created with their
         * request times and customer fields set to null */
        StringNode Alice = new StringNode("Alice", null, null);
        StringNode Bob = new StringNode("Bob", null, null);
        StringNode Carol = new StringNode("Carol", null, null);
        StringNode David = new StringNode("David", null, null);
        StringNode Emily = new StringNode("Emily", null, null);

        //Below I created the next pointers to create the singly linked list
        availableReps = Alice;
        Alice.next = Bob;
        Bob.next = Carol;
        Carol.next = David;
        David.next = Emily;

        /*Below I created the other linked lists required by the assignment
        * A Linked list head pointers where the customers will wait, the other for
        active chat sessions for representatives and customers*/
        StringNode Waiting = null;
        StringNode Active = null;

        /*Nobody is waiting at the start of the simulation
        * so we set the starting value to zero */
        int maxWaitingTime = 0;

        //I used a scanner variable to be able to take in the inputs from the input files
        File file = new File(args[0]);
        Scanner input = new Scanner(file);

        /*This while loop is the main loop of the simulation,
        * while there is more input, the simulation continues to run */
        while (input.hasNextLine()) {

            /* There are 4 values that we need to keep track of
            * The operation whether it is a ChatRequest, QuitOnHold, ChatEnded,
            * PrintAvailableRepList, or PrintMaxWaitTime. I gave the Variable Operation to this part of the input.
            * timeInt is the time which is originally in a string to be converted to int for the data manipulation.
            * the name is the name of the customer doing the operation, and the action for ChatRequest if they want
            * to try later or wait. */
            String Operation;
            int timeInt;
            String name;
            String action;

            /* For each iteration the input line becomes what that iteration is dealing with.
            * Then we read from the start of the string to the first space to find the operation,
            * then followed by a set of if statements to find the rest of the line correctly,
            * Based on the operation */
            String line = input.nextLine();
            int toSubstring = line.indexOf(" ");
            Operation = line.substring(0, toSubstring);

            /*The first Operation type is ChatRequest, the first Code block of each operation type reads the input line
            * and saves the information accordingly. Then prints the input. */
            if (Operation.equalsIgnoreCase("ChatRequest")) {

                //This block reads the input line
                int toSubstringTwo = line.indexOf(" ", toSubstring + 1);
                String time =  line.substring(toSubstring + 1, toSubstringTwo);
                timeInt = Integer.parseInt(time);
                Integer requestTime = timeInt;
                String formattedTime = String.format("%04d", timeInt);
                toSubstring = line.indexOf(" ", toSubstringTwo + 1);
                name = line.substring(toSubstringTwo + 1, toSubstring);
                toSubstringTwo = line.indexOf(" ", toSubstring + 1);
                action = line.substring(toSubstring + 1, line.length());


                System.out.println(Operation + " " + formattedTime + " " + name + " " + action);
                /*We check if there is an available representative, if there is one available they get assigned to the customer
                * if there is not one available it goes to the next if condition*/
                if (!isEmpty(availableReps)) {
                    StringNode rep = availableReps;
                    availableReps = availableReps.next;
                    rep.next = null;
                    rep.customer = name;
                    rep.requestTime = timeInt;
                    System.out.println("RepAssignment " + rep.customer + " " + rep.representative + " " + formattedTime);
                    Active = append(Active, rep);

                }
                //If the customer wanted to wait then they get put on hold
                else if (action.equalsIgnoreCase("wait")) {
                    System.out.println("PutOnHold " + name + " " + formattedTime);
                    StringNode customer = new StringNode(null, name, requestTime);
                    Waiting = append(Waiting, customer);
                }
                //If they get to here, there was not a representative available and they didn't want to wait so they are try later.
                else {
                    System.out.println("TryLater " + name + " " + formattedTime);
                }
            }
            //If the Operation is QuitOnHold they will make it to this if statement
            else if (Operation.equalsIgnoreCase("QuitOnHold")) {

                //Read through the line to get the required information for data manipulation
                int toSubstringTwo = line.indexOf(" ", toSubstring + 1);
                String time =  line.substring(toSubstring + 1, toSubstringTwo);
                timeInt = Integer.parseInt(time);
                String formattedTime = String.format("%04d", timeInt);
                toSubstring = line.indexOf(" ", toSubstringTwo + 1);
                name = line.substring(toSubstringTwo + 1, line.length());

                 /*This is the logic for iterating through the linked list to find the Node we need to remove
                * First we need to check what the max wait time is, if there is a new max we update it after that,
                * this removes the customer */
                StringNode current = Waiting;
                while (current != null) {
                    if (current.customer.equals(name)) {
                        if (timeInt - current.requestTime > maxWaitingTime) {
                            maxWaitingTime = timeInt - current.requestTime;
                        }
                        break;
                    }
                    current = current.next;
                }
                Waiting = removeCustomer(Waiting, name);
                System.out.println("QuitOnHold " + formattedTime + " " + name);
            }
            /*If the operation is a chat ended it will get caught by this block */
            else if (Operation.equalsIgnoreCase("ChatEnded")) {

                //This block reads the input line
                int toSubstringTwo = line.indexOf(" ", toSubstring + 1);
                name = line.substring(toSubstring + 1, toSubstringTwo);
                toSubstring  = line.indexOf(" ", toSubstringTwo + 1);
                String representative = line.substring(toSubstringTwo + 1, toSubstring);
                toSubstringTwo = line.indexOf(" ", toSubstring + 1);
                String time =  line.substring(toSubstring + 1, line.length());
                timeInt = Integer.parseInt(time);
                String formattedTime = String.format("%04d", timeInt);

                //This prints the input line
                System.out.println("ChatEnded " + name + " " + representative + " " + formattedTime);

                /*If there are no customers waiting the representative goes back to being available */
                if (isEmpty(Waiting)) {
                    StringNode current = Active;
                    while (current != null) {
                        if (current.customer.equals(name) && current.representative.equals(representative)) {
                            StringNode toMove = new StringNode(current.representative, null, null);
                            Active = removeCustomerAndRep(Active, current.customer, current.representative);
                            availableReps = append(availableReps, toMove);
                            break;
                        }
                        current = current.next;
                    }
                }
                //If there is a customer waiting, the representative that just finished a chat gets assigned
                else {
                    StringNode current = Active;
                    while (current != null) {
                        if (current.customer.equals(name) && current.representative.equals(representative)) {
                            if (timeInt - Waiting.requestTime > maxWaitingTime) {
                                maxWaitingTime = timeInt - Waiting.requestTime;
                            }
                            current.customer = Waiting.customer;
                            Waiting = Waiting.next;
                            break;
                        }
                        current = current.next;
                    }
                    System.out.println("RepAssignment " +  current.customer + " " + current.representative + " " + formattedTime);
                }
            }
            //If the operation is Print Available representatives, it makes it to this block
            else if (Operation.equalsIgnoreCase("PrintAvailableRepList")) {

                //this block reads the input line
                int toSubstringTwo = line.indexOf(" ", toSubstring + 1);
                String time =  line.substring(toSubstring + 1, line.length());
                timeInt = Integer.parseInt(time);
                String formattedTime = String.format("%04d", timeInt);

                //This prints the input
                System.out.print("AvailableRepList " + formattedTime);

                //Here we loop through the available reps, if there are no it prints nothing, if there is it prints until there are none left
                StringNode current = availableReps;
                if (current != null) {
                    System.out.print(" ");
                }
                while (current != null) {
                    System.out.print(current.representative);
                    if (current.next != null) {
                        System.out.print(" ");
                    }
                    current = current.next;
                }
                System.out.println();
            }
            //Here we print the input if it is PrintMaxWaitTime and then the max waiting time is printed
            else if (Operation.equalsIgnoreCase("PrintMaxWaitTime")) {
                int toSubstringTwo = line.indexOf(" ", toSubstring + 1);
                String time =  line.substring(toSubstring + 1, line.length());
                timeInt = Integer.parseInt(time);
                String formattedTime = String.format("%04d", timeInt);
                String formattedMaxWaitTime = String.format("%04d", maxWaitingTime);
                System.out.println("MaxWaitTime " + formattedTime + " " + formattedMaxWaitTime);
            }
            //This is an error catcher in case the operation does not match any of the above
            else {System.out.println("Unknown Operation " + Operation);}
        }
        input.close();

    }
    //This method adds a Node to the end of the given head of a Linked List that needs to be added to.
    public static StringNode append(StringNode head, StringNode toInsert) {
        StringNode current = head;
        if (head == null) {
            return toInsert;
        }
        if (head.next == null) {
            head.next = toInsert;
            return head;
        }
        while (current.next != null) {
            current = current.next;
        }
        current.next = toInsert;
        return head;
    }
    //This method removes a customer from a linked list that is with a representative
    public static StringNode removeCustomer(StringNode head, String name) {
        StringNode current = head;
        if (head == null) {
            return head;
        }
        if (head.customer.equals(name)) {
            head = head.next;
            return head;
        }
        while (current.next != null) {
            if (current.next.customer.equals(name)) {
                current.next = current.next.next;
                return head;
            } else {current = current.next;}
        }
        return head;
    }
    /*This method removes the Node of the given customer and representative, It sets the Customer field to null,
    * and moves the representative back to being available
    * the parameter head is the start of the linked list that you want to remove the customer and representative from
    * the name is the name of the customer, and repName is the name of the representative */
    public static StringNode removeCustomerAndRep(StringNode head, String name, String repName) {
        StringNode current = head;
        if (head == null) {
            return head;
        }
        if (head.customer.equals(name) && head.representative.equals(repName)) {
            head = head.next;
            return head;
        }
        while (current.next != null) {
            if (current.next.customer.equals(name) && current.next.representative.equals(repName)) {
                current.next = current.next.next;
                return head;
            } else {current = current.next;}
        }
        return head;
    }
    //This method checks if a certain linked list is empty given the head of the linked list
    public static boolean isEmpty(StringNode head) {
        if (head == null) {
            return true;
        }
        return false;
    }
}
//This class is what I used to keep track of all my data, with a Representative, customer, next, and RequestTime fields.
class StringNode {
    String representative;
    String customer;
    StringNode next;
    Integer requestTime;
    StringNode(String r, String c, Integer t){
        representative=r;
        customer = c;
        next = null;
        requestTime = t;
    }
}


