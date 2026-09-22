// Author: Adrian Santana
// Email: santanaa2025@my.fit.edu
// Course: CSE 2010
// Section: 1
// Description: The program presented below simulates an online retail service that allows customers to chat with a representative.
// Methods remove() and findCustomer() were added to the SinglyLinkedList class and called in this one.
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.io.File;

public class HW1 {
    public static void main(String[] args) throws FileNotFoundException {

        // Create a list for available representatives and add the representatives' names
        SinglyLinkedList<String> availableRepresentatives = new SinglyLinkedList<>();
        availableRepresentatives.addLast("Alice");
        availableRepresentatives.addLast("Bob");
        availableRepresentatives.addLast("Carol");
        availableRepresentatives.addLast("David");
        availableRepresentatives.addLast("Emily");

        // Create an empty list for customers on hold and chat sessions
        SinglyLinkedList<String> customersOnHold = new SinglyLinkedList<>();
        SinglyLinkedList<String> chatSessions = new SinglyLinkedList<>();

        // Make a scanner object that takes in the input file
        File inputFile = new File(args[0]);
        Scanner scanner = new Scanner(inputFile);

        // Declare a variable that keeps track of the maximum wait time throughout the code
        int maxWaitTime = 0;

        // Create a while loop that continues the code until the scanner has no more lines to read
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();

            // Use the line split tool to break up the input and use each part individually
            String[] parts = line.split(" ");

            // If a customer enters "ChatRequest" and a representative is available, remove it from the list and place it in the chatSessions list with the customer
            if (parts[0].equals("ChatRequest")) {
                System.out.println("ChatRequest " + parts[1] + " " + parts[2] + " " + parts[3]);
                if (!availableRepresentatives.isEmpty()) {
                    String representative = availableRepresentatives.removeFirst();
                    System.out.println("RepAssignment " + parts[2] + " " + representative + " " + parts[1]);
                    chatSessions.addLast(parts[2] + " " + representative);

                    // If the customer wishes to wait, put it in the customersOnHold list
                } else if (parts[3].equals("wait")) {
                    customersOnHold.addLast(parts[2] + " " + parts[1]);
                    System.out.println("PutOnHold " + parts[2] + " " + parts[1]);

                    // If the customer wishes to try later, print the "TryLater" message
                } else if (parts[3].equals("later")) {
                    System.out.println("TryLater " + parts[2] + " " + parts[1]);
                }

                // If the chat has ended, remove the chat from the chatSessions list
            } else if (parts[0].equals("ChatEnded")) {
                System.out.println("ChatEnded " + parts[1] + " " + parts[2] + " " + parts[3]);
                chatSessions.remove(parts[1] + " " + parts[2]);

                // If there are no customers on hold, add the representative back to availableRepresentatives
                if (customersOnHold.isEmpty()) {
                    availableRepresentatives.addLast(parts[2]);

                    // If there is a customer on hold, assign the representative to the next customer in line
                } else {
                    String nextCustomer = customersOnHold.removeFirst();
                    String[] nameAndTime = nextCustomer.split(" ");
                    System.out.println("RepAssignment " + nameAndTime[0] + " " + parts[2] + " " + parts[3]);
                    chatSessions.addLast(nameAndTime[0] + " " + parts[2]);
                    // Calculate the wait time of the customer

                    int requestTime = Integer.parseInt(nameAndTime[1]);
                    int requestMinutes = ((requestTime / 100) * 60) + (requestTime % 100);
                    int assignmentTime = Integer.parseInt(parts[3]);
                    int assignmentMinutes = ((assignmentTime / 100) * 60) + (assignmentTime % 100);
                    int waitTime = assignmentMinutes - requestMinutes;

                    // If the customer's wait time is greater than the current max wait time, update max wait time
                    if (waitTime > maxWaitTime) {
                        maxWaitTime = waitTime;
                        System.out.println("MaxWaitTime " + parts[3] + " " + maxWaitTime);
                    }
                }

                // If the customer quits while on hold, search for that customer and remove it from customersOnHold
            } else if (parts[0].equals("QuitOnHold")) {
                System.out.println("QuitOnHold " + parts[1] + " " + parts[2]);
                String customer = customersOnHold.findCustomer(parts[2]);
                customersOnHold.remove(customer);
                String[] nameAndTimeQuit = customer.split(" ");

                // Calculate the wait time of the customer who quit while on hold
                int requestTimeQuit = Integer.parseInt(nameAndTimeQuit[1]);
                int requestMinutesQuit = ((requestTimeQuit / 100) * 60) + (requestTimeQuit % 100);
                int quitTime = Integer.parseInt(parts[1]);
                int quitTimeMinutes = ((quitTime / 100) * 60) + (quitTime % 100);
                int quitWaitTime = quitTimeMinutes - requestMinutesQuit;

                // If the customer who quit had a greater wait than the current max wait time, update max wait time
                if (quitWaitTime > maxWaitTime) {
                    maxWaitTime = quitWaitTime;
                    System.out.println("MaxWaitTime " + parts[1] + " " + maxWaitTime);
                }

                // If the customer enters "PrintAvailableRepList", print the representatives in the list
            } else if (parts[0].equals("PrintAvailableRepList")) {
                System.out.println("AvailableRepList " + parts[1] + " " + availableRepresentatives);

                // If the customer enters "PrintMaxWaitTime", print max wait time
            } else if (parts[0].equals("PrintMaxWaitTime")) {
                System.out.println("MaxWaitTime " + parts[1] + " " + maxWaitTime);
            }
        }
    }
}

