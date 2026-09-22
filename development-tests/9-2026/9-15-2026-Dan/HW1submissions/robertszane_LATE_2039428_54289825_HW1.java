/*
Author: Zane Roberts
Email: zroberts2025@fit.edu
Course: CSE 2010
Section: 3
Description of this file: It is a program that assigns repersentatives to customers and logs when their chat starts, when
it is put on wait, and when it ends. It also documents the time at which each customer was assigned and when thier chat ended.
*/

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW1 {

    /**
     * Represents a single node in a singly linked list.
     * Can hold customer names, representative names, and request timestamps.
     */
    static class Node {
        // Name of customer or available representative
        String data1; 
        // Name of assigned representative (used primarily in active sessions)
        String data2; 
        // Timestamp in HHMM integer format for tracking customer request time
        int time;     
        // Pointer reference to the next node in the list
        Node next;

        /**
         * Constructor to create a new Node with given data values.
         * customer or representative name
         * assigned representative name
         * Timestamp value associated with the event
         */
        Node(String data1, String data2, int time) {
            this.data1 = data1;
            this.data2 = data2;
            this.time = time;
            this.next = null;
        }
    }

    /**
     * singly linked list implementation for managing queues.
     * Handles operations such as append, remove front, search/remove, and display.
     */
    static class SimpleLinkedList {
        // Reference to the head (first node) of the list
        Node head = null;
        // Reference to the tail (last node) of the list
        Node tail = null;

        /**
         * Checks if the linked list currently contains any elements.
         * return true if the list is empty, false otherwise
         */
        boolean isEmpty() {
            return head == null;
        }

        /**
         * Adds a new node to the end of the singly linked list.
         * Updates tail pointer and handles initial insertion when head is null.
         * param data1 is customer or representative name
         * param data2 is assigned representative name
         * param time is the Timestamp associated with the element
         */
        void addLast(String data1, String data2, int time) {
            Node newNode = new Node(data1, data2, time);
            if (isEmpty()) {
                head = newNode;
                tail = newNode;
            } else {
                tail.next = newNode;
                tail = newNode;
            }
        }

        /**
         * Removes and returns the node at the front of the list.
         * Updates head pointer and resets tail pointer if the list becomes empty.
         * return The removed Node, or null if the list is empty
         */
        Node removeFirst() {
            if (isEmpty()) return null;
            Node temp = head;
            head = head.next;
            if (head == null) {
                tail = null;
            }
            return temp;
        }

        /**
         * Searches for and removes a node by matching data1 against a target name.
         * looks through pointers to un-link the matching node from the list.
         * param name finds Target string to find and remove customer name
         * return The removed Node if found, or null if not there
         */
        Node removeByName(String name) {
            if (isEmpty()) return null;

            // Check if head node contains target name
            if (head.data1.equals(name)) {
                Node removed = head;
                head = head.next;
                if (head == null) tail = null;
                return removed;
            }

            // Iterate through list to locate matching target node
            Node current = head;
            while (current.next != null && !current.next.data1.equals(name)) {
                current = current.next;
            }

            // Unlink target node from previous node if found
            if (current.next != null) {
                Node removed = current.next;
                current.next = current.next.next;
                if (current.next == null) tail = current;
                return removed;
            }

            return null;
        }

        /**
         * finds available representatives list and prints contents in order.
         * Formats time output with 4 digits and with leading zeros.
         * param printTime saves Current timestamp for the command execution
         */
        void printReps(int printTime) {
            System.out.print("AvailableRepList " + String.format("%04d", printTime));
            Node current = head;
            while (current != null) {
                System.out.print(" " + current.data1);
                current = current.next;
            }
            System.out.println();
        }
    }

    /**
     * Converts HHMM formatted integer time into total elapsed minutes from midnight.
     * Extracts hours via integer division and minutes via modulus operation.
     * param time Integer time value in HHMM format
     * returns Total minutes past 00:00 midnight
     */
    private static int getMinutes(int time) {
        int hours = time / 100;
        int mins = time % 100;
        return hours * 60 + mins;
    }

    /**
     * Computes difference in total minutes between start and end HHMM times.
     * Uses getMinutes helper to perform subtraction in standard total minutes.
     * param startTime is the Beginning timestamp in HHMM format
     * param endTime is the Ending timestamp in HHMM format
     * returns Calculated elapsed wait time in total minutes
     */
    private static int calculateWaitTime(int startTime, int endTime) {
        return getMinutes(endTime) - getMinutes(startTime);
    }

    /**
     * Converts a total duration in minutes into a 4-digit HHMM formatted String.
     * Formats hours and minutes with leading zeros to maintain 4-digit requirement.
     * param totalMinutes Total elapsed duration in minutes
     * returns Formatted string representation in HHMM format
     */
    private static String formatWaitTime(int totalMinutes) {
        int hours = totalMinutes / 60;
        int mins = totalMinutes % 60;
        return String.format("%02d%02d", hours, mins);
    }

    /**
     * Main program method. Handles command line argument parsing, input file reading,
     * queue initializations, command dispatching, and tracking maximum wait times.
     * param args Command line arguments containing target input file name
     */
    public static void main(String[] args) {
        /*
         * Command Line Validation Block:
         * Verifies that an input filename argument was supplied to the program.
         * Displays error usage text and terminates main execution if missing.
         */
        if (args.length < 1) {
            System.err.println("Usage: java HW1 <input_file>");
            return;
        }

        /*
         * System State Initialization Block:
         * Instantiates singly linked lists for available representatives,
         * waiting customer hold queue, and active customer-rep chat sessions.
         * Populates default available representatives in designated order
         */
        SimpleLinkedList availableReps = new SimpleLinkedList();
        SimpleLinkedList holdQueue = new SimpleLinkedList();
        SimpleLinkedList activeSessions = new SimpleLinkedList();

        availableReps.addLast("Alice", null, 0);
        availableReps.addLast("Bob", null, 0);
        availableReps.addLast("Carol", null, 0);
        availableReps.addLast("David", null, 0);
        availableReps.addLast("Emily", null, 0);

        // Tracks maximum wait time encountered across all customer interactions
        int maxWaitTimeSoFar = 0;

        /*
         * File Processing & Main Execution Loop Block:
         * Opens input file specified in command-line arguments using Scanner.
         * Line by line, parses whitespace-delimited event request arguments and
         * sends execution to corresponding event handlers inside switch block.
         */
        try (Scanner scanner = new Scanner(new File(args[0]))) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");
                String eventType = parts[0];

                switch (eventType) {
                    /*
                     * ChatRequest Event Block:
                     * Parses timestamp, customer name, and hold decision preference.
                     * Assigns available representative immediately if present;
                     * otherwise, places customer on hold queue or outputs TryLater.
                     */
                    case "ChatRequest": {
                        int requestTime = Integer.parseInt(parts[1]);
                        String customer = parts[2];
                        String waitOrLater = parts[3];

                        System.out.println("ChatRequest " + String.format("%04d", requestTime) + " " + customer + " " + waitOrLater);

                        if (!availableReps.isEmpty()) {
                            Node repNode = availableReps.removeFirst();
                            String rep = repNode.data1;
                            
                            activeSessions.addLast(customer, rep, 0);
                            System.out.println("RepAssignment " + customer + " " + rep + " " + String.format("%04d", requestTime));
                        } else {
                            if (waitOrLater.equalsIgnoreCase("wait")) {
                                holdQueue.addLast(customer, null, requestTime);
                                System.out.println("PutOnHold " + customer + " " + String.format("%04d", requestTime));
                            } else {
                                System.out.println("TryLater " + customer + " " + String.format("%04d", requestTime));
                            }
                        }
                        break;
                    }

                    /*
                     * ChatEnded Event Block:
                     * Removes customer-rep session from active list upon chat completion.
                     * Reassigns available rep to first customer in hold queue if present,
                     * calculating customer wait duration and updating maximum wait time tracker.
                     */
                    case "ChatEnded": {
                        String customer = parts[1];
                        String rep = parts[2];
                        int endTime = Integer.parseInt(parts[3]);

                        System.out.println("ChatEnded " + customer + " " + rep + " " + String.format("%04d", endTime));
                        activeSessions.removeByName(customer);

                        if (!holdQueue.isEmpty()) {
                            Node nextCustomer = holdQueue.removeFirst();
                            activeSessions.addLast(nextCustomer.data1, rep, 0);
                            System.out.println("RepAssignment " + nextCustomer.data1 + " " + rep + " " + String.format("%04d", endTime));

                            int waitTime = calculateWaitTime(nextCustomer.time, endTime);
                            if (waitTime > maxWaitTimeSoFar) {
                                maxWaitTimeSoFar = waitTime;
                            }
                        } else {
                            availableReps.addLast(rep, null, 0);
                        }
                        break;
                    }

                    /*
                     * QuitOnHold Event Block:
                     * Handles customer leaving hold queue before rep assignment.
                     * Searches hold queue, removes customer, calculates total time spent on hold,
                     * and updates running maximum wait time tracker accordingly.
                     */
                    case "QuitOnHold": {
                        int quitTime = Integer.parseInt(parts[1]);
                        String customer = parts[2];

                        Node removed = holdQueue.removeByName(customer);
                        if (removed != null) {
                            System.out.println("QuitOnHold " + String.format("%04d", quitTime) + " " + customer);
                            int waitTime = calculateWaitTime(removed.time, quitTime);
                            if (waitTime > maxWaitTimeSoFar) {
                                maxWaitTimeSoFar = waitTime;
                            }
                        }
                        break;
                    }

                    /*
                     * Print Commands Handling Block:
                     * Prints currently available representatives list in sequence, or
                     * formats and displays highest wait time recorded so far in HHMM.
                     */
                    case "PrintAvailableRepList": {
                        int printTime = Integer.parseInt(parts[1]);
                        availableReps.printReps(printTime);
                        break;
                    }

                    case "PrintMaxWaitTime": {
                        int printTime = Integer.parseInt(parts[1]);
                        System.out.println("MaxWaitTime " + String.format("%04d", printTime) + " " + formatWaitTime(maxWaitTimeSoFar));
                        break;
                    }
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("Error: File not found - " + args[0]);
        }
    }
}
