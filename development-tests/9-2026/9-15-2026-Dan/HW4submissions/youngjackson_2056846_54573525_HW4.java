/*
 * Author: Jackson Young
 * Email: young2025@my.fit.edu
 * Course: CSE2010
 * Section: 1
 * Driver program for processing input files and printing output
*/
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;
public class HW4 {
    // main method runs when the program is executed from the command line
    // It takes two command-line arguments:
    //    1. A .txt file of names and ratings used to construct the priority queue heap
    //    2. Another .txt file that list the actions to be performed on the above heap
    public static void main(String[] args) {
        // First implement two Scanner objects, one for each input file,
        // using a try-catch block to terminate the program if an error occurs during file reading
        Scanner ratingScanner, actionScanner; // Declare scanner variables
        try {
            // Attempt to construct File and Scanner objects using input
            ratingScanner = new Scanner(new File(args[0]));
            actionScanner = new Scanner(new File(args[1]));
        }
        // If the file cannot be read, safely terminate the program
        catch (FileNotFoundException e) {
            System.out.println("ERROR: FileNotFoundException.\n" + e.getMessage() +"\nTerminating Program.");
            return; // End program
        }

        // Next, use Scanner to loop through the first input and construct the heap
        String targetCustomer = ratingScanner.next(); // Keep track of target customer whom we give recommendations to
        Customer targetCustomerObj = null; // Prepare a variable to hold the object version of the target customer
        ArrayList<Customer> customerList = new ArrayList(); // Create an array to hold the alphabetized list of customers
        Heap<Customer> heap = new Heap(); // Create an empty heap of type Customer to compare distance values

        while (ratingScanner.hasNext()) { // Check if there is something else to read in input file 1
            // First word is name
            String customerName = ratingScanner.next();
            // Read the next 10 inputs as the customer's ratings for the songs
            // and put them in a int array of size 10
            int[] ratingList = new int[10];
            for (int i = 0; i < 10; i++) {
                ratingList[i] = ratingScanner.nextInt();
            }

            // Construct a Customer object to store the above information
            Customer customer = new Customer(customerName, ratingList);
            // Add to list
            customerList.add(customer);
            // If the customer is the target customer, assign it to targetCustomerObj
            if (customerName.equals(targetCustomer)) targetCustomerObj = customer;
        }

        // After the customer list is complete, sort the list alphabetically using customers' names
        // This method uses the Comparator object to compare the customers' name fields only
        customerList.sort(Comparator.comparing(customer -> customer.name));

        // Once the array is complete, add each customer to the heap
        for (Customer customer : customerList) {
            // Check if the distance is negative and the customer isn't the target
            double distance = customer.findDistance(targetCustomerObj);
            if (distance > 0 && !customer.equals(targetCustomerObj)) {
                heap.add(customer, distance);
            }
        }

        // Next, use the second Scanner object to read the second input file
        // and perform the actions specified by the text
        while (actionScanner.hasNext()) { // Check if there is something else to read in input file 2
            // First word is the action type
            String action = actionScanner.next();
            
            // Check the name of the action against valid actions
            if (action.equals("AddCustomer")) {
                // First word is name
                String customerName = actionScanner.next();
                // Print first part of output
                System.out.print("AddCustomer " + customerName);

                // Read the next 10 inputs as the customer's ratings for the songs
                // and put them in a int array of size 10
                int[] ratingList = new int[10];
                for (int i = 0; i < 10; i++) {
                    int rating = actionScanner.nextInt();
                    ratingList[i] = rating;
                    // Print rating
                    System.out.print(" " + rating);
                }
                System.out.println(); // Line break

                // Construct a Customer object to store the above information
                Customer newCustomer = new Customer(customerName, ratingList);
                // Add to list
                customerList.add(newCustomer);
                // Re-sort the list
                customerList.sort(Comparator.comparing(customer -> customer.name));
                // Check if the distance is positive and the customer isn't the target
                double distance = newCustomer.findDistance(targetCustomerObj);
                if (distance > 0 && !newCustomer.equals(targetCustomerObj)) {
                    // Add to heap
                    heap.add(newCustomer, distance);
                }

            } else if (action.equals("RecommendSongs")) {
                // Pick the customer with the smallest distance value to recommend songs
                // This customer is located at the root of the heap
                // If this customer cannot recommend songs (no unique song is rated 4+) remove them from the heap
                // If there is no customer in the heap, print none
                System.out.print("RecommendSongs "); // Default print
                String ratingString = "";
                while (ratingString.length() == 0) {
                    if (heap.isEmpty()) {
                        ratingString += "none";
                    } else {
                        // Get closest customer
                        Customer closestCustomer = heap.getMin();
                        // Look through the songs that the customer has rated and the target hasn't
                        // Recommend those songs that have a rating of 4 or 5
                        int[] ratings = closestCustomer.ratings;
                        for (int i = 0; i < 10; i++) {
                            if (ratings[i] >= 4 && targetCustomerObj.ratings[i] == 0) {
                                ratingString += " song" + i + " " + ratings[i];
                            }
                        }
                        // If ratingString is STILL empty, remove the customer from the heap
                        if (ratingString.length() == 0) heap.removeMin();
                        // Otherwise, recommendations have been found and
                        // the customer's name should be printed along with the ratingString
                        else System.out.println(closestCustomer.name + ratingString);
                    }
                }

            } else if (action.equals("PrintCustomerDistanceRatings")) {
                // Print out format header
                System.out.println("PrintCustomerDistanceRatings");
                System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");

                // Print all customers' ratings of each song and the distance to the target customer
                // using the specified format
                for (Customer customer : customerList) {
                    // Check if the specified customer is the target or distance can't be found
                    if (customer.name.equals(targetCustomer)) {
                        System.out.print("         ");
                    } else if (customer.findDistance(targetCustomerObj) == -1) {
                        System.out.print("-----    ");
                    } else {
                        System.out.print(String.format("%.3f", customer.findDistance(targetCustomerObj)) + "    ");
                    }

                    // Print name then the remaining spaces up to column 21
                    System.out.printf("%-11s", customer.name);

                    // Print all ratings
                    for (int i = 0; i < 10; i++) {
                        System.out.print(customer.ratings[i] + "  ");
                    }

                    // New line
                    System.out.println();
                }
            }
        }
    }
}