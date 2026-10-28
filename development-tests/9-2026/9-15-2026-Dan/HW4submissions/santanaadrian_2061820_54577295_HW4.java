// Author: Adrian Santana
// Email: santanaa2025@my.fit.edu
// Course: CSE 2010
// Section: 1
// Description: The program presented below uses a priority queue implemented with a head to recommend songs
// to a customer based another customer's similar music taste. It uses Customer.java and the provided Java files.

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW4 {
    // Declare an array of type Customer to store the 100 customers
    private static Customer[] customers = new Customer[100];
    // Declare a variable to keep track of the number of customers
    private static int customerCount = 0;
    // Declare a variable of type Customer to represent the target customer
    private static Customer targetCust;

    // Create a method that takes in a customer of type Customer and inserts it into the customers
    // array in alphabetical order
    public static void insertCustomer(Customer customer) {

        // If the number of customers is greater than or equal to the size of the array, return nothing
        if (customerCount >= customers.length) {
            return;
        }
        int i = 0;

        // Create a while loop that updates i if it is less than the number of customers and customer names
        // are in alphabetical order
        while (i < customerCount && customers[i].getName().compareTo(customer.getName()) < 0) {
            i++;
        }
        // Use a for loop to shift customers to the right one position
        for (int j = customerCount; j > i; j--) {
            customers[j] = customers[j - 1];
        }
        // Set the customer object at index i to be the customer and update the number of customers
        customers[i] = customer;
        customerCount++;
    }

    // Create a method that takes in a customer's name and uses binary search to find the customer
    // in the customers array
    public static int findCustomer(String name) {
        int low = 0;
        int high = customerCount - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            int compareName = name.compareTo(customers[mid].getName());

            if (compareName == 0) {
                return mid;
            } else if (compareName < 0) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return -1;
    }

    // Create a method that takes in two customer objects and uses the distance formula to calculate their
    // distance based on their ratings
    public static double customerDistance(Customer customer1, Customer customer2) {
        int sharedSongs = 0;
        int absoluteDifference = 0;

        // Use a for loop to go through each song and get each customer's rating
        for (int i = 0; i < 10; i++) {
            int rating1 = customer1.getRatings(i);
            int rating2 = customer2.getRatings(i);
            // If ratings 1 and 2 are not 0, update sharedSongs and find the absolute difference
            if (rating1 != 0 && rating2 != 0) {
                sharedSongs++;
                absoluteDifference += Math.abs(rating1 - rating2);
            }
        }
        // If there are no shared songs, return -1.0
        if (sharedSongs == 0) {
            return -1.0;
        }
        // Plug in the variables into the given distance formula
        return 1.0 / sharedSongs + (double) absoluteDifference / sharedSongs;
    }

    // Create a method that uses a priority queue and compare() from HeapPriorityQueue.java to organize customers based on their distance from
    // the target customer
    public static HeapPriorityQueue<Double, String> findClosestCustomers() {
        HeapPriorityQueue<Double, String> customerQueue = new HeapPriorityQueue<>(Double::compare);
        for (int i = 0; i < customerCount; i++) {
            Customer customer = customers[i];
            // If the current customer's name is not the same as the target customer's name use customerDistance() to
            // calculate the distance between them
            if (!customer.getName().equals(targetCust.getName())) {
                double distance = customerDistance(targetCust, customer);
                if (distance != -1.0) {
                    customerQueue.insert(distance, customer.getName());
                }
            }
        }
        return customerQueue;
    }

    // Create a method that prints each customer's distance, name, and their ratings for each song
    public static void printCustomerDistanceRatings() {
        System.out.println("PrintCustomerDistanceRatings");
        System.out.printf("%-8s %-10s ", "distance", "customer");
        // Create a for loop that prints the heading for each of the songs from 0 to 9
        for (int i = 0; i < 10; i++) {
            System.out.printf("%-3s", "s" + i);
        }
        System.out.println();
        System.out.printf("%8s ", "");
        System.out.printf("%-10s ", targetCust.getName());
        // Create a for loop that print the target customer's rating for each song
        for (int i = 0; i < 10; i++) {
            System.out.printf("%-3d", targetCust.getRatings(i));
        }
        System.out.println();
        // Use a for loop to go through each customer in the customers array and print their calculated
        // distances and names
        for (int i = 0; i < customerCount; i++) {
            if (!customers[i].getName().equals(targetCust.getName())) {
                double distance = customerDistance(targetCust, customers[i]);
                if (distance == -1.0) {
                    System.out.printf("%-8s", "-----");
                } else {
                    System.out.printf("%-8.3f", distance);
                }
                System.out.printf(" %-10s ", customers[i].getName());

                for (int j = 0; j < 10; j++) {
                    System.out.printf("%-3d", customers[i].getRatings(j));
                }
                System.out.println();
            }
        }
    }

    // Create a method that uses the priority queue from findClosestCustomers() to recommend songs the target customer
    // has not rated based on other close customers' ratings
    public static void recommendSongs() {
        HeapPriorityQueue<Double, String> customerQueue = findClosestCustomers();
        while (!customerQueue.isEmpty()) {
            String customerName = customerQueue.removeMin().getValue();
            int index = findCustomer(customerName);
            Customer customer = customers[index];
            // Make a boolean variable to determine if the customer found a song to recommend
            boolean foundSong = false;
            for (int i = 0; i < 10; i++) {
                // If the target customer did not rate the song and another customer's rating is greater than
                // or equal to 4, print the customer's recommendation
                if (targetCust.getRatings(i) == 0 && customer.getRatings(i) >= 4) {
                    if (!foundSong) {
                        System.out.print("RecommendSongs " + customer.getName());
                        foundSong = true;
                    }
                    System.out.print(" song" + i + " " + customer.getRatings(i));
                }
            }
            if (foundSong) {
                System.out.println();
                return;
            }
        }
        // If no song was found to recommend, print the message
        System.out.println("RecommendSongs none");
    }

    // Main method
    public static void main(String[] args) throws FileNotFoundException {
        // Create a file and scanner object to read the ratings file
        Scanner ratingsFile = new Scanner(new File(args[0]));
        String targetCustName = ratingsFile.nextLine();

        // Make a while loop that reads each line of the ratingsFile
        while (ratingsFile.hasNextLine()) {
            String line = ratingsFile.nextLine();

            // If the line is not empty, use the split tool to break and use each part of the line individually
            if (!line.isEmpty()) {
                String[] ratingsParts = line.split(" ");
                String name = ratingsParts[0];
                int[] ratings = new int[10];
                // Use a for loop to go through each song rating and use Integer.parseInt to convert them to integers
                for (int i = 0; i < 10; i++) {
                    ratings[i] = Integer.parseInt(ratingsParts[i + 1]);
                }
                Customer customer = new Customer(name, ratings);
                insertCustomer(customer);

                if (name.equals(targetCustName)) {
                    targetCust = customer;
                }
            }
        }

        // Make a file and scanner object to read the actions file
        Scanner actionsFile = new Scanner(new File(args[1]));

        // Make a while loop that reads each line of the actions file
        while (actionsFile.hasNextLine()) {
            String line = actionsFile.nextLine();

            // If the line is not empty, use the split tool to break and use each part of the line individually
            if (!line.isEmpty()) {
                String[] actionParts = line.split(" ");
                String action = actionParts[0];

                // If the action is "AddCustomer"
                if (action.equals("AddCustomer")) {
                    System.out.println(line);
                    String name = actionParts[1];
                    int[] ratings = new int[10];

                    for (int i = 0; i < 10; i++) {
                        ratings[i] = Integer.parseInt(actionParts[i + 2]);
                    }
                    Customer customer = new Customer(name, ratings);
                    insertCustomer(customer);

                    // If the action is "RecommendSongs", call the recommendSongs() method so it handles the case
                    // and prints the output
                } else if (action.equals("RecommendSongs")) {
                    recommendSongs();

                    // If the action is "PrintCustomerDistanceRatings", call the printCustomerDistanceRatings() method
                    // handle and print the required output
                } else if (action.equals("PrintCustomerDistanceRatings")) {
                    printCustomerDistanceRatings();
                }
            }
        }
    }


}
