/*
  Author:Carson Rocha
  Email:crocha2025@my.fit.edu
  Course: CSE 2010
  Section:930am
  Description: Recommends songs using customer ratings and a heap priority queue.
*/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Locale;

public class HW4 {

    static final int MAX_CUSTOMERS = 100;
    static final int NUM_SONGS = 10;

    static Customer[] customers = new Customer[MAX_CUSTOMERS];
    static int customerCount = 0;
    static String targetName;

    // Stores one customer's name and ten song ratings.
    static class Customer {
        String name;
        int[] ratings;

        Customer(String name, int[] ratings) {
            this.name = name;
            this.ratings = ratings;
        }
    }

    // The provided HeapPriorityQueue stores distance as the key and name as the value.
    // When two distances are equal, the alphabetically earlier name comes first.
    static class CustomerHeap extends HeapPriorityQueue<Double, String> {

        @Override
        protected int compare(Entry<Double, String> first,
                              Entry<Double, String> second) {
            int difference = Double.compare(first.getKey(), second.getKey());
            if (difference != 0) {
                return difference;
            }
            return first.getValue().compareTo(second.getValue());
        }

        // Returns the minimum entry without removing it.
        Entry<Double, String> getMin() {
            return min();
        }

        // Returns true if the priority queue has reached its capacity.
        boolean isFull() {
            return size() >= MAX_CUSTOMERS;
        }
    }

    // Uses binary search to find a customer in the alphabetically sorted array.
    // Returns -1 when the name is not present.
    static int findCustomer(String name) {
        int low = 0;
        int high = customerCount - 1;

        while (low <= high) {
            int middle = (low + high) / 2;
            int result = customers[middle].name.compareTo(name);

            if (result == 0) {
                return middle;
            } else if (result < 0) {
                low = middle + 1;
            } else {
                high = middle - 1;
            }
        }
        return -1;
    }

    // Adds a customer while keeping the array in alphabetical order.
    // If the customer already exists, replaces their ratings.
    static void insertCustomer(String name, int[] ratings) {
        int existing = findCustomer(name);
        if (existing != -1) {
            customers[existing].ratings = ratings;
            return;
        }

        if (customerCount == MAX_CUSTOMERS) {
            return;
        }

        int position = 0;
        while (position < customerCount &&
               customers[position].name.compareTo(name) < 0) {
            position++;
        }

        for (int i = customerCount; i > position; i--) {
            customers[i] = customers[i - 1];
        }
        customers[position] = new Customer(name, ratings);
        customerCount++;
    }

    // Calculates the distance using only songs rated by both customers.
    // Positive infinity means that they have no songs rated in common.
    static double distance(Customer first, Customer second) {
        int common = 0;
        int differenceSum = 0;

        for (int i = 0; i < NUM_SONGS; i++) {
            if (first.ratings[i] > 0 && second.ratings[i] > 0) {
                common++;
                differenceSum += Math.abs(first.ratings[i] - second.ratings[i]);
            }
        }

        if (common == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return (1.0 + differenceSum) / common;
    }

    // Prints the target first and all other customers in alphabetical order.
    static void printCustomerDistanceRatings() {
        System.out.println("PrintCustomerDistanceRatings");
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");

        int targetIndex = findCustomer(targetName);
        if (targetIndex == -1) {
            return;
        }
        Customer target = customers[targetIndex];
        printCustomerRow("", target);

        for (int i = 0; i < customerCount; i++) {
            Customer current = customers[i];
            if (current.name.equals(targetName)) {
                continue;
            }

            double d = distance(target, current);
            String distanceText;
            if (Double.isInfinite(d)) {
                distanceText = "-----";
            } else {
                distanceText = String.format(Locale.US, "%.3f", d);
            }
            printCustomerRow(distanceText, current);
        }
    }

    // Prints one customer in the required fixed-column table format.
    static void printCustomerRow(String distanceText, Customer customer) {
        System.out.printf("%-9s%-11s", distanceText, customer.name);
        for (int i = 0; i < NUM_SONGS; i++) {
            if (i > 0) {
                System.out.print("  ");
            }
            System.out.print(customer.ratings[i]);
        }
        System.out.println();
    }

    // Picks the closest customer with a liked song (rating 4 or 5)
    // that the target customer has not already rated.
    static void recommendSongs() {
        int targetIndex = findCustomer(targetName);
        if (targetIndex == -1) {
            System.out.println("RecommendSongs none");
            return;
        }
        Customer target = customers[targetIndex];
        CustomerHeap heap = new CustomerHeap();

        // Insert all comparable customers into the min-heap.
        for (int i = 0; i < customerCount; i++) {
            Customer current = customers[i];
            if (current.name.equals(targetName)) {
                continue;
            }

            double d = distance(target, current);
            if (!Double.isInfinite(d) && !heap.isFull()) {
                heap.insert(d, current.name);
            }
        }

        // Try the nearest customer first, and keep trying until
        // someone has an eligible song to recommend.
        while (!heap.isEmpty()) {
            Entry<Double, String> entry = heap.getMin();
            heap.removeMin();
            Customer closest = customers[findCustomer(entry.getValue())];

            boolean foundSong = false;
            for (int i = 0; i < NUM_SONGS; i++) {
                if (target.ratings[i] == 0 && closest.ratings[i] >= 4) {
                    foundSong = true;
                }
            }

            if (foundSong) {
                System.out.print("RecommendSongs " + closest.name);
                for (int i = 0; i < NUM_SONGS; i++) {
                    if (target.ratings[i] == 0 && closest.ratings[i] >= 4) {
                        System.out.print(" song" + i + " " + closest.ratings[i]);
                    }
                }
                System.out.println();
                return;
            }
        }

        System.out.println("RecommendSongs none");
    }

    // Reads the initial target name and customer ratings from the first file.
    static void readRatings(String filename) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line = reader.readLine();
            if (line == null) {
                return;
            }
            targetName = line.trim();

            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length < NUM_SONGS + 1) {
                    continue;
                }
                int[] ratings = new int[NUM_SONGS];
                for (int i = 0; i < NUM_SONGS; i++) {
                    ratings[i] = Integer.parseInt(parts[i + 1]);
                }
                insertCustomer(parts[0], ratings);
            }
        }
    }

    // Runs AddCustomer, RecommendSongs, and PrintCustomerDistanceRatings actions.
    static void readActions(String filename) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");
                if (parts[0].equals("AddCustomer") && parts.length >= NUM_SONGS + 2) {
                    int[] ratings = new int[NUM_SONGS];
                    for (int i = 0; i < NUM_SONGS; i++) {
                        ratings[i] = Integer.parseInt(parts[i + 2]);
                    }
                    insertCustomer(parts[1], ratings);
                    System.out.print("AddCustomer " + parts[1]);
                    for (int rating : ratings) {
                        System.out.print(" " + rating);
                    }
                    System.out.println();
                } else if (parts[0].equals("RecommendSongs")) {
                    recommendSongs();
                } else if (parts[0].equals("PrintCustomerDistanceRatings")) {
                    printCustomerDistanceRatings();
                }
            }
        }
    }

    // First command-line argument: ratings file; second: actions file.
    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("Usage: java HW4 ratingsFile actionsFile");
            return;
        }
        readRatings(args[0]);
        readActions(args[1]);
    }
}
