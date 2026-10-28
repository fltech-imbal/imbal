/*
Author: Zane Roberts
Email: zroberts2025@my.fit.edu
Course: CES 2002
Section: 3
Description of this file: This program sorts through people and their ratings of different songs
*/
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * Music Recommendation System.
 */
public class hw4 {
    private static final int MAX_CUSTOMERS = 100;
    private static final int NUM_SONGS = 10;

    private static Customer[] customers = new Customer[MAX_CUSTOMERS];
    private static int customerCount = 0;
    private static String targetCustomerName = "";

    //
    // NESTED INTERFACES & CLASSES
    //

    /**
     * Interface representing a key-value pair entry.
     */
    public interface Entry<K, V> {
        /** Returns the key stored in this entry. */
        K getKey();

        /** Returns the value stored in this entry. */
        V getValue();
    }

    /**
     * A self-contained heap-based priority queue implementation using an ArrayList.
     */
    public static class HeapPriorityQueue<K extends Comparable<K>, V extends Comparable<V>> {
        private ArrayList<Entry<K, V>> heap = new ArrayList<>();
        private static final int CAPACITY = 100;

        /**
         * use of the Entry interface for the priority queue.
         */
        private static class PQEntry<K, V> implements Entry<K, V> {
            private K key;
            private V value;

            public PQEntry(K key, V value) {
                this.key = key;
                this.value = value;
            }

            public K getKey() { return key; }
            public V getValue() { return value; }
        }

        protected int parent(int j) { return (j - 1) / 2; }
        protected int left(int j) { return 2 * j + 1; }
        protected int right(int j) { return 2 * j + 2; }
        protected boolean hasLeft(int j) { return left(j) < heap.size(); }
        protected boolean hasRight(int j) { return right(j) < heap.size(); }

        protected void swap(int i, int j) {
            Entry<K, V> temp = heap.get(i);
            heap.set(i, heap.get(j));
            heap.set(j, temp);
        }

        protected int compare(Entry<K, V> a, Entry<K, V> b) {
            int cmp = a.getKey().compareTo(b.getKey());
            if (cmp != 0) {
                return cmp;
            }
            return a.getValue().compareTo(b.getValue());
        }

        protected void upheap(int j) {
            while (j > 0) {
                int p = parent(j);
                if (compare(heap.get(j), heap.get(p)) >= 0) break;
                swap(j, p);
                j = p;
            }
        }

        protected void downheap(int j) {
            while (hasLeft(j)) {
                int leftIndex = left(j);
                int smallChildIndex = leftIndex;
                if (hasRight(j)) {
                    int rightIndex = right(j);
                    if (compare(heap.get(leftIndex), heap.get(rightIndex)) > 0) {
                        smallChildIndex = rightIndex;
                    }
                }
                if (compare(heap.get(smallChildIndex), heap.get(j)) >= 0) break;
                swap(j, smallChildIndex);
                j = smallChildIndex;
            }
        }

        public int size() { return heap.size(); }
        public boolean isEmpty() { return heap.isEmpty(); }
        public boolean isFull() { return heap.size() >= CAPACITY; }

        public Entry<K, V> min() {
            if (heap.isEmpty()) return null;
            return heap.get(0);
        }

        public Entry<K, V> getMin() {
            return min();
        }

        public Entry<K, V> insert(K key, V value) {
            if (isFull()) return null;
            Entry<K, V> newest = new PQEntry<>(key, value);
            heap.add(newest);
            upheap(heap.size() - 1);
            return newest;
        }

        public Entry<K, V> removeMin() {
            if (heap.isEmpty()) return null;
            Entry<K, V> answer = heap.get(0);
            swap(0, heap.size() - 1);
            heap.remove(heap.size() - 1);
            downheap(0);
            return answer;
        }
    }

    /**
     * Represents a customer with a name and an array of 10 song ratings.
     */
    public static class Customer implements Comparable<Customer> {
        private String name;
        private int[] ratings;

        /**
         * Constructs a Customer object with a given name and rating array.
         */
        public Customer(String name, int[] ratings) {
            this.name = name != null ? name.trim() : "";
            this.ratings = ratings != null ? ratings : new int[NUM_SONGS];
        }

        /** Returns the name of the customer. */
        public String getName() { return name; }

        /** Returns the array of song ratings for this customer. */
        public int[] getRatings() { return ratings; }

        /**
         * Compares this customer to another customer lexicographically by name.
         */
        @Override
        public int compareTo(Customer other) {
            if (other == null) return 1;
            return this.name.compareTo(other.name);
        }
    }

    //
    // MAIN PROGRAM & DRIVER METHODS
    //

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage: java HW4 <ratings_filename> <actions_filename>");
            return;
        }

        loadRatings(args[0]);
        processActions(args[1]);
    }

    /**
     * Reads the initial ratings file, pulls the target customer name from the first line,
     * and loads subsequent customer ratings into the array while maintaining alphabetical order.
     */
    public static void loadRatings(String filename) {
        try {
            Scanner scanner = new Scanner(new File(filename));
            if (scanner.hasNextLine()) {
                targetCustomerName = scanner.nextLine().trim().replaceAll("\r", "");
            }

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");
                if (parts.length < (NUM_SONGS + 1)) continue;

                String name = parts[0].trim().replaceAll("\r", "");
                int[] ratings = new int[NUM_SONGS];
                for (int i = 0; i < NUM_SONGS; i++) {
                    ratings[i] = Integer.parseInt(parts[i + 1]);
                }
                insertCustomer(name, ratings);
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error: Ratings file not found - " + filename);
        }
    }

    /**
     * Inserts a new customer into the customer array while maintaining alphabetical order.
     */
    public static void insertCustomer(String name, int[] ratings) {
        if (customerCount >= MAX_CUSTOMERS || name == null) {
            return;
        }

        Customer newCust = new Customer(name, ratings);
        int i = customerCount - 1;

        while (i >= 0 && customers[i] != null && customers[i].getName().compareTo(name) > 0) {
            customers[i + 1] = customers[i];
            i--;
        }
        customers[i + 1] = newCust;
        customerCount++;
    }

    /**
     * Searches for a customer by name using binary search.
     */
    public static Customer findCustomer(String name) {
        if (name == null) return null;

        int low = 0;
        int high = customerCount - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (customers[mid] == null || customers[mid].getName() == null) {
                low = mid + 1;
                continue;
            }

            int cmp = customers[mid].getName().compareTo(name);

            if (cmp == 0) {
                return customers[mid];
            } else if (cmp < 0) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
        return null;
    }

    /**
     * Reads and executes actions sequentially from the specified actions text file.
     */
    public static void processActions(String filename) {
        try {
            Scanner scanner = new Scanner(new File(filename));
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");
                String action = parts[0];

                if (action.equals("AddCustomer")) {
                    if (parts.length < (NUM_SONGS + 2)) continue;
                    String name = parts[1].trim().replaceAll("\r", "");
                    int[] ratings = new int[NUM_SONGS];
                    for (int i = 0; i < NUM_SONGS; i++) {
                        ratings[i] = Integer.parseInt(parts[i + 2]);
                    }
                    insertCustomer(name, ratings);
                    System.out.println(line);
                } else if (action.equals("RecommendSongs")) {
                    recommendSongs();
                } else if (action.equals("PrintCustomerDistanceRatings")) {
                    printCustomerDistanceRatings();
                }
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            System.out.println("Error: Actions file not found - " + filename);
        }
    }

    /**
     * Calculates the musical taste distance between two customers x and y based on formula
     */
    public static double calculateDistance(Customer x, Customer y) {
        if (x == null || y == null) {
            return -1.0;
        }

        int[] rX = x.getRatings();
        int[] rY = y.getRatings();

        if (rX == null || rY == null) {
            return -1.0;
        }

        int sharedCount = 0;
        int absoluteDiffSum = 0;

        for (int i = 0; i < NUM_SONGS; i++) {
            if (i < rX.length && i < rY.length && rX[i] > 0 && rY[i] > 0) {
                sharedCount++;
                absoluteDiffSum += Math.abs(rX[i] - rY[i]);
            }
        }

        if (sharedCount == 0) {
            return -1.0;
        }

        return (1.0 / sharedCount) + ((1.0 / sharedCount) * absoluteDiffSum);
    }

    /**
     * Recommends songs for the target customer by finding the most similar customer 
     * using the heap priority queue
     */
    public static void recommendSongs() {
        Customer target = findCustomer(targetCustomerName);
        if (target == null) {
            System.out.println("RecommendSongs none");
            return;
        }

        HeapPriorityQueue<Double, String> pq = new HeapPriorityQueue<>();

        for (int i = 0; i < customerCount; i++) {
            Customer other = customers[i];
            if (other == null || other.getName().equals(targetCustomerName)) continue;

            double dist = calculateDistance(target, other);
            if (dist >= 0) {
                pq.insert(dist, other.getName());
            }
        }

        while (!pq.isEmpty()) {
            Entry<Double, String> entry = pq.removeMin();
            if (entry == null || entry.getValue() == null) continue;

            String closestName = entry.getValue();
            Customer closestCust = findCustomer(closestName);
            if (closestCust == null) continue;

            StringBuilder recommendationBuilder = new StringBuilder();
            recommendationBuilder.append("RecommendSongs ").append(closestName);

            boolean hasValidRecommendation = false;
            int[] tRatings = target.getRatings();
            int[] cRatings = closestCust.getRatings();

            for (int i = 0; i < NUM_SONGS; i++) {
                if (cRatings[i] >= 4 && tRatings[i] == 0) {
                    recommendationBuilder.append(" song").append(i).append(" ").append(cRatings[i]);
                    hasValidRecommendation = true;
                }
            }

            if (hasValidRecommendation) {
                System.out.println(recommendationBuilder.toString());
                return;
            }
        }

        System.out.println("RecommendSongs none");
    }

    /**
     * Prints a formatted table of customer distance ratings
     */
    public static void printCustomerDistanceRatings() {
        System.out.println("PrintCustomerDistanceRatings");
        Customer target = findCustomer(targetCustomerName);
        if (target == null) {
            return;
        }

        System.out.printf("%-9s %-11s", "distance", "customer");
        for (int i = 0; i < NUM_SONGS; i++) {
            System.out.printf("s%d%s", i, (i == NUM_SONGS - 1 ? "" : " "));
        }
        System.out.println();

        printCustomerRow(target, null);

        for (int i = 0; i < customerCount; i++) {
            Customer c = customers[i];
            if (c != null && !c.getName().equals(targetCustomerName)) {
                printCustomerRow(c, target);
            }
        }
    }

    /**
     * Helper method to format and print a single customer's row in the distance table.
     */
    public static void printCustomerRow(Customer customer, Customer targetCust) {
        if (customer == null) return;

        if (targetCust == null || customer.getName().equals(targetCust.getName())) {
            System.out.printf("%-9s %-11s", "-----", customer.getName());
        } else {
            double dist = calculateDistance(targetCust, customer);
            if (dist < 0) {
                System.out.printf("%-9s %-11s", "-----", customer.getName());
            } else {
                System.out.printf("%-9.3f %-11s", dist, customer.getName());
            }
        }

        int[] ratings = customer.getRatings();
        for (int i = 0; i < NUM_SONGS; i++) {
            int val = (ratings != null && i < ratings.length) ? ratings[i] : 0;
            System.out.printf("  %d", val);
        }
        System.out.println();
    }
}