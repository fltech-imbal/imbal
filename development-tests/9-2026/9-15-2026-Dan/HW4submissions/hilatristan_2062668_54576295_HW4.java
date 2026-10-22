import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;

/**
 * Recommends songs by finding the customer with the most similar ratings.
 */
public class HW4 {
    private static final int MAX_CUSTOMERS = 100;
    private static final int SONG_COUNT = 10;

    private static Customer[] customers = new Customer[MAX_CUSTOMERS];
    private static int customerCount = 0;
    private static String targetName;

    /** Stores one customer's name and ten song ratings. */
    private static class Customer {
        private final String name;
        private final int[] ratings;

        Customer(String name, int[] ratings) {
            this.name = name;
            this.ratings = ratings;
        }
    }

    /** Orders a queue entry by distance and then by customer name. */
    private static class CustomerDistance implements Comparable<CustomerDistance> {
        private final double distance;
        private final String name;

        CustomerDistance(double distance, String name) {
            this.distance = distance;
            this.name = name;
        }

        @Override
        public int compareTo(CustomerDistance other) {
            int comparison = Double.compare(distance, other.distance);
            return comparison != 0 ? comparison : name.compareTo(other.name);
        }
    }

    /** Reads ratings and performs every requested action. */
    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println("Usage: java HW4 ratingsFile actionsFile");
            return;
        }

        try {
            readInitialCustomers(args[0]);
            performActions(args[1]);
        } catch (IOException | IllegalArgumentException exception) {
            System.err.println("Input error: " + exception.getMessage());
        }
    }

    /** Reads the target customer and the initial customers. */
    private static void readInitialCustomers(String filename) throws FileNotFoundException {
        try (Scanner input = new Scanner(new File(filename))) {
            if (!input.hasNext()) {
                throw new IllegalArgumentException("ratings file is empty");
            }

            targetName = input.next();
            while (input.hasNext()) {
                String name = input.next();
                insertCustomer(name, readRatings(input));
            }
        }

        if (findCustomer(targetName) == null) {
            throw new IllegalArgumentException("target customer was not found");
        }
    }

    /** Reads and runs one action per line. */
    private static void performActions(String filename) throws IOException {
        try (BufferedReader input = new BufferedReader(new java.io.FileReader(filename))) {
            String line;
            while ((line = input.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }

                String[] parts = line.split("\\s+");
                switch (parts[0]) {
                    case "AddCustomer":
                        addCustomer(parts, line);
                        break;
                    case "RecommendSongs":
                        System.out.println("RecommendSongs " + recommendSongs());
                        break;
                    case "PrintCustomerDistanceRatings":
                        printCustomerDistanceRatings();
                        break;
                    default:
                        throw new IllegalArgumentException("unknown action: " + parts[0]);
                }
            }
        }
    }

    /** Adds a customer while preserving alphabetical order. */
    private static void addCustomer(String[] parts, String originalLine) {
        if (parts.length != SONG_COUNT + 2) {
            throw new IllegalArgumentException("AddCustomer needs a name and ten ratings");
        }

        int[] ratings = new int[SONG_COUNT];
        for (int song = 0; song < SONG_COUNT; song++) {
            ratings[song] = parseRating(parts[song + 2]);
        }
        insertCustomer(parts[1], ratings);
        System.out.println(originalLine);
    }

    /** Inserts or replaces a customer using a binary-search position. */
    private static void insertCustomer(String name, int[] ratings) {
        if (name.length() > 10) {
            throw new IllegalArgumentException("customer name is longer than ten letters");
        }

        int position = findCustomerPosition(name);
        if (position >= 0) {
            customers[position] = new Customer(name, ratings);
            return;
        }
        if (customerCount == MAX_CUSTOMERS) {
            throw new IllegalArgumentException("too many customers");
        }

        int insertionPoint = -position - 1;
        for (int index = customerCount; index > insertionPoint; index--) {
            customers[index] = customers[index - 1];
        }
        customers[insertionPoint] = new Customer(name, ratings);
        customerCount++;
    }

    /** Finds a customer with binary search. */
    private static Customer findCustomer(String name) {
        int position = findCustomerPosition(name);
        return position >= 0 ? customers[position] : null;
    }

    /** Returns an index or Java's negative insertion-point encoding. */
    private static int findCustomerPosition(String name) {
        int low = 0;
        int high = customerCount - 1;
        while (low <= high) {
            int middle = (low + high) / 2;
            int comparison = name.compareTo(customers[middle].name);
            if (comparison == 0) {
                return middle;
            }
            if (comparison < 0) {
                high = middle - 1;
            } else {
                low = middle + 1;
            }
        }
        return -low - 1;
    }

    /** Prints the target first and all other customers alphabetically. */
    private static void printCustomerDistanceRatings() {
        Customer target = findCustomer(targetName);
        System.out.println("PrintCustomerDistanceRatings");
        System.out.printf("%-9s%-11s", "distance", "customer");
        for (int song = 0; song < SONG_COUNT; song++) {
            System.out.print(song == 0 ? "s0" : " s" + song);
        }
        System.out.println();

        printCustomerRow(target, Double.NaN);
        for (int index = 0; index < customerCount; index++) {
            Customer customer = customers[index];
            if (customer != target) {
                printCustomerRow(customer, distance(target, customer));
            }
        }
    }

    /** Prints one fixed-width customer row. */
    private static void printCustomerRow(Customer customer, double customerDistance) {
        String distanceText;
        if (Double.isNaN(customerDistance)) {
            distanceText = "";
        } else if (Double.isInfinite(customerDistance)) {
            distanceText = "-----";
        } else {
            distanceText = String.format("%.3f", customerDistance);
        }
        System.out.printf("%-9s%-9s", distanceText, customer.name);
        for (int rating : customer.ratings) {
            System.out.printf("%3d", rating);
        }
        System.out.println();
    }

    /** Finds the nearest customer who has a strong unrated recommendation. */
    private static String recommendSongs() {
        Customer target = findCustomer(targetName);
        HeapPriorityQueue<CustomerDistance, Customer> queue = buildDistanceQueue(target);

        while (!queue.isEmpty()) {
            Customer closest = queue.removeMin().getValue();
            String songs = recommendedSongs(target, closest);
            if (!songs.isEmpty()) {
                return closest.name + songs;
            }
        }
        return "none";
    }

    /** Creates a priority queue of all customers with a defined distance. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private static HeapPriorityQueue<CustomerDistance, Customer> buildDistanceQueue(Customer target) {
        Entry<CustomerDistance, Customer>[] entries = (Entry<CustomerDistance, Customer>[]) new Entry[customerCount];
        int entryCount = 0;
        for (int index = 0; index < customerCount; index++) {
            Customer customer = customers[index];
            if (customer != target) {
                double customerDistance = distance(target, customer);
                if (!Double.isInfinite(customerDistance)) {
                    entries[entryCount++] = new AbstractPriorityQueue.PQEntry<>(
                            new CustomerDistance(customerDistance, customer.name), customer);
                }
            }
        }

        HeapPriorityQueue<CustomerDistance, Customer> queue = new HeapPriorityQueue<>();
        queue.initialize(entries, entryCount);
        return queue;
    }

    /** Returns recommended songs in song-number order. */
    private static String recommendedSongs(Customer target, Customer closest) {
        StringBuilder result = new StringBuilder();
        for (int song = 0; song < SONG_COUNT; song++) {
            if (target.ratings[song] == 0 && closest.ratings[song] >= 4) {
                result.append(" song").append(song).append(' ').append(closest.ratings[song]);
            }
        }
        return result.toString();
    }

    /** Calculates the assignment's rating distance. */
    private static double distance(Customer first, Customer second) {
        int sharedRatings = 0;
        int totalDifference = 0;
        for (int song = 0; song < SONG_COUNT; song++) {
            if (first.ratings[song] != 0 && second.ratings[song] != 0) {
                sharedRatings++;
                totalDifference += Math.abs(first.ratings[song] - second.ratings[song]);
            }
        }
        if (sharedRatings == 0) {
            return Double.POSITIVE_INFINITY;
        }
        return 1.0 / sharedRatings + (double) totalDifference / sharedRatings;
    }

    /** Reads ten validated ratings. */
    private static int[] readRatings(Scanner input) {
        int[] ratings = new int[SONG_COUNT];
        for (int song = 0; song < SONG_COUNT; song++) {
            if (!input.hasNext()) {
                throw new IllegalArgumentException("customer is missing a rating");
            }
            ratings[song] = parseRating(input.next());
        }
        return ratings;
    }

    /** Converts a valid rating token to an integer. */
    private static int parseRating(String token) {
        try {
            int rating = Integer.parseInt(token);
            if (rating < 0 || rating > 5) {
                throw new IllegalArgumentException("rating must be from 0 through 5");
            }
            return rating;
        } catch (NumberFormatException exception) {
            throw new IllegalArgumentException("invalid rating: " + token);
        }
    }
}
