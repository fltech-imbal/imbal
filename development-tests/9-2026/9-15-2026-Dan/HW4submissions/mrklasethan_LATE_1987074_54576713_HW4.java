import java.io.File;
import java.io.FileNotFoundException;
import java.util.Locale;
import java.util.Scanner;

// stores customers and recommends songs
public class HW4 {
    private static final int SONGS = 10;
    private static final int MAX_CUSTOMERS = 100;

    private static class Customer {
        String name;
        int[] ratings;
        Customer(String name, int[] ratings) {
            this.name = name;
            this.ratings = ratings;
        }
    }

    private static Customer[] customers = new Customer[MAX_CUSTOMERS];
    private static int count = 0;
    private static String targetName;

    // finds a customer using binary search
    private static int findCustomer(String name) {
        int low = 0, high = count - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            int comparison = customers[mid].name.compareTo(name);
            if (comparison == 0) return mid;
            if (comparison < 0) low = mid + 1;
            else high = mid - 1;
        }
        return -(low + 1);
    }

    // adds a customer in alphabetical order
    private static void insertCustomer(String name, int[] ratings) {
        int location = findCustomer(name);
        if (location >= 0) {
            customers[location] = new Customer(name, ratings);
            return;
        }
        if (count == MAX_CUSTOMERS) throw new IllegalStateException("Too many customers");
        location = -location - 1;
        for (int i = count; i > location; i--) customers[i] = customers[i - 1];
        customers[location] = new Customer(name, ratings);
        count++;
    }

    // calculates the distance between two customers
    private static double distance(Customer a, Customer b) {
        int common = 0, differences = 0;
        for (int i = 0; i < SONGS; i++) {
            if (a.ratings[i] != 0 && b.ratings[i] != 0) {
                common++;
                differences += Math.abs(a.ratings[i] - b.ratings[i]);
            }
        }
        if (common == 0) return Double.POSITIVE_INFINITY;
        return (double) differences / common + 1.0 / common;
    }

    private static Customer target() {
        int index = findCustomer(targetName);
        if (index < 0) throw new IllegalArgumentException("Unknown target: " + targetName);
        return customers[index];
    }

    private static void printCustomerDistanceRatings() {
        Customer t = target();
        System.out.println("PrintCustomerDistanceRatings");
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
        for (int i = 0; i < count; i++) {
            Customer c = customers[i];
            String d = c == t ? "" :
                (Double.isInfinite(distance(t, c)) ? "-----" :
                    String.format(Locale.US, "%.3f", distance(t, c)));
            System.out.printf(Locale.US, "%-9s%-10s", d, c.name);
            for (int j = 0; j < SONGS; j++) System.out.printf(j == 0 ? "%2d" : "%3d", c.ratings[j]);
            System.out.println();
        }
    }

    // finds the closest customers using a heap
    private static void recommendSongs() {
        Customer t = target();
        HeapPriorityQueue<Double, Customer> pq = new HeapPriorityQueue<>();
        for (int i = 0; i < count; i++) {
            Customer c = customers[i];
            if (c != t && !Double.isInfinite(distance(t, c)))
                pq.insert(distance(t, c), c);
        }
        Customer closest = null;
        while (!pq.isEmpty()) {
            double minimum = pq.min().getKey();
            Customer best = null;
            // checks customers with the same distance in alphabetical order
            while (!pq.isEmpty() && Double.compare(pq.min().getKey(), minimum) == 0) {
                Customer next = pq.removeMin().getValue();
                if (best == null || next.name.compareTo(best.name) < 0) best = next;
            }
            if (best != null) {
                boolean useful = false;
                for (int j = 0; j < SONGS; j++)
                    if (t.ratings[j] == 0 && best.ratings[j] >= 4) useful = true;
                if (useful) { closest = best; break; }
            }
        }
        System.out.print("RecommendSongs " + (closest == null ? "none" : closest.name));
        if (closest != null) {
            for (int j = 0; j < SONGS; j++)
                if (t.ratings[j] == 0 && closest.ratings[j] >= 4)
                    System.out.print(" song" + j + " " + closest.ratings[j]);
        }
        System.out.println();
    }

    private static int[] readRatings(Scanner input) {
        int[] result = new int[SONGS];
        for (int j = 0; j < SONGS; j++) result[j] = input.nextInt();
        return result;
    }

    public static void main(String[] args) throws FileNotFoundException {
        if (args.length != 2) {
            System.err.println("Usage: java HW4 ratings.txt actions.txt");
            return;
        }
        try (Scanner ratings = new Scanner(new File(args[0]));
             Scanner actions = new Scanner(new File(args[1]))) {
            targetName = ratings.next();
            while (ratings.hasNext()) insertCustomer(ratings.next(), readRatings(ratings));
            while (actions.hasNext()) {
                String action = actions.next();
                if (action.equals("AddCustomer")) {
                    String name = actions.next();
                    int[] values = readRatings(actions);
                    insertCustomer(name, values);
                    System.out.print("AddCustomer " + name);
                    for (int r : values) System.out.print(" " + r);
                    System.out.println();
                } else if (action.equals("RecommendSongs")) recommendSongs();
                else if (action.equals("PrintCustomerDistanceRatings")) printCustomerDistanceRatings();
                else throw new IllegalArgumentException("Unknown action: " + action);
            }
        }
    }
}
