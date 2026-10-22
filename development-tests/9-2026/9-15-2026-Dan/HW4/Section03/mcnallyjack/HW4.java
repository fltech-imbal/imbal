import java.io.*;
import java.util.*;

public class HW4 {
    static final int MAX_CUSTOMERS = 100;
    static final int NUM_SONGS = 10;

    static class Customer {
        String name;
        int[] ratings;

        Customer(String name, int[] ratings) {
            this.name = name;
            this.ratings = ratings.clone();
        }
    }

    static class Entry {
        double distance;
        String name;

        Entry(double distance, String name) {
            this.distance = distance;
            this.name = name;
        }
    }

    static class MinHeap {
        Entry[] heap = new Entry[MAX_CUSTOMERS];
        int size = 0;

        boolean before(Entry a, Entry b) {
            int cmp = Double.compare(a.distance, b.distance);
            return cmp < 0 || (cmp == 0 &&
                    a.name.compareTo(b.name) < 0);
        }

        boolean isEmpty() {
            return size == 0;
        }

        boolean isFull() {
            return size == heap.length;
        }

        Entry getMin() {
            if (isEmpty()) return null;
            return heap[0];
        }

        void insert(Entry e) {
            if (isFull()) return;

            int i = size++;
            while (i > 0) {
                int parent = (i - 1) / 2;
                if (!before(e, heap[parent])) break;
                heap[i] = heap[parent];
                i = parent;
            }
            heap[i] = e;
        }

        Entry removeMin() {
            if (isEmpty()) return null;

            Entry min = heap[0];
            Entry last = heap[--size];
            heap[size] = null;

            if (size > 0) {
                int i = 0;
                while (2 * i + 1 < size) {
                    int child = 2 * i + 1;
                    if (child + 1 < size &&
                            before(heap[child + 1], heap[child])) {
                        child++;
                    }

                    if (!before(heap[child], last)) break;
                    heap[i] = heap[child];
                    i = child;
                }
                heap[i] = last;
            }

            return min;
        }

        void initialize(Entry[] entries, int numEntries) {
            size = Math.min(numEntries, heap.length);
            for (int i = 0; i < size; i++) {
                heap[i] = entries[i];
            }
            for (int i = size / 2 - 1; i >= 0; i--) {
                siftDown(i);
            }
        }

        void siftDown(int i) {
            Entry item = heap[i];

            while (2 * i + 1 < size) {
                int child = 2 * i + 1;

                if (child + 1 < size &&
                        before(heap[child + 1], heap[child])) {
                    child++;
                }

                if (!before(heap[child], item)) break;
                heap[i] = heap[child];
                i = child;
            }

            heap[i] = item;
        }
    }

    static Customer[] customers = new Customer[MAX_CUSTOMERS];
    static int customerCount = 0;
    static Customer target;

    static int findIndex(String name) {
        int low = 0, high = customerCount - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            int cmp = customers[mid].name.compareTo(name);

            if (cmp == 0) return mid;
            if (cmp < 0) low = mid + 1;
            else high = mid - 1;
        }

        return -1;
    }

    static Customer findCustomer(String name) {
        int i = findIndex(name);
        return i < 0 ? null : customers[i];
    }

    static boolean insertCustomer(String name, int[] ratings) {
        int existing = findIndex(name);

        if (existing >= 0) {
            customers[existing] = new Customer(name, ratings);
            if (name.equals(target.name)) {
                target = customers[existing];
            }
            return true;
        }

        if (customerCount >= MAX_CUSTOMERS) return false;

        int i = customerCount;
        while (i > 0 && customers[i - 1].name.compareTo(name) > 0) {
            customers[i] = customers[i - 1];
            i--;
        }

        customers[i] = new Customer(name, ratings);
        customerCount++;

        if (name.equals(target.name)) target = customers[i];
        return true;
    }

    static double distance(Customer x, Customer y) {
        int count = 0;
        int sum = 0;

        for (int i = 0; i < NUM_SONGS; i++) {
            if (x.ratings[i] > 0 && y.ratings[i] > 0) {
                count++;
                sum += Math.abs(x.ratings[i] - y.ratings[i]);
            }
        }

        if (count == 0) return -1.0;
        return 1.0 / count + (double) sum / count;
    }

    static MinHeap buildHeap() {
        Entry[] entries = new Entry[MAX_CUSTOMERS];
        int count = 0;

        for (int i = 0; i < customerCount; i++) {
            Customer c = customers[i];

            if (c.name.equals(target.name)) continue;

            double d = distance(target, c);
            if (d >= 0) {
                entries[count++] = new Entry(d, c.name);
            }
        }

        MinHeap heap = new MinHeap();
        heap.initialize(entries, count);
        return heap;
    }

    static void recommendSongs() {
        MinHeap heap = buildHeap();

        while (!heap.isEmpty()) {
            Entry entry = heap.removeMin();
            Customer candidate = findCustomer(entry.name);

            StringBuilder result = new StringBuilder();
            for (int i = 0; i < NUM_SONGS; i++) {
                if (target.ratings[i] == 0 &&
                        candidate.ratings[i] >= 4) {
                    result.append(" ").append(i)
                          .append(" ").append(candidate.ratings[i]);
                }
            }

            if (result.length() > 0) {
                System.out.println("RecommendSongs " +
                        candidate.name + result);
                return;
            }
        }

        System.out.println("RecommendSongs none");
    }

    static void printCustomerDistanceRatings() {
        System.out.printf("%-9s%-11s", "distance", "customer");
        for (int i = 0; i < NUM_SONGS; i++) {
            System.out.printf("s%-2d", i);
        }
        System.out.println();

        printRatingRow(target, true);

        for (int i = 0; i < customerCount; i++) {
            Customer c = customers[i];

            if (c.name.equals(target.name)) continue;
            printRatingRow(c, false);
        }
    }

    static void printRatingRow(Customer c, boolean isTarget) {
        if (isTarget) {
            System.out.printf("%-9s%-11s", "-----", c.name);
        } else {
            double d = distance(target, c);
            if (d < 0) {
                System.out.printf("%-9s", "-----");
            } else {
                System.out.printf("%-9.3f", d);
            }
            System.out.printf("%-11s", c.name);
        }

        for (int rating : c.ratings) {
            System.out.printf("%-3d", rating);
        }
        System.out.println();
    }

    static int[] parseRatings(String[] parts) {
        if (parts.length < 11) {
            throw new IllegalArgumentException(
                    "Each customer needs 10 ratings.");
        }

        int[] ratings = new int[NUM_SONGS];

        for (int i = 0; i < NUM_SONGS; i++) {
            ratings[i] = Integer.parseInt(parts[i + 1]);
            if (ratings[i] < 0 || ratings[i] > 5) {
                throw new IllegalArgumentException("Invalid rating.");
            }
        }

        return ratings;
    }

    static void readInitialFile(String filename) throws IOException {
        try (BufferedReader br = new BufferedReader(
                new FileReader(filename))) {
            String line = br.readLine();

            if (line == null || line.trim().isEmpty()) {
                throw new IOException("Missing target customer name.");
            }

            String targetName = line.trim();

            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");
                insertCustomer(parts[0], parseRatings(parts));
            }

            target = findCustomer(targetName);

            if (target == null) {
                throw new IOException(
                        "Target customer not found: " + targetName);
            }
        }
    }

    static void processActions(String filename) throws IOException {
        try (BufferedReader br = new BufferedReader(
                new FileReader(filename))) {
            String line;

            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");

                switch (parts[0]) {
                    case "AddCustomer": {
                        if (parts.length < 12) {
                            System.err.println(
                                    "Invalid AddCustomer action: " + line);
                            break;
                        }

                        String name = parts[1];
                        int[] ratings = parseRatings(
                                Arrays.copyOfRange(parts, 1, parts.length));

                        if (insertCustomer(name, ratings)) {
                            System.out.println(line);
                        }
                        break;
                    }

                    case "RecommendSongs":
                        recommendSongs();
                        break;

                    case "PrintCustomerDistanceRatings":
                        printCustomerDistanceRatings();
                        break;

                    default:
                        System.err.println("Unknown action: " + line);
                }
            }
        }
    }

    public static void main(String[] args) {
        if (args.length != 2) {
            System.err.println(
                    "Usage: java HW4 initialRatingsFile actionsFile");
            return;
        }

        try {
            readInitialFile(args[0]);
            processActions(args[1]);
        } catch (IOException | IllegalArgumentException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}