
/*

  Author: Joseph Venech
  Email: jvenech2025@my.fit.edu
  Course: CSE2010
  Section: 1
  Description: This program manages customer song ratings, calculates
      distances between customers, prints customer distance tables,
      and recommends songs using a priority queue.

*/

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;

public class HW4Extra {

    // Store customer's name and ratings
    public class Customer {
        int[] ratings;
        String name;

        Customer(String name, int[] ratings) {
            this.name = name;
            this.ratings = ratings;
        }
    }

    // Stores a customer's distance from the target
    public class CustomerDistance implements Comparable<CustomerDistance> {

        double distance;
        String name;

        public CustomerDistance(double distance, String name) {
            this.distance = distance;
            this.name = name;
        }

        @Override
        public int compareTo(CustomerDistance other) {
            int result = Double.compare(distance, other.distance);

            if (result != 0) {
                return result;
            }

            return name.compareTo(other.name);
        }
    }

    // Entry interface for the priority queue
    public interface Entry<K, V> {
        K getKey();
        V getValue();
    }

    // Stores a key-value pair
    private static class DataEntry<K, V> implements Entry<K, V> {
        private K key;
        private V value;

        public DataEntry(K key, V value) {
            this.key = key;
            this.value = value;
        }

        @Override
        public K getKey() {
            return key;
        }

        @Override
        public V getValue() {
            return value;
        }
    }

    // Min-priority queue implemented using a binary heap
    public static class HeapPriorityQueue<K extends Comparable<K>, V> {

        private ArrayList<Entry<K, V>> heap = new ArrayList<>();

        public int size() {
            return heap.size();
        }

        public boolean isEmpty() {
            return heap.isEmpty();
        }

        // Insert one entry into the heap
        public void insert(K key, V value) {
            heap.add(new DataEntry<>(key, value));
            upheap(heap.size() - 1);
        }

        // bottom-up heap construction
        public void initialize(ArrayList<Entry<K, V>> entries) {

            heap.clear();
            heap.addAll(entries);

            for (int i = heap.size() / 2 - 1; i >= 0; i--) {
                downheap(i);
            }
        }

        // Return the entry with the smallest key
        public Entry<K, V> min() {
            if (heap.isEmpty()) {
                return null;
            }

            return heap.get(0);
        }

        // Remove and return the entry with the smallest key
        public Entry<K, V> removeMin() {
            if (heap.isEmpty()) {
                return null;
            }

            Entry<K, V> answer = heap.get(0);
            Entry<K, V> last = heap.remove(heap.size() - 1);

            if (!heap.isEmpty()) {
                heap.set(0, last);
                downheap(0);
            }

            return answer;
        }

        // Move an entry upward to restore heap order
        private void upheap(int index) {
            while (index > 0) {
                int parent = (index - 1) / 2;

                if (heap.get(index).getKey().compareTo(heap.get(parent).getKey()) >= 0) {
                    break;
                }

                swap(index, parent);
                index = parent;
            }
        }

        // Move an entry downward to restore heap order
        private void downheap(int index) {
            while (2 * index + 1 < heap.size()) {
                int left = 2 * index + 1;
                int smallChild = left;
                int right = left + 1;

                if (right < heap.size() && heap.get(right).getKey().compareTo(heap.get(left).getKey()) < 0) {
                    smallChild = right;
                }

                if (heap.get(index).getKey().compareTo(heap.get(smallChild).getKey()) <= 0) {
                    break;
                }

                swap(index, smallChild);
                index = smallChild;
            }
        }

        // Swap two entries
        private void swap(int first, int second) {
            Entry<K, V> temp = heap.get(first);
            heap.set(first, heap.get(second));
            heap.set(second, temp);
        }
    }

    // Array that stores the customers
    private Customer[] customers = new Customer[100];

    // Number of customers currently stored
    private int customerCount = 0;

    // Insert customer while maintaining alphabetical order
    public void insertCustomer(String name, int[] ratings) {
        if (customerCount >= customers.length) {
            return;
        }

        Customer newCustomer = new Customer(name, ratings);

        int i = customerCount - 1;

        while (i >= 0 && customers[i].name.compareTo(name) > 0) {
            customers[i + 1] = customers[i];
            i--;
        }

        customers[i + 1] = newCustomer;
        customerCount++;
    }

    // Find customer by name using binary search
    public Customer findCustomer(String name) {
        int low = 0;
        int high = customerCount - 1;

        while (low <= high) {
            int mid = (low + high) / 2;
            int comparison = customers[mid].name.compareTo(name);

            if (comparison == 0) {
                return customers[mid];
            } else if (comparison < 0) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }

        return null;
    }

    // Calculate distance between two customers
    public double calculateDistance(Customer x, Customer y) {
        int commonSongs = 0;
        int totalDifference = 0;

        for (int i = 0; i < 10; i++) {
            if (x.ratings[i] != 0 && y.ratings[i] != 0) {
                commonSongs++;
                totalDifference += Math.abs(x.ratings[i] - y.ratings[i]);
            }
        }

        if (commonSongs == 0) {
            return -1.0;
        }

        return (1.0 / commonSongs) + ((double) totalDifference / commonSongs);
    }

    // Print target customer and all other customers
    public void printCustomerDistanceRatings(String targetName) {
        Customer target = findCustomer(targetName);

        if (target == null) {
            return;
        }

        System.out.println("distance customer s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");

        System.out.print(target.name);

        for (int i = 0; i < 10; i++) {
            System.out.print(" " + target.ratings[i]);
        }

        System.out.println();

        for (int i = 0; i < customerCount; i++) {
            Customer current = customers[i];

            if (current.name.equals(targetName)) {
                continue;
            }

            double distance = calculateDistance(target, current);

            if (distance < 0) {
                System.out.print("-----");
            } else {
                System.out.printf("%.3f", distance);
            }

            System.out.print(" " + current.name);

            for (int j = 0; j < 10; j++) {
                System.out.print(" " + current.ratings[j]);
            }

            System.out.println();
        }
    }

    // Recommend songs that the target customer has not rated
    public void recommendSongs(String targetName) {
        Customer target = findCustomer(targetName);

        if (target == null) {
            System.out.println("none");
            return;
        }

        HeapPriorityQueue<CustomerDistance, String> queue = new HeapPriorityQueue<>();

        ArrayList<Entry<CustomerDistance, String>> entries = new ArrayList<>();

        for (int i = 0; i < customerCount; i++) {
            Customer current = customers[i];

            if (current.name.equals(targetName)) {
                continue;
            }

            double distance = calculateDistance(target, current);

            if (distance >= 0) {
                CustomerDistance key = new CustomerDistance(distance, current.name);

                entries.add(new DataEntry<>(key, current.name));
            }
        }

        
        queue.initialize(entries);

        boolean[] recommended = new boolean[10];
        int count = 0;

        // Process customers from closest to farthest
        while (!queue.isEmpty() && count < 3) {
            Entry<CustomerDistance, String> entry =
                    queue.removeMin();

            Customer current = findCustomer(entry.getValue());

            int previousCount = count;

            for (int i = 0; i < 10 && count < 3; i++) {
                if (target.ratings[i] == 0 && current.ratings[i] > 3 && !recommended[i]) {

                    if (count == 0) {
                        System.out.print(current.name);
                    }

                    System.out.print(
                            " song" + i + " " + current.ratings[i]);

                    recommended[i] = true;
                    count++;
                }
            }

            // Stop after the closest customer who has recommendations
            if (count > previousCount) {
                break;
            }
        }

        if (count == 0) {
            System.out.print("none");
        }

        System.out.println();
    }

    public static void main(String[] args) throws FileNotFoundException {

        HW4 program = new HW4();

        String ratingsFile = args[0];
        String actionsFile = args[1];

        Scanner ratingsScanner = new Scanner(new File(ratingsFile));

        String targetCustomerName = "";

        // Read target customer name
        if (ratingsScanner.hasNextLine()) {
            targetCustomerName =
                    ratingsScanner.nextLine().trim();
        }

        // Read initial customer ratings
        while (ratingsScanner.hasNextLine()) {
            String line = ratingsScanner.nextLine().trim();

            if (line.length() == 0) {
                continue;
            }

            Scanner lineScanner = new Scanner(line);

            String name = lineScanner.next();
            int[] ratings = new int[10];

            for (int i = 0; i < 10; i++) {
                ratings[i] = lineScanner.nextInt();
            }

            program.insertCustomer(name, ratings);
            lineScanner.close();
        }

        ratingsScanner.close();

        Scanner actionsScanner = new Scanner(new File(actionsFile));

        // Process actions
        while (actionsScanner.hasNextLine()) {
            String line = actionsScanner.nextLine().trim();

            if (line.length() == 0) {
                continue;
            }

            Scanner lineScanner = new Scanner(line);
            String action = lineScanner.next();

            if (action.equals("RecommendSongs")) {
                System.out.print("RecommendSongs ");
                program.recommendSongs(targetCustomerName);

            } else if (action.equals(
                    "PrintCustomerDistanceRatings")) {

                System.out.println("PrintCustomerDistanceRatings");
                program.printCustomerDistanceRatings(targetCustomerName);

            } else if (action.equals("AddCustomer")) {
                String name = lineScanner.next();
                int[] ratings = new int[10];

                for (int i = 0; i < 10; i++) {
                    ratings[i] = lineScanner.nextInt();
                }

                System.out.print("AddCustomer " + name);

                for (int i = 0; i < 10; i++) {
                    System.out.print(" " + ratings[i]);
                }

                System.out.println();

                program.insertCustomer(name, ratings);
            }

            lineScanner.close();
        }

        actionsScanner.close();
    }
}
