/*

  Author: Clarence Masuecos
  Email: mmasuecos2025@my.fit.edu
  Course: CSE 2010
  Section: 3 
  Description of this file: reads customer song ratings, calculates the similarity 
  distances between them, then recommends new songs based on 
  the closest preferences of the customer
  using heap-based priority queue

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.Locale;

public class HW4 {

    /*
      description of Customer class:
      Stores customers name and 
      their 10 song ratings 
      in array
    */
    static class Customer {
        String name;
        int[] r = new int[10];

        public Customer(String name, int[] ratings) {
            this.name = name;
            for (int i = 0; i < 10; i++) {
                this.r[i] = ratings[i];
            }
        }
    }

    /*
      Description PQEntry class:
      simple key-value 
      pairing for  Priority Queue
       key is  calculated distance,
        and  value is  Customer
    */
    static class PQEntry {
        double key;
        Customer value;

        public PQEntry(double key, Customer value) {
            this.key = key;
            this.value = value;
        }
    }

    /*
      Description  HeapPQ class:
      custom min-heap array built
       to manage closest customers
      includes initialize() method 
      to help with the bottom-up construction
    */
    static class HeapPQ {
        PQEntry[] heap = new PQEntry[100];
        int size = 0;

        // extra credit: a bottom-up heap construction
        public void initialize(PQEntry[] entryArray, int numEntries) {
            for (int i = 0; i < numEntries; i++) {
                heap[i] = entryArray[i];
            }
            size = numEntries;
            // start from  last internal node down to  root
            for (int i = (size / 2) - 1; i >= 0; i--) {
                downheap(i);
            }
        }

        public void insert(PQEntry e) {
            heap[size] = e;
            upheap(size);
            size++;
        }

        public PQEntry removeMin() {
            if (size == 0) return null;
            PQEntry min = heap[0];
            heap[0] = heap[size - 1]; 
            size--;
            downheap(0);
            return min;
        }

        public boolean isEmpty() {
            return size == 0;
        }

        private void upheap(int j) {
            while (j > 0) {
                int p = (j - 1) / 2;
                if (compare(heap[j], heap[p]) >= 0) break;
                swap(j, p);
                j = p;
            }
        }

        private void downheap(int j) {
            while (2 * j + 1 < size) {
                int left = 2 * j + 1;
                int smallChild = left;
                int right = 2 * j + 2;
                
                if (right < size && compare(heap[right], heap[left]) < 0) {
                    smallChild = right;
                }
                if (compare(heap[smallChild], heap[j]) >= 0) break;
                
                swap(j, smallChild);
                j = smallChild;
            }
        }

        private void swap(int i, int j) {
            PQEntry temp = heap[i];
            heap[i] = heap[j];
            heap[j] = temp;
        }

        // compare function gets
        //  distance first, then
        //  breaks ties alphabetically
        private int compare(PQEntry a, PQEntry b) {
            if (a.key < b.key) return -1;
            if (a.key > b.key) return 1;
            return a.value.name.compareTo(b.value.name); 
        }
    }

    // global array 
    // variables  in order to
    // track all the customers
    static Customer[] customers = new Customer[100];
    static int custCount = 0;
    static String targetName = "";

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Please provide data and action files.");
            return;
        }

        /* read initial ratings data file with Scanner */
        try {
            Scanner dataScan = new Scanner(new File(args[0]));
            if (dataScan.hasNextLine()) {
                targetName = dataScan.nextLine().trim();
            }

            while (dataScan.hasNextLine()) {
                String line = dataScan.nextLine().trim();
                if (line.isEmpty()) continue;
                
                String[] parts = line.split(" ");
                String name = parts[0];
                int[] rats = new int[10];
                for (int i = 0; i < 10; i++) {
                    rats[i] = Integer.parseInt(parts[i + 1]);
                }
                insertCustomer(name, rats);
            }
            dataScan.close();
        } catch (FileNotFoundException e) {
            System.out.println("Data file not found.");
            return;
        }

        /* 
        reads and executes the actions from second file
        */
        try {
            Scanner actionScan = new Scanner(new File(args[1]));
            while (actionScan.hasNextLine()) {
                String line = actionScan.nextLine().trim();
                if (line.isEmpty()) continue;
                
                String[] parts = line.split(" ");
                String action = parts[0];

                if (action.equals("AddCustomer")) {
                    System.out.println(line);
                    String name = parts[1];
                    int[] rats = new int[10];
                    for (int i = 0; i < 10; i++) {
                        rats[i] = Integer.parseInt(parts[i + 2]);
                    }
                    insertCustomer(name, rats);
                } 
                else if (action.equals("RecommendSongs")) {
                    System.out.print("RecommendSongs ");
                    recommendSongs();
                } 
                else if (action.equals("PrintCustomerDistanceRatings")) {
                    System.out.println("PrintCustomerDistanceRatings");
                    printRatings();
                }
            }
            actionScan.close();
        } catch (FileNotFoundException e) {
            System.out.println("Action file not found.");
        }
    }

    /*
      description insertCustomer:
      adds  customer to  array 
      while maintaining precise alphabetical 
      order by shifting larger names to right
    */
    public static void insertCustomer(String name, int[] r) {
        Customer c = new Customer(name, r);
        int i = custCount - 1;
        
        while (i >= 0 && customers[i].name.compareTo(name) > 0) {
            customers[i + 1] = customers[i];
            i--;
        }
        customers[i + 1] = c;
        custCount++;
    }

    /*
      description findCustomer function:
      uses Binary Search to find 
      a customer object by name, which uses
      O(log N) time
    */
    public static Customer findCustomer(String name) {
        int low = 0, high = custCount - 1;
        while (low <= high) {
            int mid = (low + high) / 2;
            int cmp = customers[mid].name.compareTo(name);
            
            if (cmp == 0) return customers[mid];
            else if (cmp < 0) low = mid + 1;
            else high = mid - 1;
        }
        return null;
    }

    /*
      description calculateDistance function:
      calculates mathematical distance between
      target and other customer 
     with mutually rated songs, then
     returns -1.0 if no songs  shared
    */
    public static double calculateDistance(Customer t, Customer c) {
        int count = 0;
        double diffSum = 0;
        
        for (int i = 0; i < 10; i++) {
            if (t.r[i] > 0 && c.r[i] > 0) {
                count++;
                diffSum += Math.abs(t.r[i] - c.r[i]);
            }
        }
        
        if (count == 0) return -1.0;
        return (1.0 / count) + (diffSum / count);
    }

    /*
      descriptino printRatings:
      formats + outputs  customer distance table
       making sure target 
      is printed first and 
      distances are rounded to 3 decimals
    */
    public static void printRatings() {
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
        
        Customer target = findCustomer(targetName);
        System.out.printf("%-9s%-11s%d  %d  %d  %d  %d  %d  %d  %d  %d  %d\n",
            "         ", target.name, 
            target.r[0], target.r[1], target.r[2], target.r[3], target.r[4],
            target.r[5], target.r[6], target.r[7], target.r[8], target.r[9]);
            
        for (int i = 0; i < custCount; i++) {
            Customer c = customers[i];
            if (c == target) continue;
            
            double dist = calculateDistance(target, c);
            String distStr = (dist < 0) ? "-----" : String.format(Locale.US, "%.3f", dist);
            
            System.out.printf("%-9s%-11s%d  %d  %d  %d  %d  %d  %d  %d  %d  %d\n",
                distStr, c.name, 
                c.r[0], c.r[1], c.r[2], c.r[3], c.r[4],
                c.r[5], c.r[6], c.r[7], c.r[8], c.r[9]);
        }
    }

    /*
      description  recommendSongs function:
      builds priority queue of distances, then
      pops  closest customers, and 
      checks for unrated songs with 
       rating >= 4 to recommend
    */
    public static void recommendSongs() {
        Customer target = findCustomer(targetName);
        
        // prepare for bottom-up heap initialization
        PQEntry[] tempArray = new PQEntry[100];
        int tempCount = 0;
        
        for (int i = 0; i < custCount; i++) {
            if (customers[i] == target) continue;
            double dist = calculateDistance(target, customers[i]);
            if (dist >= 0) {
                tempArray[tempCount++] = new PQEntry(dist, customers[i]);
            }
        }
        
        HeapPQ pq = new HeapPQ();
        pq.initialize(tempArray, tempCount); 
        
        boolean found = false;
        while (!pq.isEmpty()) {
            PQEntry min = pq.removeMin();
            Customer c = min.value;
            
            boolean hasRec = false;
            String recs = "";
            
            for (int s = 0; s < 10; s++) {
                if (target.r[s] == 0 && c.r[s] >= 4) {
                    hasRec = true;
                    recs += "song" + s + " " + c.r[s] + " ";
                }
            }
            
            if (hasRec) {
                System.out.println(c.name + " " + recs.trim());
                found = true;
                break;
            }
        }
        
        if (!found) {
            System.out.println("none");
        }
    }
}