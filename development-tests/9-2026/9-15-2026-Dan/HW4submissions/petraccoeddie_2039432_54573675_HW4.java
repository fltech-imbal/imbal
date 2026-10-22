/**
 * Eddie Petracco III
 * epetracco2025@my.fit.edu
 * CSE 2010
 * 9:30 am Lab Section
 * Using a priority queue, recommends songs to a certain target customer based on closest customer data and their respective song ratings. 
 * This program can also add new customers along the way and also output recommended songs based on the closest target customer.
 */

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;

public class HW4
{
     
    // Represents a customer with their name and 10 song ratings. 
     
    public static class Customer {
        private String name;
        private int[] ratings = new int[10];
        
        // simple customer constructor
        public Customer(String name, int[] ratings) {
            this.name = name;
            System.arraycopy(ratings, 0, this.ratings, 0, 10);
        }

        // the getters 
        public String getName() {
            return this.name;
        }
        
        public int[] getRatings() {
            return this.ratings;
        }
    }
    
    // Global arrays and variables for managing our customer database 
    private static Customer[] customers = new Customer[100];
    private static int numCustomers = 0;
    private static Customer targetCustomerObj = null;
    private static String targetCustomerName = "";

    
    // Inserts the customer in the array and keeps it alphabetical using insertion sort logic.
     
    public static void insertCustomer(String name, int[] ratings) {
        Customer newCust = new Customer(name, ratings);
        int i = numCustomers - 1;
        
        // Shift elements to the right to maintain alphabetical order. This is similar to the alphabeticallized from HW3
        while (i >= 0 && customers[i].getName().compareTo(name) > 0) {
            customers[i + 1] = customers[i];
            i--;
        }
        customers[i + 1] = newCust;
        numCustomers++;
        
        // Update the target customer reference if this is the target
        if (name.equals(targetCustomerName)) {
            targetCustomerObj = newCust;
        }
    }
    
    
    // Uses binary search to find a customer by name in the sorted array.
     
    public static Customer findCustomer(String name) {
        int left = 0;
        int right = numCustomers - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int cmp = customers[mid].getName().compareTo(name);
            if (cmp == 0) {
                return customers[mid];
            } else if (cmp < 0) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return null; 
    }

    
     // Calculates distance between the target customer and another customer using the formula from HW4 pdf
     // Returns -1.0 if distance cannot be calculated (no shared rated songs).
     
    public static double calculateDistance(Customer target, Customer other) {
        int sharedCount = 0;
        double sumDifferences = 0;
        
        int[] tRatings = target.getRatings();
        int[] oRatings = other.getRatings();
        
        for (int i = 0; i < 10; i++) {
            if (tRatings[i] > 0 && oRatings[i] > 0) {
                sharedCount++;
                sumDifferences += Math.abs(tRatings[i] - oRatings[i]);
            }
        }
        
        if (sharedCount == 0) {
            return -1.0;
        }
        
        return (1.0 / sharedCount) + (sumDifferences / sharedCount);
    }

    
     // PQEntry acts as our Key-Value pair (distance, customerName). This is from the sample java files.
     
    public static class PQEntry {
        double distance;
        String customerName;
        
        public PQEntry(double distance, String customerName) {
            this.distance = distance;
            this.customerName = customerName;
        }
    }

    
     // Array-based Heap Priority Queue for customers based on distance.
     
    public static class MinHeap {
        private PQEntry[] heap = new PQEntry[100];
        private int size = 0;
        
        public boolean isEmpty() { return size == 0; }
        
        // method to compare entries
        private int compare(PQEntry a, PQEntry b) {
            if (a.distance < b.distance) return -1;
            if (a.distance > b.distance) return 1;
            // Break ties alphabetically
            return a.customerName.compareTo(b.customerName);
        }
        
        // method for swapping entires in heap
        private void swap(int i, int j) {
            PQEntry temp = heap[i];
            heap[i] = heap[j];
            heap[j] = temp;
        }
        
        // method for inserting a new customer
        public void insert(double distance, String customerName) {
            heap[size] = new PQEntry(distance, customerName);
            upheap(size);
            size++;
        }
        
        // method for removing the minimum in the heap 
        public PQEntry removeMin() {
            if (isEmpty()) return null;
            PQEntry min = heap[0];
            swap(0, size - 1);
            size--;
            downheap(0);
            return min;
        }
        
        // upheap method
        private void upheap(int j) {
            while (j > 0) {
                int p = (j - 1) / 2;
                if (compare(heap[j], heap[p]) >= 0) break;
                swap(j, p);
                j = p;
            }
        }
        
        // downheap method
        private void downheap(int j) {
            while (2 * j + 1 < size) {
                int leftIndex = 2 * j + 1;
                int smallChildIndex = leftIndex;
                int rightIndex = 2 * j + 2;
                if (rightIndex < size && compare(heap[rightIndex], heap[leftIndex]) < 0) {
                    smallChildIndex = rightIndex;
                }
                if (compare(heap[smallChildIndex], heap[j]) >= 0) break;
                swap(j, smallChildIndex);
                j = smallChildIndex;
            }
        }
    }

    public static void main(String[] args) {
        
        
         // Validate command-line arguments to ensure both the ratings data file 
         // and the query actions file are provided before proceeding. 
         
        if (args.length < 2) {
            System.err.println("Please provide both data and query file arguments.");
            return;
        }
        
        // Store the filenames from the command-line arguments into variables 
        String fileRatings = args[0];
        String fileActions = args[1];
        
         
        
         // Attempt to open the ratings file. The first entry represents the 
         // target customer whose recommendations we will eventually build.
         
        try (Scanner fileScanner = new Scanner(new File(fileRatings))) {
            targetCustomerName = fileScanner.next();
            
            
             // Loop through the rest of the ratings file. For each customer, 
             // grab their name and their 10 integer song ratings, then insert 
             // them into our sorted customer database.
             
            while (fileScanner.hasNext()) {
                String name = fileScanner.next();
                int[] ratings = new int[10];
                for (int i = 0; i < 10; i++) {
                    ratings[i] = fileScanner.nextInt();      
                } 
                insertCustomer(name, ratings);
            }  
        } catch (FileNotFoundException e) {
            System.err.println("File name: " + fileRatings + " was not found!"); 
            return;
        }
        
        
         
         // Attempt to open the actions file and process it line by line. 
         // We read the keyword to determine which operation to perform next.
         
        try (Scanner fileScanner = new Scanner(new File(fileActions))) {
            while (fileScanner.hasNext()) {
                String keyword = fileScanner.next();
                
                if (keyword.equals("PrintCustomerDistanceRatings")) {
                    
                     // Formating for desired output                    
                    System.out.println("PrintCustomerDistanceRatings");
                    System.out.printf("%-9s%-11s%s %s %s %s %s %s %s %s %s %s\n", 
                                      "distance", "customer", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9");
                     
                    System.out.printf("         %-11s", targetCustomerObj.getName());
                    int[] tr = targetCustomerObj.getRatings();
                    // iteraritng to print each song out sequentially for a given customer
                    
                    for (int i = 0; i < 9; i++) System.out.printf("%d  ", tr[i]);
                    System.out.printf("%d\n", tr[9]);
                    
                    // Loop through all other customers in our alphabetical database. 
                     
                    for (int i = 0; i < numCustomers; i++) {
                        Customer c = customers[i];
                        if (c.getName().equals(targetCustomerName)) continue;
                        
                    
                         // Calculate distance. If negative, there are no shared songs, 
                         // thus follow sample output files for format
                         
                        double dist = calculateDistance(targetCustomerObj, c);
                        if (dist < 0) {
                            System.out.printf("-----    %-11s", c.getName());
                        } else {
                            System.out.printf("%-9.3f%-11s", dist, c.getName());
                        }
                        
                        // Print out this customer's 10 song ratings.
                        int[] cr = c.getRatings();
                        for (int j = 0; j < 9; j++) System.out.printf("%d  ", cr[j]);
                        System.out.printf("%d\n", cr[9]);
                    }
                }
                else if (keyword.equals("RecommendSongs")) {
                
                     // Initialize an empty MinHeap for our priority queue.
                     // find the distance for all valid customers and insert 
                     // them into the heap so the closest customer is at the top.
                     
                    MinHeap heap = new MinHeap();
                    
                    for (int i = 0; i < numCustomers; i++) {
                        Customer c = customers[i];
                        if (!c.getName().equals(targetCustomerName)) {
                            double dist = calculateDistance(targetCustomerObj, c);
                            if (dist >= 0) {
                                heap.insert(dist, c.getName());
                            }
                        }
                    }
                    
                    
                     // Continuously remove the minimum distance customer from the heap. 
                     // check if they have highly rated songs that the target hasn't read already
                    
                    boolean foundRecommendation = false;
                    while (!heap.isEmpty() && !foundRecommendation) {
                        PQEntry closest = heap.removeMin();
                        Customer closestCust = findCustomer(closest.customerName);
                        
                        StringBuilder recommended = new StringBuilder();
                        int[] tr = targetCustomerObj.getRatings();
                        int[] cr = closestCust.getRatings();
                        
                        // Loop through the 10 songs comparing target and closest customer ratings 
                        for (int i = 0; i < 10; i++) {
                            if (tr[i] == 0 && cr[i] >= 4) {
                                recommended.append(" song").append(i).append(" ").append(cr[i]);
                                foundRecommendation = true;
                            }
                        }
                        
                        // Print out the recommendation if one was discovered 
                        if (foundRecommendation) {
                            System.out.println("RecommendSongs " + closestCust.getName() + recommended.toString());
                        }
                    }
                    
                    // If the heap empties with no valid recommendations found
                    if (!foundRecommendation) {
                        System.out.println("RecommendSongs none");
                    }
                }
                else if (keyword.equals("AddCustomer")) {
                    
                    // Read the new customer's name and their 10 song ratings and print respective action
                     
                    String name = fileScanner.next();
                    int[] ratings = new int[10];
                    System.out.print("AddCustomer " + name);
                    for (int i = 0; i < 10; i++) {
                        ratings[i] = fileScanner.nextInt();
                        System.out.print(" " + ratings[i]);
                    }
                    System.out.println();
                    insertCustomer(name, ratings);
                }
            }
        } catch (FileNotFoundException e) {
            System.err.println("File name: " + fileActions + " was not found!"); 
        }
    }
}