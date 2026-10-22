/*
   Author: Alexander Williams
   Email: alexanderwil2025@my.fit.edu
   Course: CSE 2010
   Section: 1
   Description: Recommends music to customers.
*/

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class HW4 {

    private static final int MAX_CUSTOMERS = 100;
    private static final int NUM_SONGS = 10;
    private static Customer[] customers = new Customer[MAX_CUSTOMERS];
    private static int customerCount = 0;
    private static String targetCustName;

   //stores customers name and song ratings
    private static class Customer {
        String name;
        int[] ratings;

        Customer(String name, int[] ratings) {
            this.name = name;
            this.ratings = ratings;
           }
     }

    //organize customers by distance
    private static class CustomerPriorityQueue extends HeapPriorityQueue<Double, String> {
        protected int compare(Entry<Double, String> first, Entry<Double, String> second) {
            int result = Double.compare(first.getKey(), second.getKey());
            if (result != 0) {
                return result;
               }
            return first.getValue().compareTo(second.getValue());
         }

        public Entry<Double, String> getMin() {
            return min();
         } 

        public boolean isFull() {
            return size() >= MAX_CUSTOMERS;
       }
      }
   
    //finds customer by name
    private static Customer findCustomer(String name) {
        int low = 0;
        int high = customerCount - 1;

        while (low <= high) {
            int middle = (low + high) / 2;
            int comparison = customers[middle].name.compareTo(name);
            if (comparison == 0) {
                return customers[middle];
            } else if (comparison < 0) {
                low = middle + 1;
            } else {
                high = middle - 1;
              }
             }
           return null;
     }
    //adds customers in alphabetical order	
    private static void insertCustomer(String name, int[] ratings) {
        Customer existing = findCustomer(name);
        if (existing != null) {
            existing.ratings = ratings;
            return;
         }
        if (customerCount >= MAX_CUSTOMERS) {
            return;
         }

        int index = customerCount;
        while (index > 0 && customers[index - 1].name.compareTo(name) > 0) {
            customers[index] = customers[index - 1];
            index--;
       }
        customers[index] = new Customer(name, ratings);
        customerCount++;
     }

    private static double distance(Customer targetCust, Customer otherCust) {
        int sharedSongs = 0;
        int totalDifference = 0;
        for (int song = 0; song < NUM_SONGS; song++) {
            if (targetCust.ratings[song] != 0 && otherCust.ratings[song] != 0) {
                sharedSongs++;
                totalDifference += Math.abs(targetCust.ratings[song] - otherCust.ratings[song]);
              }
          }
        if (sharedSongs == 0) {
            return Double.POSITIVE_INFINITY;
      }
        return (1.0 + totalDifference) / sharedSongs;
    }
    //prints customer distance, name, and rating
    private static void printCustomerRow(Customer customer, Customer targetCust) {
        String distanceText = "";
        if (customer != targetCust) {
            double d = distance(targetCust, customer);
            if (Double.isInfinite(d)) {
                distanceText = "-----";
              } else {
                distanceText = String.format("%.3f", d);
             }
          }

        System.out.printf("%-9s%-11s", distanceText, customer.name);
        for (int song = 0; song < NUM_SONGS; song++) {
            if (song > 0) {
                System.out.print("  ");
            }
            System.out.print(customer.ratings[song]);
        }
        System.out.println();
      }

    private static void printCustomerDistanceRatings() {
        System.out.println("PrintCustomerDistanceRatings");
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");

        Customer targetCust = findCustomer(targetCustName);
        if (targetCust == null) {
            return;
     }
        printCustomerRow(targetCust, targetCust);
        for (int i = 0; i < customerCount; i++) {
            if (customers[i] != targetCust) {
                printCustomerRow(customers[i], targetCust);
           }
      }
    }

   //finds closest customer and recommends songs y

    private static void recommendSongs() {
        Customer targetCust = findCustomer(targetCustName);
        if (targetCust == null) {
            System.out.println("RecommendSongs none");
            return;
           }
    
        CustomerPriorityQueue queue = new CustomerPriorityQueue();
        for (int i = 0; i < customerCount; i++) {
            Customer otherCust = customers[i];
            if (otherCust != targetCust && !queue.isFull()) {
                double d = distance(targetCust, otherCust);
                if (!Double.isInfinite(d)) {
                    queue.insert(d, otherCust.name);
                }
               }
        }

        while (!queue.isEmpty()) {
            Entry<Double, String> closestEntry = queue.removeMin();
            Customer closestCust = findCustomer(closestEntry.getValue());
            String recommendations = "";
            for (int song = 0; song < NUM_SONGS; song++) {
                if (targetCust.ratings[song] == 0 && closestCust.ratings[song] >= 4) {
                    recommendations += " song" + song + " " + closestCust.ratings[song];
                }
            }
            if (!recommendations.isEmpty()) {
                System.out.println("RecommendSongs " + closestCust.name + recommendations);
                  return;
            }
        }
        System.out.println("RecommendSongs none");
    }

    private static int[] readRatings(String[] parts, int firstRating) {
        int[] ratings = new int[NUM_SONGS];
        for (int i = 0; i < NUM_SONGS; i++) {
            ratings[i] = Integer.parseInt(parts[firstRating + i]);
        }
        return ratings;
    }

    private static void loadCustomers(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        targetCustName = reader.readLine().trim();
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty()) {
                String[] parts = line.split("\\s+");
                insertCustomer(parts[0], readRatings(parts, 1));
            }
          }
        reader.close();
    }

    private static void performActions(String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty()) {
                String[] parts = line.split("\\s+");
                if (parts[0].equals("AddCustomer")) {
                    insertCustomer(parts[1], readRatings(parts, 2));
                    System.out.println(line);
                } else if (parts[0].equals("RecommendSongs")) {
                    recommendSongs();
                } else if (parts[0].equals("PrintCustomerDistanceRatings")) {
                    printCustomerDistanceRatings();
                  }
          }
        }
        reader.close();
      }

    public static void main(String[] args) throws IOException {
        if (args.length != 2) {
            System.err.println("Usage: java HW4 <ratings-file> <actions-file>");
            return;
           }
        loadCustomers(args[0]);
        performActions(args[1]);
    }
}
