/*

  Author: Joseph Venech
  Email: jvenech2025@my.fit.edu
  Course: CSE2010
  Section: 1
  Description of this file: This program manages customer song ratings, calculates distances between customers
      based on their shared ratings, prints customer distance and ratings tables, and recommends songs using
      a priority queue.

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW4
{
    //Store customer's name and ratings
    public class Customer {
        int[] ratings;
        String name;
        
        //Constructor
        Customer(String name, int[] ratings) {
            this.name = name;
            this.ratings = ratings;
        }
    }
    
    //Stores customer's distance from target for priority queue ordering
    public class CustomerDistance implements Comparable<CustomerDistance> {
        double distance;
        String name;
        
        //Constructor
        public CustomerDistance(double distance, String name) {
            this.distance = distance;
            this.name = name;
        }
        
        //Orders customers by distance
        @Override
        public int compareTo(CustomerDistance other) {
            int result = Double.compare(distance, other.distance);
            
            if (result != 0) {
                return result;
            }
            
            return name.compareTo(other.name);
        }
    }
    
    //Array that stores the customers
    private Customer[] customers = new Customer[100];
    
    //Number of customers currently stored
    private int customerCount = 0;
    
    //Inserts customer into the array while maintaining order
    public void insertCustomer(String name, int[] ratings) {
        //Does not insert if array is full
        if (customerCount >= customers.length) {
            return;
        }

        Customer newCustomer = new Customer(name, ratings);
        
        //Start at last occupied position
        int i = customerCount - 1;
        
        //Shift customer with later names one position to the right
        while (i >= 0 && customers[i].name.compareTo(name) > 0) {
            customers[i + 1] = customers[i];
            i--;
        }

        //Insert new customer into correct position
        customers[i + 1] = newCustomer;
        customerCount++;
    }
    
    //Finds and returns customer by name using binary search
    public Customer findCustomer(String name) {
        //Binary Search
        int low = 0;
        int high = customerCount - 1;

        while (low <= high) {
            int mid = (low + high) / 2;

            int comparison = customers[mid].name.compareTo(name);

            if (comparison == 0) {
                // Customer found
                return customers[mid];
            } else if (comparison < 0) {
                // Search the right half
                low = mid + 1;
            } else {
                // Search the left half
                high = mid - 1;
            }
        }

        // Customer was not found
        return null;
    }

    //Calculates distance between two customers using shared ratings
    public double calculateDistance(Customer x, Customer y) {
        int commonSongs = 0;
        int totalDifference = 0;

        for (int i = 0; i < 10; i++) {
            // Only compare songs both customers rated
            if (x.ratings[i] != 0 && y.ratings[i] != 0) {
                commonSongs++;
                //Add the absolute difference between ratings
                totalDifference += Math.abs(x.ratings[i] - y.ratings[i]);
            }
        }

        // No common songs thus cannot be calculated
        if (commonSongs == 0) {
            return -1.0;
        }

        //Return distance using given formula
        return (1.0 / commonSongs) + ((double) totalDifference / commonSongs);
    }

    //Print target customer followed by rest of customers
    public void printCustomerDistanceRatings(String targetName) {
        Customer target = findCustomer(targetName);
        
        //No target customer
        if (target == null) {
            return;
        }

        //table header
        System.out.printf("%-9s %-8s", "distance", "customer");
        for (int i = 0; i < 10; i++) {
            System.out.printf(" s%d", i);
        }
        System.out.println();

        // Print target customer without a distance.
        System.out.printf("%-9s %-8s", "", target.name);
        for (int i = 0; i < 10; i++) {
            System.out.printf(" %2d", target.ratings[i]);
        }
        System.out.println();

        // Print remaining customers alphabetically.
        for (int i = 0; i < customerCount; i++) {
            Customer current = customers[i];

            if (current.name.equals(targetName)) {
                continue;
            }

            double distance = calculateDistance(target, current);

            if (distance < 0) {
                System.out.printf("%-9s ", "-----");
            } else {
                System.out.printf("%-9.3f ", distance);
            }

            System.out.printf("%-8s", current.name);

            for (int j = 0; j < 10; j++) {
                System.out.printf(" %2d", current.ratings[j]);
            }
            
            System.out.println();
        }
    }
    
    //Recommends songs that target hasn't rated
    public void recommendSongs(String targetName) {
        Customer target = findCustomer(targetName);

        //No target
        if (target == null) {
            System.out.println("none");
            return;
        }

        //Used to process closest customers first
        HeapPriorityQueue<CustomerDistance, String> queue = new HeapPriorityQueue<>();

        for (int i = 0; i < customerCount; i++) {
            Customer current = customers[i];

            //Don't compare target with themself
            if (current.name.equals(targetName)) {
                continue;
            }

            double distance = calculateDistance(target, current);
    
            //Add customers with distances
            if (distance >= 0) {
                queue.insert(new CustomerDistance(distance, current.name), current.name);
            }
        }
        
        //tracks songs recommended
        boolean[] recommended = new boolean[10];
        int count = 0;

        //Process customers from closest to farthest
        while (!queue.isEmpty() && count < 3) {
            Entry<CustomerDistance, String> entry = queue.removeMin();

            //Get customer with removed entry
            Customer current = findCustomer(entry.getValue());

            int previousCount = count;
            //Check each song for possible recommendation
            for (int i = 0; i < 10 && count < 3; i++) {
                // Recommend unrated songs with ratings above 3.
                if (target.ratings[i] == 0 && current.ratings[i] > 3 && !recommended[i]) {

                    if (count == 0) {
                        System.out.print(current.name);
                    }

                    System.out.print(" song" + i + " " + current.ratings[i]);
                
                    recommended[i] = true;
                    count++;
                }
            }
            
            if (count > previousCount) {
                break;
            }
        }

        //No songs were recommended
        if (count == 0) {
            System.out.print("none");
        }

        System.out.println();
    }

    
    public static void main(String[] args) throws FileNotFoundException   {
        HW4 program = new HW4();
        
        //Get input files
        String ratingsFile = args[0];
        String actionsFile = args[1];
        
        Scanner ratingsScanner = new Scanner(new File(ratingsFile));
        
        String targetCustomerName = "";
        
        //Read target customer
        if(ratingsScanner.hasNextLine()) {
            targetCustomerName = ratingsScanner.nextLine().trim();
        }
        
        
        //Read remaining customers
        while (ratingsScanner.hasNextLine()) {
            String line = ratingsScanner.nextLine().trim();
            
            //Skip empty lines
            if (line.length() == 0) {
                continue;
            }
            
            Scanner lineScanner = new Scanner(line);
            
            
            String name = lineScanner.next();
            int[] ratings = new int[10];
            
            //Get ratings
            for (int i = 0; i < 10; i++) {
                //Get rating and add to array
                ratings[i] = lineScanner.nextInt();
            }
            
            //Store customer
            program.insertCustomer(name, ratings);
           
            lineScanner.close();
        }
        
        ratingsScanner.close();
        
        Scanner actionsScanner = new Scanner(new File(actionsFile));
        
        //Read actions
        while (actionsScanner.hasNextLine()) {
            String line = actionsScanner.nextLine().trim();
            
            //Skip empty lines
            if (line.length() == 0) {
                continue;
            }

            Scanner lineScanner = new Scanner(line);

            String action = lineScanner.next();
            
            if (action.equals("RecommendSongs")) {
                System.out.print("RecommendSongs ");
                program.recommendSongs(targetCustomerName);
                
            } else if (action.equals("PrintCustomerDistanceRatings")) {
                System.out.println("PrintCustomerDistanceRatings");
                program.printCustomerDistanceRatings(targetCustomerName);
                
            } else if (action.equals("AddCustomer")) {
                //Read name
                String name = lineScanner.next();
                int[] ratings = new int[10];
            
                //Get ratings
                for (int i = 0; i < 10; i++) {
                    //Get rating and add to array
                    ratings[i] = lineScanner.nextInt();
                }
                
                //Format Output
                System.out.print("AddCustomer " + name);
                for (int i = 0; i < 10; i++) {
                    System.out.print(" " + ratings[i]);
                }
                System.out.println();
                
                //insert Customer
                program.insertCustomer(name, ratings);
                
            }
            
            lineScanner.close();
        }
        
    }

}
