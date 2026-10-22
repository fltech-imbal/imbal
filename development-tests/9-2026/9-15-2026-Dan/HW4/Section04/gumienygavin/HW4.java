/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
  This file recommends music for another customer based on another customer who has similar tastes.
  It collects and stores ratings of songs from each customer (along with the customer name itseslf)
  and finds the customer with the closest ratings. Then, it can recommend songs to a target customer
  who has not rated those songs yet. It takes two input file names as a command-line argument and
  reads each input line within each file. The first file is the list of customers with their song
  ratings (and who the target customer is) and the second file is the list of actions to be done
  (AddCustomer, RecommendSongs, PrintCustomerDistanceRatings).

 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

public class HW4
{
    // An array to store the list of customers in alphabetical order.
    private static Customer[] customers = new Customer[100];
    
    // Variable to keep track of the total number of customers.
    private static int numCustomers = 0;
    
    // Variable to store the target customer.
    private static String targetCust = "";
    
    /** Inserts a new customer into the array of customers. Maintains alphabetical order.
     * 
     *  @param name  the name of the customer
     *  @param ratings  the list of song ratings associated with the customer
     */
    public static void insertCustomer(String name, List<Integer> ratings) {
        Customer newCustomer = new Customer(name, ratings);  // Create new customer object
        
        // Shifts elements in the customer array to the right if name being checked comes
        // after the name being inserted.
        int i = numCustomers - 1;
        while (i >= 0 && customers[i].getCustomerName().compareTo(name) > 0) {
            customers[i + 1] = customers[i];  // Shifts current element once to right
            i--;
        }
        
        // Inserts the new customer in its correct place, number of customers increase.
        customers[i + 1] = newCustomer;
        numCustomers++;
    }
    
    /** Finds a customer in an array of customers given a name. Uses binary search.
     * 
     *  @param name  the name of the customer to be found
     *  @return  the object of the customer found or null if not found
     */
    public static Customer findCustomer(String name) {
        // Low and high variables to keep track of our range.
        int low = 0;
        int high = numCustomers - 1;
        
        // While low is less than or equal to high we calculate the mid variable and compare the mid.
        while (low <= high) {
            int mid = (low + high) / 2;  // The middle of the range
            
            // Determines whether the mid name comes alphabetically before or after the name we want
            // to find, otherwise the name to be found is the mid.
            int compareName = customers[mid].getCustomerName().compareTo(name);
            
            // If negative then not in first half.
            if (compareName < 0) {
                low = mid + 1;
            }
            // If positive then not in second half.
            else if (compareName > 0) {
                high = mid - 1;
            }
            // If neither then the customer to be found is the mid.
            else {
                return customers[mid]; // Found :D
            }
        }
        return null; // Not found :(
    }
    
    /** Calcluates the distance between two customers (usually the target and some other given customer).
     *  The distance is the difference in musical tastes between two customers.
     *  
     *  @param x  the target customer
     *  @param y  the other customer
     *  @return  the distance between the target and another customer or -1 if they have no shared songs.
     */
    public static double calculateDistance(Customer x, Customer y) {
        // Lists to store customer x and y's song ratings.
        List<Integer> songRatingsX = x.getRatings();
        List<Integer> songRatingsY = y.getRatings();
        
        // Keep track of the number of shared songs between customer x and y.
        int sharedSongs = 0;
        
        // Keep track of the sum of the difference between each song rating by both customers.
        int sumDiff = 0;
        
        // For each song rating in the list of song ratings we store that song rating and check if
        // they're greater than 0.
        for (int s = 0; s < 10; s++) {
            int ratingX = songRatingsX.get(s);   // Stores customer x's rating for song s.
            int ratingY = songRatingsY.get(s);   // Stores customer y's rating for song s.
            
            // If both ratings are greater than 0, they both share a song rating and we calculate the sum of the difference
            // between both ratings.
            if (ratingX > 0 && ratingY > 0) {
                sharedSongs++;
                sumDiff += Math.abs(ratingX - ratingY);
            }
        }
        
        // If there are no shared songs we cannot find a distance.
        if (sharedSongs == 0) {
            return -1;
        }
        
        // Otherwise return the distane of customer x and y.
        return (1.0 / sharedSongs) + ((1.0 / sharedSongs) * sumDiff);
    }
    
    /** Prints out the list of customers along with their distance from the target customer
     *  and their song ratings.
     * 
     */
    public static void printCustomerDistanceRatings() {
        // Header format
        System.out.println("PrintCustomerDistanceRatings");
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
        
        // Finds the target customer.
        Customer target = findCustomer(targetCust);
        if (target == null) return;     // No target found.
        
        // Target customer line
        formatLine("         ", targetCust, target.getRatings());
        
        // Rest of customers
        for (int i = 0; i < numCustomers; i++) {
            // If the current customer does not equal the target customer, print
            // out the current customer.
            if (!(customers[i].getCustomerName().equals(targetCust))) {
                double distance = calculateDistance(target, customers[i]); // Finds the distance
                String dist = "";
                
                // If no distance was found, print "-----".
                if (distance == -1) {
                    dist = "-----";
                }
                // If a distance was found, format so that it's displayed up to three decimal places.
                else {
                    dist = String.format("%.3f", distance);
                }
                
                // Call formatLine to print out the line in the correct format.
                formatLine(dist, customers[i].getCustomerName(), customers[i].getRatings());
            }
        }
    }
    
    /** Formats a line for PrintCustomerDistanceRatings so that the proper amout of spaces are displayed.
     * 
     *  @param dist  the customer's distance
     *  @param name  the name of the customer
     *  @param ratings  the list of ratings associated with the customer
     */
    public static void formatLine(String dist, String name, List<Integer> ratings) {
        // String builder to format the line.
        StringBuilder sb = new StringBuilder();
        sb.append(dist);    // Distance is added first
        
        // Distance column 1-9. Add spaces while the length of the line is less than 9.
        while (sb.length() < 9) {
            sb.append(" ");
        }
        sb.append(name); // Add name after distance column ends.
        
        // Customer name columns 10-20. Add spaces while the length of the line is less than 20.
        while (sb.length() < 20) {
            sb.append(" ");
        }
        
        // Song rating columns 21-23 (for song0), song1 starts at 24, etc...
        // For each song rating in the list of song ratings, add the song rating followed by two spaces.
        for (int s = 0; s < 10; s++) {
            sb.append(ratings.get(s)); //Adds the song rating
            // Adds two spaces after the song rating unless it is the last song rating has been added.
            if (s < 9) {
                sb.append("  ");
            }
        }
        
        // Prints out the fully formatted line.
        System.out.println(sb.toString());
    }
    
    /** Recommends songs that a closest customer gave the highest ratings to. The closest
     *  customer is the customer who has the smallest distance from the target customer.
     *  The recommended songs are for the target customer who has not yet rated those songs.
     * 
     */
    public static void recommendSongs() {
        Customer target = findCustomer(targetCust);     // Finds the target customer
        
        // If there is no target then we cannot recommend any songs.
        if (target == null) {
            System.out.println("RecommendSongs none");
            return;
        }
        
        // Creates a new heap to store a customer with a key (the distance).
        HeapPriorityQueue<Double, Customer> hpq = new HeapPriorityQueue<>();
        
        // Inserts each customer from the customer array into the heap if they have a valid distance.
        for (int i = 0; i < numCustomers; i++) {
            // Do not insert the target customer.
            if (customers[i] != target) {
                double distance = calculateDistance(target, customers[i]); // Calculates distance
                if (distance >= 0) {
                    hpq.insert(distance, customers[i]);
                }
            }
        }
        
        // While the heap is not empty, we check to see if the closest customer has
        // rated any songs that the target customer has not rated, then we recommend
        // if the closest customer liked the song.
        while (!hpq.isEmpty()) {
            // Removes and obtains the closest customer.
            Entry<Double, Customer> closest = hpq.removeMin();
            Customer closestCust = closest.getValue();
            
            // String builder to store the potential recommendations.
            StringBuilder recommendations = new StringBuilder();
            boolean foundSongs = false;         // Keeps track of whether we found a song to recommend.
            
            // Lists to store the song rating for the target and closest customer.
            List<Integer> songRatingsTarget = target.getRatings();
            List<Integer> songRatingsClosest = closestCust.getRatings();
            
            // For each song in the list of songs, we get the ratings and decided whether we should recommend the song.
            for (int s = 0; s < 10; s++) {
                int ratingTarget = songRatingsTarget.get(s);    // Stores target customers' song rating for song s.
                int ratingClosest = songRatingsClosest.get(s);  // Stores closest customers' song rating for song s.
                
                // If the target customer has not rated the current song AND the closest customer has
                // rated the song, then we found a song to recommend.
                if (ratingTarget == 0 && ratingClosest >= 4) {
                    foundSongs = true;
                    recommendations.append(" song").append(s).append(" ").append(ratingClosest); // Adds the song we want to recommend.
                }
            }
            
            // If we found a song we print out the result and return. :D
            if (foundSongs) {
                System.out.println("RecommendSongs " + closestCust.getCustomerName() + recommendations.toString());
                return;
            }
        }
        
        // If no songs were found. :(
        System.out.println("RecommendSongs none");
    }
    
    /** The main method (entry point) for the program.
     * 
     *  @param args  an array of Strings passed into the program (in this case a two String elements, the ratings input file and the actions input file.).
     *  @throws FileNotFoundException  if the attempt to access the file through args fails.
     */
    public static void main(String[] args)
    {   
        try {
            // Checks if there were any arguments passed into the program
            if (args.length < 2) {
                throw new FileNotFoundException("Incorrect file entry! Enter ratings file first followed by the actions file.");
            }
            
            // Scanner variable to read the customers and song ratings.
            Scanner scnrRtgs = new Scanner(new File(args[0]));
            
            // Read the first line and store the target customer.
            if (scnrRtgs.hasNextLine()) {
                String line = scnrRtgs.nextLine();
                if (line != null && !line.trim().isEmpty()) {
                    targetCust = line.trim();
                }
            }
            
            // Read each customer and their song ratings.
            while (scnrRtgs.hasNextLine()) {
                // Puts an entire line into a string variable
                String line = scnrRtgs.nextLine().trim();
                if (line.isEmpty()) continue;       // Nothing to read
                
                // Reads the text stored in each line
                Scanner lineScanner = new Scanner(line);
                
                // If the first words in a line exists, the word (the customer name)
                // is read and stored.
                if (lineScanner.hasNext()) {
                    // Stores the customer name
                    String customerName = lineScanner.next();
                    
                    // List to store song ratings for each customer;
                    List<Integer> customerSongRatings = new ArrayList<>();
                    
                    // While the line has integers, the integers (customer's song ratings) are read and stored.
                    while (lineScanner.hasNextInt()) {
                        int songRating = lineScanner.nextInt();
                        customerSongRatings.add(songRating);
                    }
                    
                    // Adds the new customer
                    insertCustomer(customerName, customerSongRatings);
                }
                lineScanner.close();
            }
            // Close first scanner.
            scnrRtgs.close();
            
            // Open second scanner and read the actions.
            Scanner scnrActions = new Scanner(new File(args[1]));
            
            // Reads each line from the action input file.
            while (scnrActions.hasNextLine()) {
                // Puts an entire line into a single variable.
                String line = scnrActions.nextLine();
                if (line.isEmpty()) continue;       // Nothing to read
                
                // Reads the text stored in each line.
                Scanner lineScanner = new Scanner(line);
                
                // If the first word in a line exists, the word (the action)
                // is read and stored.
                if (lineScanner.hasNext()) {
                    // Stores the current query.
                    String action = lineScanner.next();
                    
                    // If the action to be called is "AddCustomer," the parameter customer is read and stored along with its ratings.
                    // Then the method addCustomer is called.
                    if (action.equals("AddCustomer")) {
                        String customerName = lineScanner.next();
                        
                        // List to store song ratings for the customer;
                        List<Integer> songRatings = new ArrayList<>();
                        StringBuilder songs = new StringBuilder();
                        
                        // While the line has integers, the integers (the customer's song ratings) are read and stored.
                        while (lineScanner.hasNextInt()) {
                            int songRating = lineScanner.nextInt();
                            songs.append(songRating).append(" ");
                            songRatings.add(songRating);
                        }
                        
                        // Inserts the customer into the array (alphabetically).
                        insertCustomer(customerName, songRatings);
                        
                        // Outputs that the customer has been added.
                        System.out.println("AddCustomer " + customerName + " " + songs.toString().trim());
                    }
                    // If the action to be called is "RecommendSongs," call recommendSongs().
                    else if (action.equals("RecommendSongs")) {
                        recommendSongs();
                    }
                    // If the action to be called is "PrintCustomerDistanceRatings," call printCustomerDistanceRatings().
                    else if (action.equals("PrintCustomerDistanceRatings")) {
                        printCustomerDistanceRatings();
                    }
                }
                lineScanner.close();
            }
            // Close second scanner.
            scnrActions.close();
        }
        // An exception that catches if the file to be read is not found.
        catch (FileNotFoundException ex){
            System.err.println("Error: File not found. " + ex.getMessage());
        }
    }
}
