/*
Author: Olivia Cicerrella
Email: ocicerrella2025@fit.edu
Course: Algor and data Struc
Section: Section 1
Description of this file: this program is for the HW 4 assignment. to recomend music for 
 customers based on another customers with similar taste. compairing then rating from one to
 another seeing what is simlar to rec then songs based of the other person
*/
import java.io.File;
import java.util.Scanner;
import java.io.FileNotFoundException;

// main method for the class 
public class HW4 {
    // static elements to be so they can be used outside the class 
    private static Customer targetCustomer = null; // maintain memory context for target user
    private static final Customer[] customerArray = new Customer[100]; //arry to hold max threshold
    private static int customerCount = 0; // counter for how many recards are leaded to the database

    public static void main(String[] args) {
        // enfoce the peramatrers to make sure it works corectly 
        if (args.length < 2) {
            System.err.println("Error missing arguments.");
            return;
        }
        // assign input to the workspace file paths
        String initialRatingsFile = args[0];
        String actionsFile = args[1];

        //  ths part proccess the intial ratings file
        try (Scanner fileScanner = new Scanner(new File(initialRatingsFile))) {
            // ensure the the first line jhas the neame of target
            if (fileScanner.hasNextLine()) {
                String targetName = fileScanner.nextLine().trim();
                // Find or prepare target placeholder
                targetCustomer = new Customer(targetName, new int[10]); 
            }
            
            // loops throight the file until all data are gone throight
            while (fileScanner.hasNext()) {
                String name = fileScanner.next();
                int[] ratings = new int[10];
                // pop 10 ints to the block out of the file 
                for (int i = 0; i < 10; i++) {
                    ratings[i] = fileScanner.nextInt();
                }
                
                // If it matches the target rewrite 
                if (name.equals(targetCustomer.getName())) {
                    targetCustomer = new Customer(name, ratings);
                }
                insertCustomer(name, ratings);
            }
        } 
        // catches the file if it doesnt match and could no read the ratings file 
        catch (FileNotFoundException e) {
            System.err.println("Error could not read ratings file " + initialRatingsFile);
            return;
        }

        // this section gets the actions of the exicution file 
        try (Scanner actionScanner = new Scanner(new File(actionsFile))) {
            // reads thrpight the transaction logs step by step
            while (actionScanner.hasNext()) {
                String action = actionScanner.next();
                switch (action) {
                    case "AddCustomer":
                        // analize runtime info out of the command lines 
                        String newName = actionScanner.next();
                        int[] newRatings = new int[10];
                        // loops throughh the text lines to standard size
                        for (int i = 0; i < 10; i++) {
                            newRatings[i] = actionScanner.nextInt();
                        }
                        // inserts the new additions wile keeping aphabetical order 
                        insertCustomer(newName, newRatings);
                        
                        // prints the command back out to standerd
                        System.out.print("AddCustomer " + newName);
                        for (int r : newRatings) System.out.print(" " + r);
                        System.out.println();
                        break;

                    case "RecommendSongs":
                        //triggers tje sequemce of proccessing clacutalions 
                        executeRecommendSongs();
                        break;

                    case "PrintCustomerDistanceRatings":
                        // // triggers printing formation s 
                        printCustomerDistanceRatings();
                        break;
                }
            }
        } 
        // catches the errors so that noting back happens 
        //while running the program checking if the file has the needed info
        catch (FileNotFoundException e) {
            System.err.println("Error could not read actions file " + actionsFile);
        }
    }

    //  inserts the cusomers into an arracy wile keeping the apphabectical order
    private static void insertCustomer(String name, int[] ratings) {
        // prevent double entries 
        if (customerCount >= 100) return;
        
        // linear shift inserts to maintain the order of aplphbet
        int i = customerCount - 1;
        while (i >= 0 && customerArray[i].getName().compareTo(name) > 0) {
            customerArray[i + 1] = customerArray[i];
            i--;
        }
        // new profile objects placed to right locations 
        customerArray[i + 1] = new Customer(name, ratings);
        customerCount++; // increments stograge size to keep count
    }

    // gets teh distence calcutlated 
    private static Double calculateDistance(Customer x, Customer y) {
        int sharedCount = 0;
        int absDiffSum = 0;
        
        // loops over the 10 songs to get overlapigng and non zero ones 
        for (int i = 0; i < 10; i++) {
            if (x.getRatings()[i] > 0 && y.getRatings()[i] > 0) {
                sharedCount++;// keeps track of howmany rated
                absDiffSum += Math.abs(x.getRatings()[i] - y.getRatings()[i]);// computes 
            }
        }

        // returns null of no overlaping songs 
        if (sharedCount == 0) {
            return null; 
        }
    
        // returns the final caulation 
        return (1.0 / sharedCount) + ((1.0 / sharedCount) * absDiffSum);
    }

    // resoleves the song reccs using the heep prooritey queue
    private static void executeRecommendSongs() { 
        HeapPriorityQueue<DistanceKey, String> pq = new HeapPriorityQueue<>();

        // loops collections to run evals agwainst competeres 
        for (int i = 0; i < customerCount; i++) {
            Customer candidate = customerArray[i];
            // skips proccess if current row match the recommended profile
            if (candidate.getName().equals(targetCustomer.getName())) {
                continue; // skip evaluating self distance
            }

            Double distance = calculateDistance(targetCustomer, candidate);
            // insert records in to heap wehn valiid 
            if (distance != null) {
                pq.insert(new DistanceKey(distance, candidate.getName()), candidate.getName());
            }
        }

        // /loop through prioritized candidates getting recommendations
        while (!pq.isEmpty()) {
            Entry<DistanceKey, String> minEntry = pq.removeMin();
            String closestCustName = minEntry.getValue();

            // findscusomer records data matchin mame by searching the array
            Customer closestCust = null;
            for (int i = 0; i < customerCount; i++) {
                if (customerArray[i].getName().equals(closestCustName)) {
                    closestCust = customerArray[i];
                    break;
                }
            }
            // loop guard in case lookips return null 
            if (closestCust == null) continue;

            // preps buffer 
            StringBuilder recommendations = new StringBuilder();
            boolean hasRecommendation = false;

            // scans for reccomednation canidats 
            for (int i = 0; i < 10; i++) {
                int targetRating = targetCustomer.getRatings()[i];
                int closestRating = closestCust.getRatings()[i];

                // Song must be unrated by target and highly rated by closest peer
                if (targetRating == 0 && closestRating >= 4) {
                    hasRecommendation = true;
                    recommendations.append(" song").append(i).append(" ").append(closestRating);
                }
            }

            // capturing items parameters secutly 
            if (hasRecommendation) {
                System.out.println("RecommendSongs " + closestCustName + recommendations);
                return; // c resolution loop sequence
            }
        }

        // fallback default output boundary reached if queue empty without any matches
        System.out.println("RecommendSongs none");
    }


    // outoust distance in matching colums format 
    private static void printCustomerDistanceRatings() {
        System.out.println("PrintCustomerDistanceRatings");

        // render the current layout baseline reference info row
        // column 1: distance, column 10: customer, column 21: s0 ... Spaced sequentially by 3 spaces
        System.out.println("distance customer s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");

        // render target metrics data first at top
        System.out.printf("%-9s%-11s", "-----", targetCustomer.getName());
        for (int r : targetCustomer.getRatings()) {
            System.out.printf("%-3d", r);
        }
        System.out.println();

        // loops sorted registry arrays and print candidates
        for (int i = 0; i < customerCount; i++) {
            Customer current = customerArray[i];
            if (current.getName().equals(targetCustomer.getName())) {
                continue; // Do not duplicat target inside listings
            }

            Double dist = calculateDistance(targetCustomer, current);
            // foramts calulated distance valuse 
            String distanceString = (dist == null) ? "-----" : String.format("%.3f", dist);

            //prins formated rows tracking colums offsets 
            System.out.printf("%-9s%-11s", distanceString, current.getName());
            for (int r : current.getRatings()) {
                System.out.printf("%-3d", r);
            }
            System.out.println();
        }
    }
}

class DistanceKey implements Comparable<DistanceKey> {
    private final double distance;
    private final String customerName;

    // constructors for paring distance calcutlations 
    public DistanceKey(double distance, String customerName) {
        this.distance = distance;
        this.customerName = customerName;
    }

    // standar accessors to retreve datad for both
    public double getDistance() { return distance; }
    public String getCustomerName() { return customerName; }

    @Override
    public int compareTo(DistanceKey other) {
        // eval numerical distance difference 
        if (Math.abs(this.distance - other.distance) < 1e-9) {
            return this.customerName.compareTo(other.customerName); // alphabetical
        }
        // if distance are diff sort in ascending order
        return Double.compare(this.distance, other.distance);
    }
}
// customer reps 
class Customer implements Comparable<Customer> {
    private final String name;
    private final int[] ratings; // size 10

    // constructors for this section
    public Customer(String name, int[] ratings) {
        this.name = name;
        this.ratings = ratings.clone();
    }

    // gets the name and rating for both
    public String getName() { return name; }
    public int[] getRatings() { return ratings; }

    @Override
    public int compareTo(Customer other) {
        // auto compares names 
        return this.name.compareTo(other.name);
    }
}