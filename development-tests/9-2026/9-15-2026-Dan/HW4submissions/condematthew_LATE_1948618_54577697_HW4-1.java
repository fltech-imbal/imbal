/*

  Author: Matthew Conde
  Email: mconde2022@my.fit.edu
  Course:	CSE 2010
  Section:	2
  Description of this file: file takes an input of a target customer, and then up to 100 other customers
			    and organizes them in a priority queue ranking based on the distance in ratings
			    from the target customer, then outputs relevant actions
			    main purpose is to suggest unheard songs from closest customer that are rated high

 */
import java.util.ArrayList;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW4
{


	// create global variables for the list of customers and number of customers

    static Customer[] customers = new Customer[100];
    static int numCustomers = 0;

        // create customer class to hold each customers name and ratings
    static class Customer {
        String name;
        int[] ratings;

        public Customer(String name, int[] ratings) {
            this.name = name;
            this.ratings = ratings;
        }
    }


	// method to loop through customer list and insert customer alphabetically 

    public static void insertCustomer(String name, int[] ratings) {
        if (numCustomers >= 100) {
	    return;	
    	}

        Customer newCust = new Customer(name, ratings);
        int i = numCustomers - 1;
    
		// preserves order of customers by shifting to the right
        while (i >= 0 && customers[i].name.compareTo(name) > 0) {
            customers[i + 1] = customers[i];
            i--;
        }
    	
		// insert customer at appropriate spot, iterate the number of customers

        customers[i + 1] = newCust;
        numCustomers++;
    }

	// implement binary search to find a customer, if customer isnt found, return null

    public static Customer findCustomer(String name) {
        int a = 0;
        int b = numCustomers - 1;
    
        while (a <= b) {
            int c = a + (b - a) / 2;
            int status = customers[c].name.compareTo(name);
        
            if (status == 0) {
                return customers[c];
            } else if (status < 0) {
                a = c + 1;
            } else {
                b = c - 1;
            }
        }
        return null;
    }


	// method for calculating the rating distance between two customers, 
	// cycles through songs and sees how many in common and diference in rating between each
	// return rating value or return negative number (invalid distance, no songs in common)

    public static double calculateDistance(Customer x, Customer y) {

        int sCount = 0;			 // |S|, number of mutual songs
        double sumDifference = 0.0;	 // rating distance
    
	// cycle through the 10 songs per customer
        for (int i = 0; i < 10; i++) {
			
		// check to see if songs are rated, if so, calculate rating difference
            if (x.ratings[i] > 0 && y.ratings[i] > 0) {
                sCount++;
                sumDifference += Math.abs(x.ratings[i] - y.ratings[i]);
            }
         }
    	
		// if no mutual songs, return negative number (invalid distance)
        if (sCount == 0) {
            return -1.0; 
        }
	    
		// find and return actual difference
	double difference = (1.0 / sCount) + ((1.0 / sCount) * sumDifference);

        return difference;
    }

	/// method for recommending a new song

    public static void recommendSongs(HeapPriorityQueue<Double, String> pq, Customer targetCust) {
        boolean foundRecommendation = false;
    
	    // remove until we find a match or empty the queue
    	ArrayList<Entry<Double, String>> temp = new ArrayList<>();

        while (!pq.isEmpty() && !foundRecommendation) {
            Entry<Double, String> closestEntry = pq.removeMin();
	    temp.add(closestEntry);

            Customer closestCust = findCustomer(closestEntry.getValue());
        
		// if we can't find the customer, skip one iteration of the loop

            if (closestCust == null) continue;

            String recommendations = "";
        
            for (int i = 0; i < 10; i++) {

	            // recommend a song if closest customer rated at least 4 and targetCust hasn't rated

                if (closestCust.ratings[i] >= 4 && targetCust.ratings[i] == 0) {
                    recommendations += "song" + i +" " + closestCust.ratings[i] + " ";
                    foundRecommendation = true;
                }
            }
        
            if (foundRecommendation) {
                System.out.println("RecommendSongs " + closestCust.name + " " + recommendations.trim());
                return;
            }
        }
    	
        System.out.println("RecommendSongs none");
      
	// put all back in priority queue
        for (Entry<Double, String> entry : temp) {
            pq.insert(entry.getKey(), entry.getValue());
        }

    }


	/* print out target customer and respective song ratings, then loop through the rest of customers in	  */
	/* alphabetical order, printing respective song ratings. using java syntax for aligning prints to columns */


    public static void printCustomers(Customer targetCust) {
        System.out.println("PrintCustomerDistanceRatings");
    
		// print out header row, then target customer, loop through ratings	
	System.out.printf("%-9s%-11s", "distance", "customer");
	System.out.println("s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");

        System.out.printf("%-9s%-11s", "", targetCust.name);
        for (int i = 0; i < 10; i++) {
             System.out.printf("%-3d", targetCust.ratings[i]);
        }
        System.out.println();
    
    		// print the rest in alphabetical order
        for (int i = 0; i < numCustomers; i++) {
            Customer c = customers[i];
            if (c.name.equals(targetCust.name)) continue;
        
            double dist = calculateDistance(targetCust, c);
        
            if (dist == -1.0) {
                System.out.printf("%-9s%-11s", "-----", c.name);	 // if distance is invalid, print nothing
            } else {
                System.out.printf("%-9.3f%-11s", dist, c.name);
            }
        	
		// cycle through song ratings for a given customer

            for (int j = 0; j < c.ratings.length; j++) {
                int r = c.ratings[j];
                System.out.printf("%-3d", r);
            }

            System.out.println();
        }
    }


    public static void main(String[] args) throws FileNotFoundException {

	// work with first input file, 

        Scanner file1 = new Scanner(new File(args[0]));
        String targetCustName = file1.next(); // first line is target customer name
        Customer targetCust = null;	      // initialize target customer


	// cycle through following lines, each has a customer name and song ratings
	// add ratings to new array then create customer and insert to list

        while (file1.hasNext()) {
            String name = file1.next();
            int[] ratings = new int[10];
            for (int i = 0; i < 10; i++) {
                ratings[i] = file1.nextInt();
            }
            insertCustomer(name, ratings); 	// insert customer, alphabetically
        
            if (name.equals(targetCustName)) {	// if a customer name matches the target, pull target cust
                targetCust = findCustomer(name);
            }
        }

        file1.close();

        // initialize prioritiy queue heap, then insert customers after finding distance

        HeapPriorityQueue<Double, String> pq = new HeapPriorityQueue<>();
        for (int i = 0; i < numCustomers; i++) {
            if (!customers[i].name.equals(targetCustName)) {
                double dist = calculateDistance(targetCust, customers[i]);
                if (dist != -1.0) { 			// don't insert if invalid distance
                    pq.insert(dist, customers[i].name);
                }
            }
        }

	// start processing second file
	
        Scanner file2 = new Scanner(new File(args[1]));
        while (file2.hasNext()) {
            String action = file2.next();
        
		// pull the name and rating if adding customer
            if (action.equals("AddCustomer")) {
                String newName = file2.next();
                int[] newRatings = new int[10];
            
		//build string for output
                String outStr = "AddCustomer " + newName;
                for (int i = 0; i < 10; i++) {
                    newRatings[i] = file2.nextInt();
                    outStr += " " + newRatings[i];
                }
	
		// insert new customer
                insertCustomer(newName, newRatings);
                System.out.println(outStr);
            
		// find distance of new customer, add to the priority queue if valid
                double newDist = calculateDistance(targetCust, findCustomer(newName));
                if (newDist != -1.0) {
                    pq.insert(newDist, newName);
                }
            
		//recommend songs function call
            } else if (action.equals("RecommendSongs")) {
                recommendSongs(pq, targetCust);
            
	        // print customers function call
            } else if (action.equals("PrintCustomerDistanceRatings")) {

                printCustomers(targetCust);
            }
        }
        file2.close();
    }


}
