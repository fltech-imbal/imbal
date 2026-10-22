/*
    Author: Sean Delate
    Email: sdelate2025@my.fit.edu
    Course: CSE2010
    Section: 01
    Description of this file: Attempts to find song recomedations for a target person
    by using other peoples reviews and looking at their simularity to the targets.
    Then the person that has the most simaler tastes is looked at and their favorate songs 
    are reccomended to the target.

*/
import java.io.File; // Needed to read files
import java.io.FileNotFoundException; // Needed to handle errors
import java.util.ArrayList; // Needed to read files
import java.util.Arrays; // Needed to handle errors
import java.util.Collections;
import java.util.Comparator;
import java.util.List; // Needed to use as a changable array for inputs
import java.util.Scanner;
public class HW4<E extends Comparable<E>>
{
    // for the list holding reviews
    public class Customer {

        private String name; // holds customer name

        private int[] ratings; // holds ratings of customer

        public Customer() { //Defult constructor
            this.name = null;
            this.ratings = null;
        }

        public Customer(String name, int[] ratings) { //Cunstructor
            this.name = name;
            this.ratings = ratings;
        }

        public String name() { //gets name of customer
            return name;
        }

        public int[] ratings() { //gets ratings of customer
            return ratings;
        }
    }

    public HeapPriorityQueue<Double, String> customerAndCloseness = new HeapPriorityQueue<>(); //List of the customer names and difference from target reviews

    public List<Customer> Customers = new ArrayList<>(); // List of all customer names and reviews

    public Customer target = new Customer(); //Used as the target for the reccomendations

    // Sets the targets name for use in the add all customers section in assigning the targets reviews
    public void setTarget(String name) {
        target.name = name;
    }

    //Adds all of the customers to the list and heap using for loops.
    public void addAllCustomers(List<String[]> CustomersData) {
        ArrayList<Entry<Double,String>> heapEntries = new ArrayList<>();
        int numEntries = 0;
        for (String[] strA : CustomersData) { //Adds the data to customer objects
            int[] ratings = new int[10];
            for (int i = 1; i < strA.length; i++) {
                ratings[i-1] = Integer.parseInt(strA[i]);
            }
            Customer c = new Customer(strA[0], ratings);
            if (!c.name().equals(target.name)) {
                Customers.add(c);
            } else {
                target.ratings = c.ratings;
            }
        }
        Collections.sort(Customers, Comparator.comparing(Customer::name));
        for (Customer c: Customers) { //looks through all customers and adds them to the heap entries list to be heapified later
            Entry<Double, String> newEntry = new AbstractPriorityQueue.PQEntry<>(distance(target, c),c.name());
            heapEntries.add(newEntry);
            numEntries++;
        }
        customerAndCloseness.initialize(heapEntries,numEntries);
        
    }

    //inserts a customer into both the list and heap
    public void insertCustomer(String info) {
        List<String> infoList = Arrays.asList(info.split("\\s+"));
        int[] scores = new int[10];
        for (int i = 1; i < infoList.size(); i++) {
            scores[i-1] = Integer.parseInt(infoList.get(i));
        }
        Customer newCustomer = new Customer(infoList.get(0), scores);
        Customers.add(newCustomer);
        Collections.sort(Customers, Comparator.comparing(Customer::name));
        customerAndCloseness.insert(distance(newCustomer, target), newCustomer.name());
    }

    //uses the binary search on the customer list to find the customer for a name
    public Customer findCustomer(String name) {
        return binarySearch(0, Customers.size()-1, name);
    }

    //Basic binary search
    public Customer binarySearch(int l, int r, String name) {
        while (l <= r) {
            int mid = l + (r - l) / 2;

            if (Customers.get(mid).name().compareTo(name) == 0) {
                return Customers.get(mid);
            }
            if (Customers.get(mid).name().compareTo(name) > 0) {
                r = mid - 1;
            } else {
                l = mid + 1;
            }
        }
        return null;
    }

    public Double distance(Customer x, Customer y) {
        double bothRated = 0;
        double sumOfDifferences = 0;
        for (int i = 0; i < 10; i++) {
            if ((x.ratings()[i] != 0) && (y.ratings()[i] != 0)) {
                bothRated++;
                sumOfDifferences += Math.abs(x.ratings()[i] - y.ratings()[i]);
            }
        }
        if (bothRated == 0) {
            return 99999999.0;
        }
        return (1.0 / bothRated) + ((1.0 / bothRated) * (sumOfDifferences));
    }

    //Prints the customer ratings in a specified format by using the dstance and printcustomer methods
    public void printCustomerDistanceRatings() {
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
        System.out.print("         ");
        printCustomer(target);
        ArrayList<Entry<Double,String>> newEntries = new ArrayList<>(customerAndCloseness.heap);
        Collections.sort(newEntries, Comparator.comparing(Entry::getValue));
        for (Entry<Double, String> entry : newEntries) { //This section will print all customers other than target
            String distance = "-----";
            // if the difference is greater that 10000 which should only happen if they dont have any of the same reviews
            // this section should catch that.
            if (entry.getKey() < 10000) { 
                distance = "" + entry.getKey();
                if (distance.length() > 6) {
                    distance = distance.substring(0, 6);
                    if (distance.substring(5,6).compareTo("5") >= 0) {
                        distance = "" + (Double.parseDouble(distance.substring(0,5)) + 0.001);
                    } else {
                        distance = distance.substring(0,5);
                    }
                } else {
                    for (int i = distance.length(); i < 5; i++) {
                        distance += "0";
                    }
                }
            }
            String customerName = entry.getValue();
            Customer c = findCustomer(customerName);
            System.out.print(distance + "    ");
            printCustomer(c);
        }
    }

    // Prints the customer info ina way that works with the formatting of print customer distance rating
    public void printCustomer(Customer c) { //Starts from column 10
        int nameLength = c.name().length();
        System.out.print(c.name());
        for (int i = nameLength; i < 11; i++) {
            System.out.print(" ");
        }
        for (int i = 0; i < c.ratings().length-1; i++) {
            System.out.print(c.ratings()[i] + "  ");
        }
        System.out.println(c.ratings()[c.ratings().length - 1]);
    }

    //Checks for the closest customer in taste to the target and then uses a loop to reccommend songs to them
    public void recomendedSongs() {
        HeapPriorityQueue<Double, String> current = new HeapPriorityQueue<>();
        ArrayList<Entry<Double, String>> copiedEntries = new ArrayList<>(customerAndCloseness.heap);
        current.initialize(copiedEntries, copiedEntries.size());
        boolean found = false;
    
        while (!current.isEmpty() && current.min().getKey() <= 10000) {
            Entry<Double, String> closest = current.removeMin();
    
            // Makes so ties are given to alphabetical name
            while (!current.isEmpty() && current.min().getKey().equals(closest.getKey())) {
                Entry<Double, String> test = current.removeMin();
                if (test.getValue().compareTo(closest.getValue()) < 0) {
                    closest = test;
                }
            }
    
            Customer closestCustomer = findCustomer(closest.getValue());
            
            //Gets eligable songs
            List<String> recs = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                if (closestCustomer.ratings()[i] >= 4 && target.ratings()[i] == 0) {
                    recs.add("song" + i + " " + closestCustomer.ratings()[i]);
                }
            }
    
            // if they have songs prints otherwise tries next customer
            if (!recs.isEmpty()) {
                System.out.print(closestCustomer.name());
                for (String rec : recs) {
                    System.out.print(" " + rec);
                }
                System.out.println();
                found = true;
                break;
            }
        }
    
        if (!found) {
            System.out.println("none");
        }
    }

    public static void main(String[] args)
    {
        HW4<String> systemRequests = new HW4<>(); // Used to run methods

		File data = new File(args[0]); // Takes the command line arguement for the file and then uses it to make a new file object to be read later

        File input = new File(args[1]); // Takes the command line arguement for the file and then uses it to make a new file object to be read later

        List<String[]> CustomersData = new ArrayList<>(); // Stores all customers and their reviews

        String targetName = null; //Gets the name for the target customer

        // A try catch is needed to handle if the file with the chat commands cannot be found
		try (Scanner scan = new Scanner(data)) {
			// Looks through the file line by line
            targetName = scan.nextLine().trim(); //Trim is probably not neccesary but it catches formatting errors
            //Sets the target as the first line

            systemRequests.setTarget(targetName); //Sets the target customer before continuing

            //Increments through every line
            while (scan.hasNextLine()) {
                String treeStart = scan.nextLine().trim();
                //If the line is empty continues
                if (treeStart.isEmpty()) {
                    continue;
                }

                String[] CustomerData = treeStart.split("\\s+"); //Splits line into seperate strings
                
                CustomersData.add(CustomerData);


            }
		} catch (FileNotFoundException e) {
			System.out.println("Error: file not found");
			e.printStackTrace();
		}

        systemRequests.addAllCustomers(CustomersData); // Adds all of the customers to the databases


        // A try catch is needed to handle if the file with the chat commands cannot be found
		try (Scanner scan = new Scanner(input)) {
			// Looks through the file line by line
			while (scan.hasNextLine()) {
				// Reads the full line of text as a string input and then splits it into an array by spaces
				String info = scan.nextLine();
				String[] instruction = info.split("\\s");
					// Looks at the first element of the string array and depending on the command excecutes a method
                    // usually prints names and instructions before the instruction results
					switch (instruction[0]) {
						case "AddCustomer":
                            System.out.print(info);
                            String customerInfo = info.substring(instruction[0].length()).trim();
                            systemRequests.insertCustomer(customerInfo);
                            System.out.println();
                            break;
						case "RecommendSongs":
							System.out.print(instruction[0] + " ");
							systemRequests.recomendedSongs();
							break;
						case "PrintCustomerDistanceRatings":
							System.out.println(instruction[0]);
							systemRequests.printCustomerDistanceRatings();
							break;
						default:
							break;
					}
			}
		} catch (FileNotFoundException e) {
			System.out.println("Error: file not found");
			e.printStackTrace();
		}
    }
}
