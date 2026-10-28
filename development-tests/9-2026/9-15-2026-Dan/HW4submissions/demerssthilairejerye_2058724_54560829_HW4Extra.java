/*
Author: Jerye Demers-St.Hilaire
Email: jdemerssthil2025@my.fit.edu
Course: CSE 2010
Section: 1
Description of this file: Storing customers and their ratings, comparing customers ratings to each other, outputting customers in aplabetical order
 (Extra Credit: added a method to initialize the priority queue using bottom-up heap construction)
*/

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;

public class HW4Extra {

 // Creating an array to store up to 100 customers 
 static Customer[] customer = new Customer[100];
 
 // Keeping track of total customers
 static int numCustomers = 0;
 
 // Creating an array for ratings of target customer
 static int[] targetRatings = new int[10];
 static String targetName = "";

 public static void main(String[] args) throws FileNotFoundException {

 // Creating a scanner for both rating and action files
 Scanner scannerRatings = new Scanner(new File(args[0]));
 Scanner scannerActions = new Scanner(new File(args[1]));
  
 targetName = scannerRatings.nextLine();
 
 // While ratings file has next line split it at each space and store each thing
 while (scannerRatings.hasNextLine()) { 
  String inputRatings = scannerRatings.nextLine();
  String[] inputRatingsSplit = inputRatings.split(" ");
  
  // Array for the ratings
  int[] ratings = new int[10];
  
  // Looping through adding each rating to the array
  for (int i = 1; i <= 10; i++) {
   int temp = Integer.parseInt(inputRatingsSplit[i]);
   ratings[i-1] = temp;
  }
  
  // If the customer is the target customer store it as target ratings
  // If its not the target customer create a new customer
  if (targetName.equals(inputRatingsSplit[0])) {
   targetRatings = ratings;
  }
  else { 
   Customer customer = new Customer(inputRatingsSplit[0], ratings);
   insertCustomer(customer);
  }


 }
 
  // While the action file has another line split the line at each space and store each thing
  while (scannerActions.hasNextLine()) {
   String inputAction = scannerActions.nextLine();
   String[] inputActionSplit = inputAction.split(" ");
  
  
   // If the action  is AddCustomer output the entire line and make a rating array
   if (inputActionSplit[0].equals("AddCustomer")) {
    System.out.println(inputAction);
    int ratings[] = new int[10];
   
    // For each rating store it as a temp to turn it into an integer, then add it to the ratings array
    for (int i = 2; i <= 11; i++) {
      int temp = Integer.parseInt(inputActionSplit[i]); 
      ratings[i - 2] = temp;
    } 
    
    // Create a new customer
    Customer newCustomer = new Customer(inputActionSplit[1], ratings);
    insertCustomer(newCustomer);
   
   }
   
   // If the action is RecommendSongs call the recommendSongs method
   if (inputActionSplit[0].equals("RecommendSongs")) {
   recommendSongs();
   }
  
   //If the action is PrintCustomerDistanceRatings call the printCustomerDistanceRatings method
   if (inputActionSplit[0].equals("PrintCustomerDistanceRatings")) {
    printCustomerDistanceRatings();
   }
 
 }  

}


 // Creating an interCustomer method
 static void insertCustomer(Customer newCustomer) {
   customer[numCustomers] = newCustomer;
   numCustomers++;
   
   //Creating and array list for customers
   ArrayList<Customer> customerList = new ArrayList<>();
   
   // Add every customer to the customer list
   for (int i = 0; i < numCustomers; i++) {
    customerList.add(customer[i]);
   }

   // Alphabetically sort the list of customers
   Collections.sort(customerList);
   
   // Copies the customer back into the array 
   for (int i = 0; i < numCustomers; i++) {
    customer[i] = customerList.get(i);
   }
       

 }

 // Creating a method to calculate the difference between ratings
 static double calculateDistance(Customer other) {
  int sharedCount = 0;
  int differenceSum = 0;
  
  // Compare customer ratings to target ratings
  for (int i = 0; i < other.ratings.length; i++) {

   // If the customer and the target customer have both rated the song calculate the difference
   // Add one to the count of amount of same songs rated
   if (other.ratings[i] > 0 && targetRatings[i] > 0) {
    sharedCount++;
    differenceSum += Math.abs(other.ratings[i] - targetRatings[i]);
   }
  } 
  
  // If the customer and target customer havent rated any of the same songs distance is undefined
  if (sharedCount == 0) {
   return -1;
  }
  
  return 1.0 / sharedCount + (double) differenceSum / sharedCount;

 }


// Creating recommendSongs method
static void recommendSongs() {
 boolean recSong = false;
 HeapPriorityQueue<Double, Customer> queue = new HeapPriorityQueue<>();
 
 Entry<Double, Customer>[] entryArray = (Entry<Double, Customer>[]) new Entry[numCustomers];
 int numEntries = 0;
  
 // Creating an array to store customer entries and initializing the priority queue
 for (int i = 0; i < numCustomers; i++) {
  double distance = calculateDistance(customer[i]);
  
  if (distance != -1) {
   entryArray[numEntries] = queue.createEntry(distance, customer[i]);
   numEntries++;
  }
 }
 queue.initialize(entryArray, numEntries);
 
 // While the queue isnt empty remove the minimum entry
 // Reset the boolean to false
 while (!queue.isEmpty()) {
  Entry<Double, Customer> entry  = queue.removeMin(); 
  Customer other = entry.getValue();
  recSong = false;
  
  // For all the customers ratings check if the target customer hasent rated the same song yet
  // If there is a song set the boolean to true that the customer has a posible song recommendation
  for (int i = 0; i < other.ratings.length; i++) {
   if (other.ratings[i] >= 4 && targetRatings[i] == 0) {
    recSong = true;
   }
  }
  
  // If customer has a song recomendation for target customer output it, if multiple songs output them all
  if (recSong == true) {
  System.out.print("RecommendSongs " + other.name);
    for (int i = 0; i < other.ratings.length; i++) {
     if (other.ratings[i] >= 4  && targetRatings[i] == 0) {
      System.out.print(" song" + i + " " + other.ratings[i] + " ");
     }
    }
   return;
  }
 }
 
 // If the customer has no available song recommendations output none
 System.out.println("RecommendSongs none"); 

}  

// Create a method for printing customer distance ratings 
static void printCustomerDistanceRatings() {
 System.out.println("PrintCustomerDistanceRatings");
 System.out.println("distance customer s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
 
 // Print the target customers name
 System.out.printf("%-20s", targetName);
 
 // Print all target ratings then pring a line
 for (int i = 0; i < targetRatings.length; i++) {
  System.out.printf("%3d", targetRatings[i]);
 }
  System.out.println();
  
  // Calculate the customers ratings difference from the target customer using shared song count
  for (int i = 0; i < numCustomers; i++) {
   double distance = calculateDistance(customer[i]);
   
   // If distance is undefined output -----
   // Else output the distance and the customers name and all of their ratings 
   if (distance == -1) {
    System.out.printf("%-9s", "-----");
   }
   else { 
    System.out.printf("%-9.3f", distance);
   }
   System.out.print(" ");
   System.out.printf("%-11s", customer[i].name);
   
   for (int j = 0; j < customer[i].ratings.length; j++) {
    System.out.printf("%3d", customer[i].ratings[j]);
   } 
  
  System.out.println();
  
  }
  
}

// Creating a method to find a customer using binary search
static Customer findCustomer(String name) {
 int low = 0;
 int high = numCustomers - 1;
 
 while (low <= high) {
  int mid = (low + high) / 2;
  
  int comparison = customer[mid].name.compareTo(name);
  
  if (comparison == 0) {
   return customer[mid];
  }
  else if (comparison < 0) {
   low = mid + 1;
  } 
  else {
   high = mid - 1;
  }
 }
 return null;
}
}

// Creating a class to represent a customer and their ratings 
class Customer implements Comparable<Customer> {

 String name;
 int[] ratings;
 
 Customer(String name, int[] ratings) {
  this.name = name;
  this.ratings = ratings;
 }
 
 public int compareTo(Customer other) {
  return this.name.compareTo(other.name);
 }
 
}


