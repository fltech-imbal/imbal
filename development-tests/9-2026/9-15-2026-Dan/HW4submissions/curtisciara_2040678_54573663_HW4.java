/* Author: Ciara Curtis
   Email: ccurtis2025@my.fit.edu
   Course: CSE2010
   Section: 4
   Description of this file: Homework 4 Submission blah blah blah 
*/

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;

public class HW4 {

  //Holds the customers read from the input file
  public static ArrayList<Customer> customerList= new ArrayList<Customer>();

  /*This method is used to presort the customer object before it goes
    into the array list */
  public static void addToList(Customer customerToAdd) {

    if(customerList.isEmpty()) { //special case: list is empty
      customerList.add(customerToAdd);
    }
    else {
      for(int i = 0; i < customerList.size(); i++) {
        if(customerToAdd.getName().compareTo(customerList.get(i).getName()) <= 0) {
          customerList.add(i,customerToAdd);
          return;
        }
      }
      customerList.add(customerToAdd);
    }
  }

  /*Method that takes an array and converts the elements to
    a formatted string */
  public static String arrayToString(int[] arr) {
    String str = "";
    for(int i = 0; i < arr.length; i++) {
      str += arr[i] + "  ";
    }

    return str;
  }

  /* Searches the arraylist for a specific customer name
     and returns the Customer Object, a basic binary search
  */
  public static Customer binarySearch(String custName) {
    int low = 0;
    int high = customerList.size() - 1;

    while(low <= high) { //if customer name at mid index matches the variable custName
      int mid = (low + high)/2;
      if(customerList.get(mid).getName().equals(custName)) return customerList.get(mid);
      else { //if doesnt match, then update high, low, and mid indexes and try again
	if(customerList.get(mid).getName().compareTo(custName) >= 0) {
          high = mid - 1;
	}
	else {
	  low = mid + 1;	  
	}

	mid = low + (high - low)/2;
      }      

    }

    return null;
  }

  public static void main(String[] args) throws FileNotFoundException {
    //Scanner to read in the initial customers file and their song ratings
    Scanner customers = new Scanner(new File(args[0]));
    
    //Holds the target customer name to later find the
    //target customer object    
    String targetCustName = customers.next();

    //Customer reference variable that will hold the target customer
    Customer targetCustomer = new Customer();

    /*While loop that reads in the customers name, and then creates
      an integer array to hold the customer's song ratings. It
      then compares the customers name to the target customer
      name and if they match, then the target customer variable is
      updated. If they do not match, then the customer object is
      added to the arraylist */
    while(customers.hasNext()) {
      String customerName = customers.next();
      int[] songRatings = new int[10];
       
      //for loop that reads in the song ratings and updates the
      //integer array
      for(int i = 0; i < songRatings.length; i++) {
        songRatings[i] = customers.nextInt();
      }
      
      //if the customer name equals the target customer name
      //then set the target customer reference variable equal
      //to a customer object with the name and songRating list
      if(customerName.equals(targetCustName)) {
	targetCustomer = new Customer(customerName, songRatings);
      }
      
      //adds them to the array list if not target customer
      else {
        addToList(new Customer(customerName, songRatings));
      }
    }

    //Priority queue that holds the customer and the distances is the key
    HeapPriorityQueue<Double, String> distances = new HeapPriorityQueue<Double, String>();

    //For each customer object in the arrayList, add to the priority queue while calculating
    //the distance of each object to the target Customer.    
    for(int i = 0; i < customerList.size(); i++) {
      customerList.get(i).setDistance(targetCustomer);
      distances.insert(customerList.get(i).getDistance(), customerList.get(i).getName());
    }    
    
    //Finished with the first file, so this one closes
    customers.close();

    //New scanner to handle inquiries from the second file
    Scanner inquiries = new Scanner(new File(args[1]));

    /*While loop that continues until the inquiries scanner
      has nothing left to read */
    while(inquiries.hasNext()) {
      String inquiryType = inquiries.next();  //stores the type of inquiry
      String outputString = inquiryType + " ";
      if(inquiryType.equals("AddCustomer")) {  
        String customerName = inquiries.next();   //stores the customers name
        outputString += customerName + " ";
        int[] songs = new int[10];                //will store the customer's song list
        for(int i = 0; i < songs.length; i++) {
	  songs[i] = inquiries.nextInt();
	  outputString += songs[i] + " ";
        }

	//Reprints the initial inquiry string
	System.out.println(outputString);

        //Creates a new Customer object to add to the distance priority heap and the arrayList
        Customer newCustomer = new Customer(customerName, songs);
        newCustomer.setDistance(targetCustomer);

        //adds the newCustomer to the priority heap and the distance from the target customer is added
        distances.insert(newCustomer.getDistance(), customerName);

        //adds the newCustomer object to the arraylist
        addToList(newCustomer);
      }

      else if(inquiryType.equals("RecommendSongs")) {
	//Get the closest Customer Object to the target customer 
        //by creating a reference variable that saves the min from
        //the priority heap
        String nearestName = distances.min().getValue();
	Customer nearestCustomer = binarySearch(nearestName);
	

        String songList = "";
	//for each song in the list, if the target customer has not rated it
        //and the nearest customer rated it over a four, print it

        while(songList.isEmpty()) {
          for(int i = 0; i < nearestCustomer.getRatings().length; i++) {
            if(targetCustomer.getRatings()[i] == 0 && nearestCustomer.getRatings()[i] >= 4)
              songList += "song" + i + " " + nearestCustomer.getRatings()[i] + " ";   
	  } //if the recommendedsongs are empty, then move onto the next closest customer
          if(songList.isEmpty()) { nearestCustomer = binarySearch(distances.removeMin().getValue()); }
        }
	System.out.println(inquiryType + " " + nearestCustomer.getName() + " " + songList);
      }

      else if(inquiryType.equals("PrintCustomerDistanceRatings")) {
        System.out.println(inquiryType);
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
        int spaceAmount = 11 - targetCustName.length();
	String spaces = "";
	for(int i = 1; i <= spaceAmount; i++) spaces += " "; //alignment after the target customer name
	System.out.println("\t " + targetCustName + spaces + arrayToString(targetCustomer.getRatings()));

        for(int i = 0; i < customerList.size(); i++) {
          spaces = "";
          spaceAmount = 11 - customerList.get(i).getName().length(); //gets the name length of the next customer to be printed
          for(int j = 1; j <= spaceAmount; j++) spaces += " "; //alignment for the songratings
          if(customerList.get(i).getDistance() == 5000) { System.out.printf("-----\t %s%s%s\n",customerList.get(i).getName(), spaces, arrayToString(customerList.get(i).getRatings())); }
	  else System.out.printf("%1.3f\t %s%s%s\n", customerList.get(i).getDistance(), customerList.get(i).getName(), spaces, arrayToString(customerList.get(i).getRatings()));
        }
      }
    }
  }
}
