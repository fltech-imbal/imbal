/*

  Author: Milena Koullapi
  Email:mkoullapi2025@my.fit.edu
  Course: CSE 2010
  Section: 01 
  Description of this file: eads customer song ratings and actions from input files, maintains an alphabetically 
  sorted customer list, and uses a min-heap priority queue to calculate distance metrics and recommend songs to a target customer.

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;

public class HW4
{

  /*
   This method reads ratings and actions filenames from command-line arguments, populates the customer 
   database from the ratings file, and processes action commands to print customer distance tables or song recommendations.
   @param args - command-line arguments; args[0] is the path to the ratings file and args[1] is the path to the actions file.
  */  
  public static void main(String[] args){
    /* 
      Filename store the name or path of the file that is going to be read.
      Thats information is received from args[0], which represents the first command-line argument 
    */
    String filename = args[0];
    String targetCustomerName = "";

    try ( Scanner scanner = new Scanner(new File(filename))){

      // Reads the first line of the ratings file, which specifies the target customer's name
      if(scanner.hasNextLine()){
        targetCustomerName = scanner.nextLine().trim();
      }

      // Checks if there is another line of text left to read in the file. The loop will continue running as long as there is more lines.
      while (scanner.hasNextLine()) {
        // Reads the next line of text and stores it in the variable called line.
        String line = scanner.nextLine().trim();
        if (line.isEmpty()) continue;

        String[] parts = line.split("\\s+");
        String name = parts[0];
        int[] ratings = new int[10];
        for (int i = 0; i < 10; i++) {
            ratings[i] = Integer.parseInt(parts[i + 1]);
        }
        Customer.insertCustomer(name, ratings);
      }

    }
    //This block only runs if the attempt to open the file specified by filename doesnt exists, cannot be opens or found in the specific path.
    catch (FileNotFoundException e) { 
      //It print "File not found" to the user
      System.out.println("File not found.");
      //it print the detailed technical details of the errors to help the developers understand where and why the error happened
      e.printStackTrace();
    }

    String actionsFilename = args[1];
    try (Scanner actionScanner = new Scanner(new File(actionsFilename))){
      // Reads the actions file line by line and processes each specific request
      while(actionScanner.hasNextLine()){
        String actionLine = actionScanner.nextLine().trim();
        if (actionLine.isEmpty()) continue;

        String[] parts = actionLine.split("\\s+");
        String command = parts[0];

        // Adds a new customer and their 10 song ratings into the sorted array
        if (command.equals("AddCustomer")) {
          System.out.println(actionLine);
          String name = parts[1];
          int[] ratings = new int[10];
          for (int i = 0; i < 10; i++) {
            ratings[i] = Integer.parseInt(parts[i + 2]);
          }
           Customer.insertCustomer(name, ratings);

        // Formats and prints all customer ratings along with their distance to the target customer
        } else if (command.equals("PrintCustomerDistanceRatings")) {
            printCustomerDistanceRatings(targetCustomerName);

        // Uses the MinHeap priority queue to recommend songs from the closest similar customer
        } else if (command.equals("RecommendSongs")) {
            RecommendSongs(targetCustomerName);
        }

      }

    }
    
     //This block only runs if the attempt to open the file specified by filename doesnt exists, cannot be opens or found in the specific path.
    catch (FileNotFoundException e) { 
      //It print "File not found" to the user
      System.out.println("File not found.");
      //it print the detailed technical details of the errors to help the developers understand where and why the error happened
      e.printStackTrace();
    }
  
  }

    /*
      Represents an individual customer with a name and an array of 10 song ratings, 
      as well as static operations for maintaining a sorted array of customers.
    */
    public static class Customer{
      
      private String name;
      private int[] rating;
      
      /*
        Constructor to initialize a Customer object with a name and ratings array.
        @param name - customer's name
        @param rating - array of 10 song ratings
      */
      public Customer(String name, int[] rating){
        this.name = name;
        this.rating = rating;
      }

      /*
        Returns the customer's name.
        @return name string
      */
      public String getName(){
        return name;
      }

      /*
        Returns the customer's array of song ratings.
        @return integer array of ratings
      */
      public int[] getRating(){
        return rating;
      }

      static Customer[] customers = new Customer[100];
        static int size = 0;

      /*
        Inserts a new customer into the static array while maintaining alphabetical order.
        @param name - customer's name
        @param ratings - customer's song ratings array
      */
      static void insertCustomer(String name, int[] ratings) {
        Customer newCust = new Customer(name, ratings);

        int pos = 0;
        while (pos < size && customers[pos].getName().compareTo(name)<0) {
          pos++;
        }

        for (int i = size; i > pos; i--) {
          customers[i] = customers[i-1];
        }

        customers[pos] = newCust;
        size = size + 1;
      }

      /*
        Searches for a customer by name using binary search on the sorted customer array.
        @param name - name of the customer to find
        @return matching Customer object, or null if not found
      */
      static Customer findCustomer(String name){
        int low = 0;
        int high = size -1; 

        while(low <= high){
          int mid = (low + high) / 2;
          int cmp = customers[mid].getName().compareTo(name);

          if(cmp == 0){
            return customers[mid];
          }
          else if(cmp < 0){
            low = mid + 1;
          }
          else{
            high = mid - 1 ;
          }
        }

        return null ;

      }

      /*
        Calculates the distance between two customers based on shared non-zero song ratings.
        @param x - first customer
        @param y - second customer
        @return calculated distance double value, or -1 if no songs are mutually rated
      */
      static double distance(Customer x, Customer y){
        int[] a = x.getRating();
        int[] b = y.getRating();

        int count = 0;
        int sum = 0;

        for (int s = 0; s < 10; s++){
          if(a[s] > 0 && b[s] > 0){
            count++;
            sum = sum + Math.abs(a[s]-b[s]);
          }
        } 
        
        if(count == 0){
          return -1;
        }

        return 1.0/count + (1.0/count) * sum; 
      }




    }

    //Represents an entry in the priority queue, holding a key (distance) and value (customer name).
    public static class HeapEntry{
      private double key;
      private  String value;

      /*
        Constructor to create a HeapEntry with a distance key and customer name value.
        @param key - distance metric
        @param value - customer name
      */
      public HeapEntry(double key, String value){
        this.key = key;
        this.value = value;
      }

      /*
        Returns the key (distance) of this heap entry.
        @return key double value
      */
      public double getKey(){
        return key;
      }

      /*
        Returns the value (customer name) of this heap entry.
        @return customer name string
      */
      public String getValue(){
        return value;
      }

      /*
        Compares two HeapEntry objects by key, breaking ties alphabetically by value.
        @param a - first entry
        @param b - second entry
        @return negative if a < b, positive if a > b, or comparison of values if equal
      */
      static int compare(HeapEntry a, HeapEntry b){
        if(a.getKey() < b.getKey()){
          return -1;
        }
        else if(a.getKey() > b.getKey()){
          return 1;
        }
        else{
          return a.getValue().compareTo(b.getValue());
        }

      }

    }

    //Array-based implementation of a MinHeap priority queue storing HeapEntry elements.
    public static class MinHeap {
      private HeapEntry[] heap = new HeapEntry[100];
      private int heapSize = 0;

      /*
        Inserts a new HeapEntry into the heap and restores heap order with upheap.
        @param entry - HeapEntry to insert
      */
      public void insert(HeapEntry entry){
        if(isFull()) return;
        heap[heapSize] = entry;
        upheap(heapSize);
        heapSize++;
      }

      /*
        Removes and returns the minimum entry (root) from the heap, restoring order with downheap.
        @return minimum HeapEntry object
      */
      public HeapEntry removeMin(){
        if (isEmpty()) return null;
        HeapEntry min = heap[0];
        heap[0] = heap[heapSize - 1];
        heapSize--;
        downheap(0);
        return min;
      }

      /*
        Returns the minimum entry at the root without removing it.
        @return root HeapEntry object
      */
      public HeapEntry getMin(){
        if(isEmpty()) return null;
        return heap[0];
      }

      /*
        Checks whether the min-heap has reached its maximum capacity.
        @return true if full, false otherwise
      */
      public boolean isFull(){
        return heapSize == 100;
      }

      /*
        Checks whether the min-heap contains no entries.
        @return true if empty, false otherwise
      */
      public boolean isEmpty(){
        return heapSize == 0;
      }

      /** Moves the entry at index j higher, if necessary, to restore the heap property. */
      private void upheap(int j) {
        while (j > 0) {            // continue until reaching root (or break statement)
          int p = (j-1)/2;
          if (HeapEntry.compare(heap[j], heap[p]) >= 0) break; // heap property verified
          swap(j, p);
          j = p;                                // continue from the parent's location
        }
      }

      /*
        Moves the entry at index j lower, if necessary, to restore the min-heap property.
        @param j - starting index for downheap operation
      */
      private void downheap(int j) {
        while (2 * j + 1 < heapSize) {
            int left = 2 * j + 1;
            int right = 2 * j + 2;
            int smallest = left;

            if (right < heapSize && HeapEntry.compare(heap[right], heap[left]) < 0) {
                smallest = right;
            }

            if (HeapEntry.compare(heap[j], heap[smallest]) <= 0) break;

            swap(j, smallest);
            j = smallest;
        }
      }

      /*
        Swaps elements at positions i and j in the heap array.
        @param i - first element index
        @param j - second element index
      */
      private void swap(int i, int j) {
        HeapEntry temp = heap[i];
        heap[i] = heap[j];
        heap[j] = temp;
      } 

    }

    /*
      Formats and prints a table of distance metrics and 10 song ratings for all customers relative to the target customer.
      @param targetCustomerName - name of the target customer
    */
    public static void printCustomerDistanceRatings(String targetCustomerName){
      System.out.println("PrintCustomerDistanceRatings");
      System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");

      Customer target = Customer.findCustomer(targetCustomerName);

      for (int i = 0; i < Customer.size; i++) {
        Customer c = Customer.customers[i];
        String distStr;

        if (c.getName().equals(targetCustomerName)) {
            distStr = "        "; // blank space for target customer
        } else {
            double dist = Customer.distance(target, c);
            if (dist < 0) {
              distStr = "-----   ";
            } else {
              distStr = String.format("%.3f   ", dist);
            }
        }

        System.out.printf("%s%-10s", distStr, c.getName());
        int[] r = c.getRating();

        for (int j = 0; j < 10; j++) {
            System.out.printf("%3d", r[j]);
        }
        System.out.println();
      }

    }

    /*
      Recommends high-rated songs (>= 4) unrated (0) by the target customer from the closest customer found via MinHeap.
      @param targetCustomerName - name of the target customer
    */
    public static void RecommendSongs(String targetCustomerName){
      Customer target = Customer.findCustomer(targetCustomerName);
      MinHeap heap = new MinHeap();

      for (int i = 0; i < Customer.size; i++) {
        Customer c = Customer.customers[i];
        
        if (!c.getName().equals(targetCustomerName)) {
          double dist = Customer.distance(target, c);
          
          if (dist >= 0) {
            heap.insert(new HeapEntry(dist, c.getName()));
          }
        }
      }

      while (!heap.isEmpty()) {
        HeapEntry entry = heap.removeMin();
        Customer closest = Customer.findCustomer(entry.getValue());

        int[] targetRatings = target.getRating();
        int[] closestRatings = closest.getRating();
        String recommendations = "";

        for (int s = 0; s < 10; s++) {
          if (targetRatings[s] == 0 && closestRatings[s] >= 4) {
            recommendations += " song" + s + " " + closestRatings[s];
          }
        }

        if (!recommendations.isEmpty()) {
          System.out.println("RecommendSongs " + closest.getName() + recommendations);
          return;
        }
      }

      System.out.println("RecommendSongs none");

    }

}
