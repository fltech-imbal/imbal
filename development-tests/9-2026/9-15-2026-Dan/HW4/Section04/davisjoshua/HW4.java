/**
 Author: Joshua Davis
 Email: joshuadavis2025@my.fit.edu
 Course: CSE 2010 - Algorithms & Data Structures
 Section: 4

 Description:
 This Program takes 2 arguments to recommend songs to a target customer based on ratings of other customers that have
 similar ratings, this is implemented by a priority queue in the form of an array
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW4 {
    public static void main(String[] args) throws FileNotFoundException {
        //Create the files to read in at program run
        File ratings = new File(args[0]);
        File actions = new File(args[1]);

        //These are the scanners that read the files
        Scanner actionReader = new Scanner(actions);
        Scanner ratingReader = new Scanner(ratings);

        //Declaration of the array of customers and set the amount of customers to 0 initially
        Customer[] customers = new Customer[100];
        int customerCounter = 0;

        //As given the first line of the line is the target name to recomend songs
        String targetName = ratingReader.nextLine();

        //This loop reads in the ratings file to make the customers and their corresponding ratings
        while (ratingReader.hasNext()) {
            String name = ratingReader.next();
            Customer customer = new Customer(name, new int[10]);
            for (int i = 0; i < 10; i++) {
                customer.ratings[i] = ratingReader.nextInt();

            }
            insertCustomer(customer, customers, customerCounter);
            customerCounter++;
        }

        //Sanity Check
        /*
        for (int i = 0; i < customerCounter; i++) {
            System.out.println(customers[i].name);
        }
        */

        //Using the targetName we use the method to find the Customer object
        Customer targetCustomer = findCustomer(targetName, customers, customerCounter);

        //We make a heap object which is the priority queue implementation to recommend songs
        Heap recommendationHeap = new Heap();

        //
        for (int i = 0; i < customerCounter; i++) {
            if (!customers[i].name.equals(targetCustomer.name)) {
                double distance = customerDistance(customers[i], targetCustomer);
                if (!Double.isNaN(distance)) {
                    Heap.HeapData entry = new Heap.HeapData();
                    entry.distance = distance;
                    entry.customerName = customers[i].name;
                    recommendationHeap.insert(entry);
                }
            }
        }

        while (actionReader.hasNextLine()) {
            String action = actionReader.nextLine();

            if (action.startsWith("AddCustomer ")) {
                Scanner lineReader = new Scanner(action.substring("AddCustomer ".length()));

                String newName = lineReader.next();
                int[] newRatings = new int[10];
                for (int i = 0; i < 10; i++) {
                    newRatings[i] = lineReader.nextInt();
                }
                Customer newCustomer = new Customer(newName, newRatings);
                customerCounter = addCustomer(newCustomer, customers, recommendationHeap, customerCounter, targetCustomer);
                System.out.println(action);
            }
            else if (action.equals("PrintCustomerDistanceRatings")) {
               System.out.println("PrintCustomerDistanceRatings");
                System.out.printf("%-9s%-11s", "distance", "customer");
               for (int i = 0; i < 10; i++) {
                   System.out.print("s" + i + " ");
               }
               System.out.println();
                System.out.printf("%-9s", "");
                System.out.printf("%-11s", targetCustomer.name);
                for (int i = 0; i < 10; i++) {
                    System.out.printf("%-3d", targetCustomer.ratings[i]);
                }
                System.out.println();
                for  (int i = 0; i < customerCounter; i++) {
                    if (!customers[i].name.equals(targetCustomer.name)) {
                        double  distance = customerDistance(customers[i], targetCustomer);
                        if (Double.isNaN(distance)) {
                            System.out.printf("%-9s","-----");
                        } else {
                            System.out.printf("%-9.3f", distance);
                        }
                        System.out.printf("%-11s", customers[i].name);
                        for  (int j = 0; j < 10; j++) {
                            System.out.printf("%-3d", customers[i].ratings[j]);
                        }
                        System.out.println();
                    }
                }
            }
            else if (action.equals("RecommendSongs")) {
                Heap.HeapData[] removedData = new Heap.HeapData[customerCounter];
                int removedCounter = 0;

                boolean foundRecommendation = false;

                while(!recommendationHeap.isEmpty()) {
                    Heap.HeapData currentData = recommendationHeap.removeMin();
                    removedData[removedCounter] = currentData;
                    removedCounter++;


                    Customer currentCustomer = findCustomer(currentData.customerName, customers, customerCounter);

                    int[] recommendedSongs = new int[10];
                    int songCounter = 0;


                    for  (int i = 0; i < currentCustomer.ratings.length; i++) {
                        if (targetCustomer.ratings[i] == 0 && currentCustomer.ratings[i] >= 4) {
                            recommendedSongs[songCounter] = i;
                            songCounter++;
                        }
                    }
                    if (songCounter > 0) {
                        foundRecommendation = true;

                        System.out.print("RecommendSongs " + currentCustomer.name);

                        for (int j = 0; j < songCounter; j++) {
                            int songIndex = recommendedSongs[j];
                            System.out.print(" song" + songIndex + " " + currentCustomer.ratings[songIndex]);
                        }
                        System.out.println();
                        break;
                    }
                }
                if (!foundRecommendation) {
                    System.out.println("RecommendSongs none");
                }
                for (int i = 0; i < removedCounter; i++) {
                    recommendationHeap.insert(removedData[i]);
                }
            }
        }

    }
    /*This method takes 3 parameters, the customer we want to insert, the array to insert into, and pass in the customer
    to be able to update the counter after insertion. Loop through the array to find where in aplhabetical order to
    insert the customer
     */
    static void insertCustomer(Customer toInsert, Customer[] toPutIn, int customerCounter) {
        for (int i = 0; i < customerCounter; i++) {
            if(toInsert.name.compareTo(toPutIn[i].name) <= 0) {
                for (int j = customerCounter; j > i; j--) {
                    toPutIn[j] = toPutIn[j-1];
                }
                toPutIn[i] = toInsert;
                return;
            }
        }
        toPutIn[customerCounter] = toInsert;
        return;
    }
    //This method takes a name of the customer to find, the array searching and the customer counter to know the bounds
    //of the array, the method uses binary search to search for the customer in the array, using alphabetical order
    static Customer findCustomer(String name, Customer[] toSearch, int customerCounter ) {
        int left = 0;
        int right = customerCounter - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            int comparison = name.compareTo(toSearch[mid].name);
            if(comparison > 0) {
                left = mid + 1;
            }
            else if(comparison < 0) {
                right = mid - 1;
            }
            else {
                return toSearch[mid];
            }
        }
        return null;
    }
    //This method calculates distance between two customers with both parameters being customers
    //for each rating it calculates distance and then finds the average of those calculated distances
    //If the distance is invalid it returns a not a number
    static double customerDistance(Customer customer1, Customer customer2) {
        int ratedSongs = 0;
        int sumOfDistance = 0;
        for (int i = 0; i < 10; i++) {
            if (customer1.ratings[i] > 0 && customer2.ratings[i] > 0) {
                ratedSongs++;
                sumOfDistance += Math.abs(customer1.ratings[i] - customer2.ratings[i]);
            }
        }
        if (ratedSongs > 0) {
            return (1.0 / ratedSongs) + ((double) sumOfDistance / ratedSongs);
        }
        return Double.NaN;
    }
    //this method has parameters of a customer to add, the array to insert into, the heap to go into, the
    //customer counter to get the bounds of the array, and the target customer
    //the method adds the customer in alphabetical order, calculates distance between it and the target customer to get
    //the priority, then adds to the heap with that priority
    static int addCustomer(Customer toAdd, Customer[] toPutIn, Heap destinationHeap, int customerCounter, Customer targetCustomer) {
        insertCustomer(toAdd, toPutIn, customerCounter);
        customerCounter++;
        double distance = customerDistance(targetCustomer, toAdd);
        if (!Double.isNaN(distance)) {
            Heap.HeapData entry = new Heap.HeapData();
            entry.distance = distance;
            entry.customerName = toAdd.name;
            destinationHeap.insert(entry);
        }
        return customerCounter;
    }
}
class Customer {
    String name;
    int[] ratings;

    public Customer(String name, int[] ratings) {
        this.name = name;
        this.ratings = ratings;
    }

}
//This is the heap class, the data stored is distance and its name, it also includes an array with the rating data.
//there are helper methods to get the left child and right child and get the parent with formulas for a binary tree
//implementation with an array
//there are also methods that support upheaping and downHeaping
//there is an insert method, a remove min method that removes the root of the heap and then moves data around
//to satisfy the heap properties
//then isEmpty and isFull checkers
class Heap  {

    static class HeapData {
        double distance;
        String customerName;

    }

    HeapData[] heapEntry = new HeapData[100];
    int entryCount = 0;

    int getLeftChild(int childIndex) {
        return 2 * childIndex + 1;
    }
    int getRightChild(int childIndex) {
        return 2 * childIndex + 2;
    }
    int getParent(int childIndex) {
        return (childIndex - 1) / 2;
    }
    void upheap(int childIndex) {
        while (childIndex > 0) {
            int parent = getParent(childIndex);
            if (compareEntries(heapEntry[childIndex], heapEntry[parent]) < 0) {
                HeapData temp = heapEntry[parent];
                heapEntry[parent] = heapEntry[childIndex];
                heapEntry[childIndex] = temp;
                childIndex = parent;
            }
            else {
                break;
            }
        }
    }
    void downheap(int parentIndex) {
        while (getLeftChild(parentIndex) < entryCount) {
            int left = getLeftChild(parentIndex);
            int right = getRightChild(parentIndex);

            int smallestChild = left;
            if (right < entryCount && compareEntries(heapEntry[right], heapEntry[left]) < 0) {
                smallestChild = right;
            }
            if (compareEntries(heapEntry[smallestChild], heapEntry[parentIndex]) < 0) {
                HeapData temp = heapEntry[smallestChild];
                heapEntry[smallestChild] = heapEntry[parentIndex];
                heapEntry[parentIndex] = temp;
                parentIndex = smallestChild;
            }
            else {
                break;
            }
        }
    }
    int compareEntries(HeapData alpha, HeapData beta) {
        if (alpha.distance < beta.distance) {
            return -1;
        }
        else if (alpha.distance > beta.distance) {
            return 1;
        }
        else {
            return alpha.customerName.compareTo(beta.customerName);
        }
    }
    void insert(HeapData Entry) {
        if (entryCount == heapEntry.length) {
            return;
        }
        heapEntry[entryCount] = Entry;
        entryCount++;
        upheap(entryCount - 1);
    }
    HeapData removeMin() {
        if (entryCount == 0) {
            return null;
        }
        HeapData smallest = heapEntry[0];
        heapEntry[0] = heapEntry[entryCount - 1];
        heapEntry[entryCount - 1] = null;
        entryCount--;
        downheap(0);
        return smallest;
    }
    HeapData getMin() {
        if (entryCount == 0) {
            return null;
        }
        return heapEntry[0];
    }
    boolean isEmpty() {
        return entryCount == 0;
    }
    boolean isFull() {
        return entryCount == heapEntry.length;
    }
}