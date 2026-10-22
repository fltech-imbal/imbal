/*
Name: Avery Haughwout
Course: CSE2012
Section: 3:30
Description: Uses heap implementation to determine the closest similarity amid simultaneous array entries between two elements
to predict future compatibilities.
 */

//IMPORTS
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class HW4{
    public static void main(String[] args)throws FileNotFoundException{
        Scanner scnr1 = new Scanner(new File(args[0]));// Reads hw4in1ratings.txt
        Scanner scnr2 = new Scanner(new File(args[1]));// Reads hw4in1actions.txt

        Customer[] custArray = new Customer[100]; //Makes the cust array holder :)

        String targetCust = scnr1.nextLine(); //The first line is our target customer - this will prove crucial
        int size = 0; //Declares initial size of the array
        while(scnr1.hasNext()){
            String input = scnr1.nextLine();
            //System.out.println(input);
            String[] inputs = input.split(" ");
            int[] reviews = new int[10]; //Preps the review array + saves the customers name
            String newCustName = inputs[0];
            for(int i = 0; i < 10; i++){ //Populates the rating array
                reviews[i] = Integer.parseInt(inputs[i+1]);
            }
            custArray[size] = new Customer(newCustName, reviews); //Adds the new customer + updates the size accordingly
            size++;
        }
        Arrays.sort(custArray, 0, size); //Sorts only the non-null section :)
        HeapPriorityQueue custHeap = new HeapPriorityQueue(); //Declares the heap (note: this implementation is FINNICKY)

        double[] keyHolder = new double[100]; //Stores all distance keys for future reference...this doesn't get sorted alongside the custArray
        for(int i = 0; i < size; i++){ //Populates the heap
            if(custArray[i].getName().equals(targetCust) == false){ //Adds all values EXCEPT the target to the heap using the getDistance method (Line 125)
                double key = getDistance(custArray[findCustomer(custArray, targetCust, 0, size)].getRatings(), custArray[i].getRatings());
                keyHolder[i] = key; //It's a surprise tool that will help us later...
                if(key != 0){ //Will NOT add the key to the heap if its zero - zero is our null response
                    custHeap.insert(key, custArray[i].getName());
                }
            }
        }


        while(scnr2.hasNext()){ //Classic setup- reads imput, breaks it into String[], switch statement for the 1st input
            String input = scnr2.nextLine();
            String[] inputs = input.split(" ");
            switch (inputs[0]){
                case "AddCustomer":
                    /*
                    1- Defines the new customer based on the input line
                    2 - Puts the next customer into the custArray and sorts
                    3- Computes a key and puts the customer onto the heap
                    */
                    int[] reviews = new int[10];
                    String newCustName = inputs[1];
                    for(int i = 0; i < 10; i++){
                        reviews[i] = Integer.parseInt(inputs[i+2]);
                    }
                    custArray[size] = new Customer(newCustName, reviews);
                    size++;
                    Arrays.sort(custArray, 0, size);
                    double key = getDistance(custArray[findCustomer(custArray, targetCust, 0, size)].getRatings(), custArray[size - 1].getRatings());
                    keyHolder[size] = key;
                    custHeap.insert(key, custArray[size - 1].getName());
                    break;

                case "RecommendSongs":
                    //Saves the two needed customers for easy reference, with closestCustomer coming from
                    Customer targetCustomer = custArray[findCustomer(custArray, targetCust, 0, size)];
                    Customer closestCustomer = custArray[findCustomer(custArray, (String) custHeap.removeMin().getValue(), 0, size)];
                    while(checkCompatibility(targetCustomer, closestCustomer) == false){//See 117- this is just to ensure that there exists reccomendable songs in closestCustomer
                        closestCustomer = custArray[findCustomer(custArray, (String) custHeap.removeMin().getValue(), 0, size)];//If not, grabs the next one from the heap
                    }
                    int[] songList = reccomendSongs(targetCustomer, closestCustomer);//Saves + Prints the reccomended songs :)
                    System.out.print("ReccomendSongs " +closestCustomer.getName()+" ");
                    for(int i = 0; i < songList.length; i++){
                        System.out.print("song"+ songList[i] + " " + closestCustomer.getRatings()[songList[i]]+" ");
                    }
                    System.out.println();
                    break;

                case "PrintCustomerDistanceRatings":
                    System.out.println("distance customer  s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
                    for(int i = 0; i < size; i++){
                        if(custArray[i].getName().equals(targetCust)){//No distance is printed if the name is the target customer
                            System.out.print("         ");
                        }
                        else if(keyHolder[i] == 0){ //What happens when the distance is not calculatable.
                            System.out.print("-----    ");
                        }
                        else{
                            System.out.printf("%.3f    ", keyHolder[i]);//Prints distance
                        }
                        System.out.print(custArray[i].getName()+" ");//Prints name
                        int nameLength = custArray[i].getName().length();
                        for(int j = 1; j < 10 - nameLength; j++){
                            System.out.print(" ");//Adds appropriate number of spaces
                        }
                        for(int j = 0; j < 10; j++){
                            System.out.print(custArray[i].getRatings()[j]+"  ");//Prints all ratings
                        }
                        System.out.println();//Breakline :)
                    }
                    break;

                default:
                    break;
            }
        }
    }
    public static boolean checkCompatibility(Customer Target, Customer Closest){
        for(int i = 0; i < 10; i++){
            if(Closest.getRatings()[i] >= 4 && Target.getRatings()[i] == 0){//Checks if a) closest likes a song enough b) target hasn't rated it
                return true;
            }
        }
        return false;
    }

    public static double getDistance(int[] A, int[] B){//Calculates the interest "distance" between two customers
        int sharedCount = 0;//Counts number of applicable instances
        double distance = 0;//Holds average distance
        for(int i = 0; i < 10; i++){
            if(A[i] != 0 && B[i] != 0){
                sharedCount++;
                distance += Math.abs(A[i] - B[i]);
            }
        }
        if(sharedCount == 0){//Null output - probably could've been better to have it as -1
            return 0;
        }
        return (1 + distance)/sharedCount;//Reciprocal + average distance
    }

    public static void insertCustomer(Customer[] custArray, int size, Customer newCust){//...do I ever use this method?
        custArray[size + 1] = newCust;
        Arrays.sort(custArray, 0, size + 1);
    }

    public static int findCustomer(Customer[] custArray, String Name, int low, int high){//Implementation of a binary search!
        if(low == high){ //Impossible
            if(custArray[low].getName().equals(Name)){
                return low;
            }
            else{
                return -1;//Error :(
            }
        }
        int mid = (low  + high)/2;
        if(custArray[mid].getName().equals(Name)){
            return mid;
        }
        //Remember: a.compareTo(b) returns a negative number if A is BEFORE B
        else{//Very trivial binary implementation :p
            if(Name.compareTo(custArray[mid].getName()) < 0){// i.e. target is BEFORE current mid
                return findCustomer(custArray, Name, low, mid);
            }
            else{
                return findCustomer(custArray, Name, mid + 1, high);
            }
        }
    }
    public static int[] reccomendSongs(Customer Target, Customer Closest){
        int[] songs = new int[10];//array large enough to hold ALL arrays - will not be the final array
        int[] tarSong = Target.getRatings();
        int[] closeSong = Closest.getRatings();
        int numFound = 0;
        for(int i = 0; i < 10; i++){//Finds songs that qualify and puts them in the storage array
            if(closeSong[i] >= 4 && tarSong[i] == 0){
                songs[numFound] = i;
                numFound++;
            }
        }
        int[] songList = new int[numFound];
        for(int i = 0; i < numFound; i++){//Copies all found songs into an appropriately sized array
            songList[i] = songs[i];
        }
        return songList;
    }
}
