/*
author: Sureena Singh
email: sureena2025@my.fit.edu
course: CSE2010 section 3
description: Song recommendation algorithm that uses a priority queue implemented with a heap to recommend songs with the least distance 
between two customers. 
*/

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
public class HW4{
    /*we use an array customers that stores data of type Customer(name, ratings). */
    public static class Customer{
        String name;
        int[] ratings = new int[10];
        Customer(String name, int[]ratings){
            this.name = name;
            this.ratings= ratings;
        }
    }
    public static class HeapPQ{
         public static class PQ{
             double distance;
             String cust;
             
        
         public PQ(double distance, String cust){
             this.distance = distance;
             this.cust = cust;
         }
     }
    
    
     ArrayList <PQ> heap = new ArrayList<>();

     protected int parent(int j){
         return (j-1)/2;
     } 
     protected int left(int j) {
         return 2*j+1;
     }
     protected int right(int j) {
         return 2*j+2;
     }
     protected boolean hasLeft(int j) {
         return left(j)<heap.size();
     }
     protected boolean hasRight(int j) {
         return right(j)<heap.size();
     }
     public int size(){return heap.size();}
     public void swap(int i, int j){
         PQ temp = heap.get(i);
         heap.set(i, heap.get(j));
         heap.set(j,temp);
     }/* we use bottom-up heap construction here by repeatedly downheaping every element that is added to the heap. */
     public void initialize(PQ entryArray[], int numEntries){
        heap.clear();
        for(int i=0; i<numEntries;i++){
            heap.add(entryArray[i]);
        }for (int i= parent(heap.size()-1);i>=0;i--) downheap(i);
     }
     public void upheap(int j){
        while (j>0){
            int p = parent(j);
            if (compare(heap.get(j),heap.get(p))>=0) break;
            swap (j,p);
            j=p;
        }
     }
     /*We check if the children of j are smaller than j and if that is true than j is swapped with the smallest child. */
     public void downheap(int j){
        while (hasLeft(j)){
            int leftIndex = left(j);
            int smallChildIndex = leftIndex;
            if(hasRight(j)){
                int rightIndex = right(j);
                if (compare(heap.get(rightIndex),heap.get(leftIndex))<0){
                    smallChildIndex= rightIndex;
                }
            }
            if (compare(heap.get(smallChildIndex),heap.get(j))>=0) break;
            swap(j,smallChildIndex);
            j= smallChildIndex;
        }
     }
     /* We use a helper method to compare the distances  */
     protected  int compare (PQ a, PQ b){
        int comp = Double.compare(a.distance, b.distance);
        if (comp != 0){
            return comp;
        }
        return a.cust.compareTo(b.cust);
     }
     /* We simply add the element at the end of the heap and then upheap it so it gets placed at the right index such that it 
     retains the properties of a heap. */
     public void insert(double dist, String cust){
        PQ entry = new PQ(dist,cust);
        heap.add(entry);
        upheap(heap.size()-1);
     }
     public void insert(PQ entry){ 
        heap.add(entry);
        upheap(heap.size()-1);
     }/*
     The removeMin method comes handy in finding the closest customer */
     public PQ removeMin(){ 
        if (isEmpty()){ return null;}
        PQ min = heap.get(0);
        PQ last = heap.remove(heap.size()-1);
        if (!heap.isEmpty()){
            heap.set(0,last);
            downheap(0);
        }return min;
     }
     public PQ getMin(){ 
         if (heap.isEmpty()) return null; 
         else return heap.get(0);
     }
     public boolean isFull(){
        return heap.size()>=100;
     }
     public boolean isEmpty(){
         return heap.isEmpty();
     }
    }
    static Customer[] customers = new Customer[100];
    static int numCustomers=0;
    static String targetName;
    /* We run the while loop if the name already stored in the array is greater than the element
    we want to insert and shift the elements in the array while that condition stands true.
    as soon as the condition becomes false we insert the new customer at the next index. */
    public static void insertCustomer(String name, int[] ratings){
        int i = numCustomers-1;
        while (i>=0 && customers[i].name.compareTo(name)>0){
            customers[i+1]=customers[i];
            i--;
        }customers[i+1]= new Customer(name, ratings);
        numCustomers++;

    }
    /*we use the binary search algoritgm to find Customers in the customers array.
    We create a variable called comparision that stores the values that are 0, greater than 0 ,
    or less than 0 and based on this we decide which part of the array we need to look at. */
    public static Customer findCustomer(String name){
        int high = numCustomers-1;
        int low = 0;
        while (high>=low){
            int mid = low+(high-low)/2;
            int comparison = customers[mid].name.compareTo(name);
            if (comparison==0) return customers[mid];
            else if (comparison<0){
                low = mid+1;
            }
            else {
                high = mid-1;
            }
        } return null;
        
    }/* We use a for loop to count the number of common songs rated and to calculate the sum of the difference in ratings.
    we finally apply the formula at the end and the distnace is returned. */
    public static double distance(Customer x, Customer y){
        int s=0;
        int sum=0;
        int[] cust_1= x.ratings;
        int[] cust_2= y.ratings;
        for (int i=0; i<10;i++){
           if ((cust_1[i]!=0)&&(cust_2[i]!=0)){
            s+=1;
            sum += Math.abs(cust_1[i]-cust_2[i]);
           }
        }
        if (s==0) return -1;
        return 1.0/s+ (double) sum/s;
    }
    public static void recommendedSongs(){
        Customer target = findCustomer(targetName);
        HeapPQ pq = new HeapPQ();
        //build a heap for all customers except the target
        for (int i =0; i< numCustomers;i++){
            Customer c = customers[i];
            if (c.name.equals(targetName)) continue;
            double d = distance(target,c);
            if (d>=0) pq.insert(d,c.name);
        }
        //checking thr closest customers to recommend songs
        while (!pq.isEmpty()){
            HeapPQ.PQ best = pq.removeMin();
            Customer closest = findCustomer(best.cust);
            String songs="";
            for (int s=0;s<10; s++){
                if(closest.ratings[s]>=4 &&target.ratings[s]==0){
                    songs+= " song"+s+" "+closest.ratings[s];
                }
            }
            if (songs.length()>0){
                System.out.println("RecommendSongs "+closest.name+songs);
                return;
            }
        }System.out.println("RecommendSongs none");
    }/* We print the output row by row using a helper method print row that formats and prints each row of a particular customer.
    We use a loop in the printCustomerDistanceRatings to ensure eacj customer's name, distance and ratings are printed */
    public static void printCustomerDistanceRatings(){
        Customer target = findCustomer(targetName);
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
        printRow("",target);
        for(int i=0; i<numCustomers;i++){
            Customer c = customers[i];
            if(c.name.equals(targetName)) continue;
            double d = distance(target,c);
            if (d<0) printRow("-----", c);
            else printRow(String.format("%.3f",d),c);
        }
    }
    public static void printRow(String dist,Customer c){
        System.out.print(String.format("%-9s%-11s", dist, c.name));
        for (int s=0;s<10;s++){
            if (s>0) System.out.print("  ");
            System.out.print(c.ratings[s]);
        }System.out.println();
    }
    public static void main(String[] args) {
        if (args.length<2){
            System.out.println("provide the ratings file and the actions file: ");
            return;
        } 
        try {
            Scanner ratingsScanner = new Scanner(new File(args[0]));
            boolean first = true;
            while(ratingsScanner.hasNextLine()){
                String line = ratingsScanner.nextLine().trim();
                if (line.isEmpty()) continue;
                String[] tokens = line.split("\\s+");
                if (first){
                    targetName = tokens[0];
                    first = false;
                    continue;
                }
                int [] ratings = new int[10];
                for (int i =0; i<10;i++){
                    ratings[i]= Integer.parseInt(tokens[i+1]);
                }
                insertCustomer(tokens[0], ratings);
            }ratingsScanner.close();
            Scanner actionScanner = new Scanner(new File(args[1]));
            while(actionScanner.hasNextLine()){
                String line = actionScanner.nextLine().trim();
                if (line.isEmpty()) continue;
                String[] tokens = line.split("\\s+");
                switch (tokens[0]){
                    case "AddCustomer":
                        int [] ratings= new int[10];
                        for (int i =0; i<10;i++){
                            ratings[i]= Integer.parseInt(tokens[i+2]);
                        }
                        insertCustomer(tokens[1], ratings);
                        System.out.println(line);
                        break;
                    case "RecommendSongs":
                        recommendedSongs();
                        break;
                    case "PrintCustomerDistanceRatings":
                        System.out.println("PrintCustomerDistanceRatings");
                        printCustomerDistanceRatings();
                        break;
                    default:
                        break;
                }
            }
            actionScanner.close();
        } catch (FileNotFoundException e) {
            System.out.println("File not found");
        }
    }
}