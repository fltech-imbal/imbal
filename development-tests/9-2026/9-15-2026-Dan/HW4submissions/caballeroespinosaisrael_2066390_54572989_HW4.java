/*
Author:Israel Caballero
Email:icaballeroes2025@my.fit.edu
Course:CSE2010-Algorithms and Data Structures
Section:01
Description of this file:Program that finds how close customer's tastes are for songs and recommends songs based on how close their tastes are.
*/
import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;
public class HW4{
//creates a heap and list for the customers
static Heap customerHeap = new Heap();
static Customer customers[] = new Customer[100];
static int numCustomers = 0;
//creates a single heap entry with the distance rating as the key and name as the vaule
public static class PQEntry{
    private double k;
    private String v;
    public PQEntry(double key, String value){
        k = key;
        v = value;
    }
    //gets the key of the entry
    public double getKey(){return k;}
    //gets the value of the entry
    public String getValue(){return v;}

    protected void setKey(double key){k = key;}
    protected void setValue(String value){v = value;}
}
//heap class
public static class Heap{
    //creates an arraylist for the heap
    protected ArrayList<PQEntry> heap = new ArrayList<>();
    //returns the parent,left, or right of an entry
    protected int parent(int j){return (j-1)/2;}
    protected int left(int j){return 2*j+1;}
    protected int right(int j){return 2*j+2;}
    //returns if the entry has a left or right entry
    protected boolean hasLeft(int j){return left(j) < heap.size();}
    protected boolean hasRight(int j){return right(j) < heap.size();}
    //Swaps entries at two positions
    protected void swap(int i, int j){
        PQEntry temp = heap.get(i);
        heap.set(i,heap.get(j));
        heap.set(j,temp);
    }
    //If an entry is added at the end of the list and it is smaller than its parent, then it swaps upwards until the heap is in the correct order
    protected void upheap(int j){
        while(j > 0){
            int p = parent(j);
            if(compareTo(heap.get(j),heap.get(p)) >= 0)break;
            swap(j,p);
            j = p;
        }
    }
    //If an entry is removed and placed in the root, it will swap with the smallest child until it cannot.
    protected void downheap(int j){
        while(hasLeft(j)){
            int leftIndex = left(j);
            int smallChildIndex = leftIndex;
            if(hasRight(j)){
                int rightIndex = right(j);
                if(compareTo(heap.get(leftIndex), heap.get(rightIndex)) > 0)
                    smallChildIndex = rightIndex;
            }
            if(compareTo(heap.get(smallChildIndex), heap.get(j)) >= 0)
                break;
            swap(j,smallChildIndex);
            j = smallChildIndex;
        }
    }
    //finds the size of the heap, also checks if the heap is full or empty
    public int size(){return heap.size();}
    public boolean isEmpty(){return heap.size() == 0;}
    public boolean isFull(){return heap.size() == 100;}
    //gets the root of the heap
    public PQEntry getMin(){
        if(heap.isEmpty()) return null;
        return heap.get(0);
    }
    //removes the smallest key of the heap
    public PQEntry removeMin(){
        if(heap.isEmpty()) return null;
        PQEntry answer = heap.get(0);
        swap(0 ,heap.size()-1);
        heap.remove(heap.size()-1);
        downheap(0);
        return answer;
    }
    //compares distances to find out which entry to swap if the distances are the same check the name.
    public int compareTo(PQEntry x, PQEntry y){
        int result;
        result = Double.compare(x.getKey(),y.getKey());
        if (result != 0) return result;
        else return x.getValue().compareTo(y.getValue());
    }
    //inserts a new customer into the heap
    public PQEntry insert(double key, String value){
        PQEntry newest = new PQEntry(key, value);
        heap.add(newest);
        upheap(heap.size()-1);
        return newest;
    }
}
//Creates a customer entry
public static class Customer{
    private String customerName;
    private int[] customerRatings;

    public Customer(String name, int[] ratings){
        customerName = name;
        customerRatings = ratings;
    }

    public String getName(){return customerName;}
    public int getRating(int i){return customerRatings[i];}
}
//Inserts a customer in the array in alphabetical order
public static void insertCustomer(Customer c){
    int index = numCustomers;
    //if c comes before the current i set index to i
    for(int i = 0; i < numCustomers; i++){
        if(c.getName().compareTo(customers[i].getName()) < 0){
            index = i;
            break;
        }
    }
    //shift all elements after and including index to the right one
    for(int j = (numCustomers-1); j >= index; j--){
        customers[j+1] = customers[j];
    }
    //put new customer in index's place
    customers[index] = c;
    numCustomers++;
}
//Finds the index of a customer in the array using binary search
public static int findCustomer(String name, int low, int high){
    //compute the middle of the array
    int mid = (low + high) / 2;
    //Base case if name is not found 
    if(low > high){
        return -1;
    }
    //Base case if the middle element equals the name return the index
    if(customers[mid].getName().equals(name)){return mid;}
    //if the name is on the left side of the list, check low and mid-1 
    else if(name.compareTo(customers[mid].getName()) < 0){
        return findCustomer(name, low, mid-1);
    }
    //check the right side of the list
    else{
        return findCustomer(name, mid+1, high);
    }
}
//Prints customer distance ratings
public static void printCDR(int targetIndex){
    String targetRatings = "";
    //Gathers all targetCustomers ratings into a string
    for(int i = 0; i < 10; i++){
        targetRatings += "  " + customers[targetIndex].getRating(i);
    }
    //Prints headings and the target customers ratings
    System.out.println("PrintCustomerDistanceRatings");
    System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
    System.out.printf("%-9s%-9s%s%n", "", customers[targetIndex].getName(), targetRatings);
    //cycles through all customers
    for(int i = 0; i < numCustomers; i++){
        String ratings = "";
        double distanceRating = 0;
        String distance = "";
        //while the index is not the target customers
        if(!customers[targetIndex].getName().equals(customers[i].getName())){
            //Cycle through the customer in i index and get all ten ratings
            for(int j = 0; j < 10; j++){
                ratings += "  " + customers[i].getRating(j);
            }
            //compute the distance rating of the target customer and the current i customer
            distanceRating = computeDistanceRating(customers[targetIndex], customers[i]);
            //if distance is < 0 distance = nothing, else distance = calculated number
            if(distanceRating < 0){distance = "-----";}
            else{distance = String.format("%.3f", distanceRating);}
            //print the distance as well as the customers name and their ratings
            System.out.printf("%-9s%-9s%s%n", distance, customers[i].getName(), ratings);
        }
    }

}
//Computes distance ratings between two customers
public static double computeDistanceRating(Customer c1, Customer c2){
    int sharedSongs = 0;
    int totalDiff = 0;
    //cycles through both customers ratings for each song
    for(int i = 0; i <= 9; i++){
        int c1Rating = c1.getRating(i);
        int c2Rating = c2.getRating(i);
        //if both customers rated the song incrememnt sharedSongs and calculate the totalDiff
        if((c1Rating > 0) && (c2Rating > 0)){
            sharedSongs++;
            totalDiff += Math.abs(c1Rating - c2Rating);
        }
    }
    //return -1 if no shared songs, else return the distance rating
    if(sharedSongs == 0){return -1;}
    else return ((1.0/sharedSongs) + ((double) totalDiff/sharedSongs));
}
public static void main(String[] args)throws Exception{
    Scanner input = new Scanner(new File(args[0]));
    //Target customer is the first line
    String targetCustomer = input.nextLine();
    //While there are still lines in ratings file
    while(input.hasNextLine()){
        //create ratings list, and splits the line into tokens
        int[] ratings = new int[10];
        String line = input.nextLine();
        String[] token = line.split("\\s+");
        String name = token[0];
        //adds the customers ratings to the ratings list
        for(int i = 1; i <= 10; i++){
            ratings[i-1] = Integer.parseInt(token[i]);
        }
        //inserts customer into the array
        insertCustomer(new Customer(name, ratings));
    }
    //finds the index of the target customer
    int targetIndex = findCustomer(targetCustomer,0,numCustomers-1);
    //cycles through all customers as long as i is not the targets index
    for(int i = 0; i < numCustomers; i++){
        if(i != targetIndex){
            //computes distance ratings of each customer to the target customer
            double distance = computeDistanceRating(customers[targetIndex], customers[i]);
            //if the distance is not -1 then insert the customer into the heap
            if(distance != -1){
                customerHeap.insert(distance, customers[i].getName());
            }
        }
    }
    

    Scanner input2 = new Scanner(new File(args[1]));
    //while there are no lines in the actions file
    while(input2.hasNextLine()){
        //Splits the lines into tokens
        String line = input2.nextLine();
        String[] token = line.split("\\s+");
        //If first word is "PrintCustomerDistanceRatings" call printCDR function
        if(token[0].equals("PrintCustomerDistanceRatings")){
            printCDR(findCustomer(targetCustomer,0,numCustomers-1));
        }
        //If first word is "RecommendSongs"
        if(token[0].equals("RecommendSongs")){
            //creates a temp ArrayList for the customers being checked
            ArrayList<PQEntry> temp = new ArrayList<>();
            String chosenName = "";
            String closeSongs = "";
            //while the heap is not empty
            while(!customerHeap.isEmpty()){
                closeSongs = "";
                //creates a PriorityQueue Entry by taking out the customer from the heap and adding it to the temp list
                PQEntry customer = customerHeap.removeMin();
                temp.add(customer);
                //finds the index of the customer being checked in the array
                int index = findCustomer(customer.getValue(),0,numCustomers-1);
                //cycles through all ratings
                for(int i = 0; i < 10;i++){
                    //if the customers rating of a song is 4 or more and the target customer did not rate it
                    if(customers[index].getRating(i) >= 4 && customers[targetIndex].getRating(i) == 0){
                        //save the customers name and append the recommended songs plus customers rating to closeSongs
                        chosenName = customers[index].getName();
                        closeSongs += " song" + i + " " + customers[index].getRating(i);
                    }
                }
                //if closedSongs is not empty exit the while loop
                if(!closeSongs.equals("")){break;}    
            }
            //If there are no recommended songs from any customers print none
            if(closeSongs.equals("")){System.out.println(token[0] + " none");}
            //else print the customers name and the songs they recommend
            else{
            System.out.println(token[0] + " " + chosenName + closeSongs);
            }
            //add the customers in temp back to the heap.
            for(int i = 0; i< temp.size(); i++){
                customerHeap.insert(temp.get(i).getKey(), temp.get(i).getValue());
            }
        }
        //If the first word is "AddCustomer"
        if(token[0].equals("AddCustomer")){
            int[] ratings = new int[10];
            String name = token[1];
            //collects all the ratings and adds it to a list
            for(int i = 2; i <= 11; i++){
                ratings[i-2] = Integer.parseInt(token[i]);
            }
            //Creates a customer object and adds it to the array
            Customer newCustomer = new Customer(name,ratings);
            insertCustomer(newCustomer);
            //finds the targets index again in case it was changed from the addition
            targetIndex = findCustomer(targetCustomer,0, numCustomers-1);
            //computes distance of the new customer and the target customer
            double distance = computeDistanceRating(customers[targetIndex], newCustomer);
            //inserts customer into the heap if distance != -1
            if(distance != -1){
                customerHeap.insert(distance, newCustomer.getName());
            }
            System.out.println(line);
            }
        }
    }
}