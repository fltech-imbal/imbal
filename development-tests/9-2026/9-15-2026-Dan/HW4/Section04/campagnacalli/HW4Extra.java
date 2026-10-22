import java.util.Scanner;
import java.io.File; 
import java.io.FileNotFoundException;


/*
 * Author: Calli Campagna 
 * Email:ccampagna2025@code01.fit.edu
 * Course:CSE2010
 * Section:1-4
 * Description of this file: 
 * This program implements a music recommendation system.It reads customer names and their
 * song ratings from input files,stores customers in alphabetical order, calculates distances
 * between customers based on their shared song ratings, and uses a minimum heap to recommend 
 * songs from the closest customer.It can also print the distances and ratings of all customers.
*/
public class HW4Extra{
    /*
     * This section has an array storing Customer objects in alphabetical order by name.
     * Then the number of customers currently stored in the array and the name of the customer
     * who will receive song recommendations.
     */
    Customer[] customers = new Customer[100];
    int customerCount = 0;
    String targetName;
    /*
     * Description: Reads the customer information and commands from two input files. 
     * Executes each command, including adding customers,recommending songs, and 
     * printing customer distances.
     * Parameter: args(the names of the input files)
     */
    public static void main(String[] args)throws FileNotFoundException{
        /*
         * This creates the HW4 object that allows the methods to work together. The the first input
         * file is read and the target name is set to the first line from the file. 
         */
        HW4Extra homework = new HW4Extra();
        Scanner sc = new Scanner(new File(args[0]));
        homework.targetName = sc.next();
        /*
         * Read each remaining customer from the first file. Each customer has a name 
         * followed by 10 song ratings. Store each customer in alphabetical order using 
         * insertCustomer. The ratings is an array that holds the cutomers ratings for 10 songs
         * and the for loop reas the 10 ratings from the file. Then the customer name and ratings list
         * is sent to the insertCustomer method 
         */
        while(sc.hasNext()){
           String customerName = sc.next();
           int[] ratings = new int[10];
           for (int i = 0; i< 10; i++){
               ratings[i] = sc.nextInt();
           }
           
           homework.insertCustomer(customerName, ratings);
        }
        /*
         * new scanner is created to open the second file. a while loop os made to read every command 
         * in the file and determines what operation the program should perform which the String action 
         * holds the command from the file telling the code what to do
         */
        Scanner scanner = new Scanner(new File(args[1]));
        while(scanner.hasNext()){
            String action = scanner.next();
            /*
             * if the action is AddCustomer reads a new customers name and ratings, 
             * sends the customer name and ratings to insertCustomer. Then prints the AddCustomer command 
             * and its ratings and name 
             */
            if (action.equals("AddCustomer")){
                String custName = scanner.next();
                int[] ratings = new int[10];
                for (int i = 0; i< 10; i++){
                  ratings[i] = scanner.nextInt();
                }
                homework.insertCustomer(custName, ratings);
                System.out.println("AddCustomer " + custName);
                for (int i = 0; i < 10; i++) {
                   System.out.print(" " + ratings[i]);
                }
                System.out.println();
                /* 
                 * if the actions is RecoomendSong it finds the closest customer who has highly rated 
                 * songs that the target customer as not rated
                 */
            }else if (action.equals("RecommendSongs")){
                homework.recommendSongs();
                /*
                 * PrintCustomerDistanceRatings displays the distance between the target 
                 * customer and every other customer, along with their song ratings.
                 */
            }else if(action.equals("PrintCustomerDistanceRatings")){
                homework.printCustomerDistanceRatings();
            }
        }
    }
    /*
     * Description: Searches for a customer by name using binary search. The customers array 
     * must be sorted alphabetically.
     * Parameters: name - the name of the customer to find.
     * Returns: The matching Customer object if found; otherwise, null.
     */
    public Customer findCustomer(String name){
        int low = 0;
        int high = customerCount -1;
        /*
         * Continue searching while there are elements remaining. Compare the middle customers 
         * name with the requested name. If the name is alphabetically earlier or later, 
         * eliminate the half of the array that cannot contain the customer. Return the customer if the 
         * name matches or search either right or left half depending and cotinue through the loop again
         */
        while(low <= high){
            int middle = (low + high)/2;
            int comparison = customers[middle].name.compareTo(name);
            if(comparison ==0){
                return customers[middle];
            }else if (comparison < 0){
                low = middle + 1;
            }else{
                high = middle - 1;
            }
        }
        return null;
    }
    /*
     * Description: Creates a new Customer and inserts it into the customers array while 
     * maintaining alphabetical order.
     * Parameters: name - the name of the customer to insert.
     * ratings - an array with the customers 10 song ratings.
     */
    public void insertCustomer(String name, int[] ratings){
       
        /*
         * creates a cutomer object and sets int value pos(position) to 0
         */
       Customer newCustomer = new Customer(name, ratings);
       int pos = 0;
       /*
        * finds the position where the new customer belongs alphabetically by comparing ot to the other
        * names in the list 
        */
       while (pos < customerCount && customers[pos].name.compareTo(name) < 0){
           pos++;
       }
       /*
        * Shift customers one position to the right to make room. Start at the end of 
        * of the array and move backward so that existing customers are not overwritten. Then place new 
        * cusotmer in the right postion based on for loop nad increase the number of customers in 
        * the array 
        */
       for (int i = customerCount; i > pos; i--){
           customers[i] = customers[i-1];
       }
       customers[pos] = newCustomer;
       customerCount++;
    }
    /*
     * Description: Finds the closest customer who can recommend at least one song that the
     * target customer has not rated. Customers are processed in increasing 
     * distance order using a minimum heap. The first qualifying customer is printed.
     */
    public void recommendSongs() {
       /*
        * find the cutomer who will receive the recommendations. Then created a minimum heap to 
        * organize cutomers by distance 
        */
        Customer target = findCustomer(targetName);

        CustomerMinHeap heap = new CustomerMinHeap(100);
       /*
        * Calculate the distance between the target and every other customer. Ignore the 
        * target customer and customers who have no songs rated in common with the target. Insert 
        * the rest of the customers into the minimum heap. A negative distance in te if stament means
        * there are no shared ratings 
        */
       
        CustomerDistance[] entryArray = new CustomerDistance[100];
        int numEntries = 0;
        for (int i = 0; i < customerCount; i++) {
            
            Customer other = customers[i];
            if (!other.name.equals(targetName)) {
               double distance = calculateDistance(target, other);

               if (distance >= 0) {
                   entryArray[numEntries] = new CustomerDistance(other, distance);

                   numEntries++;
               }
            }
        }
        heap.initialize(entryArray, numEntries);
        /*
        * Remove customers from the heap in increasing distance order. For each customer
        * check whether they rated any song at least 4 that the target customer has not rated.
        * Stop after finding the first customer with a recommendation.
        */
    
        while (!heap.isEmpty()) {
            CustomerDistance minInfo = heap.removeMin();
            Customer closest = minInfo.customer;
            /*
             * the boolean tracks whether this cutomer can reccomend a song and the for loop checks all
             * 10 songs for a recoomendation that could work
             */
            boolean hasRecommendation = false;

        
            for (int song = 0; song < 10; song++) {
               if (target.ratings[song] == 0 && closest.ratings[song] >= 4) {
                  hasRecommendation = true;
                  break;
               }
            }
            /*
             * If the customer has a recommendation, print their name and each qualifying song from
             * the for loop goes with its rating. Stops becasue the closest qualifying customer was found
             * prints none if there are no sutomers that dont wualify to recommend a song 
             */
            if (hasRecommendation) {
                 System.out.print("RecommendSongs " + closest.name);

                 for (int s = 0; s < 10; s++) {
                    if (target.ratings[s] == 0 && closest.ratings[s] >= 4) {
                      System.out.print(" song" + s + " " + closest.ratings[s]);
                   }
                }

                System.out.println();
                return;
            }
        }
       
         System.out.println("RecommendSongs none");
    }

    /*
     * Description: Calculates the distance between two customers using the absolute differences
     * between their ratings for shared songs. The total difference is adjusted by adding 1 and
     * dividing by the number of shared songs.
     * Parameters: customer1 - the first customer.
     * customer2 - the second customer.
     * Returns: The calculated distance, or -1 if the customers have no shared song ratings.
     */
    public double calculateDistance(Customer customer1, Customer customer2) {
        int sharedSongs = 0;
        int totalDiff = 0;
       /*
        * Examine all 10 songs. A rating of 0 means the customer has not rated that song. 
        * Only compare ratings when both customers have rated the song. Add the absolute difference 
        * between the raings and count the song as a shared rating if the if stament is true. Return
        * -1 when there are no shared ratings to compare
        */
        for (int i = 0; i < 10; i++) {
           if (customer1.ratings[i] != 0 && customer2.ratings[i] != 0) {

            totalDiff += Math.abs(customer1.ratings[i] - customer2.ratings[i]);

            sharedSongs++;
           }
        }

       if (sharedSongs == 0) {
          return -1;
       }
       /*
        * calculates and return the distance between the customers
        */
       return (1.0 + totalDiff) / sharedSongs;
    }
    /*
     * Description: Prints the target customer's ratings and the distances and ratings of all other customers.
     */
    public void printCustomerDistanceRatings() {
        /*
         * Find the target cutomer in the sorted customer array and stops of the target cutomer does not 
         * exist 
         */
        Customer target = findCustomer(targetName);

        if (target == null) {
            System.out.println("Customer not found");
            return;
        }
        /*
         * starts with the target customer and stores the foramted distance for the output. Prints the 
         * titles and headings to this format. then prints the labels for the 10 songs. Then it gets into
         * printing the customers name and ratings. The target customers distnace from itself is blank
         * becasue its the target 
         */
    
        Customer current = target;

        String distanceText = "-----";
        System.out.println("PrintCustomerDistanceRatings");
        System.out.printf("%-9s%-11s", "distance ", "customer ");
        for (int j =0; j < 10; j++){
           System.out.printf("%3s", "s" + j);
        }
        System.out.println();
        System.out.printf("%-9s%-11s","", current.name);

        for (int j = 0; j < 10; j++) {
            System.out.printf("%3d", current.ratings[j]);
        }
        System.out.println();

        /*
         * Visit every customer in alphabetical order. Skip the target customer because they were
         * printed already. Calculate each remaining customers distance from the target. Makes sure the target
         * name is not printed twice and then calculates the distance between this cutomer and the target 
         * 
         */
    
        for (int i = 0; i < customerCount; i++) {
            current = customers[i];

            if (current.name.equals(targetName)) {
                continue;
            }

            double distance = calculateDistance(target, current);
            /*
             * displays dahes if there are no shared ratings or it will show the distance to three 
             * decimal places. Then it will print the cutomers disnatce an name. Then the customers ratings 
             * for all 20 songs 
             * 
             */
            if (distance < 0) {
                distanceText = "-----";
            } else {
                distanceText = String.format("%.3f", distance);
            }

            System.out.printf("%-9s%-11s", distanceText, current.name);

            for (int j = 0; j < 10; j++) {
                System.out.printf("%3d", current.ratings[j]);
            }

            System.out.println();
        }
    }

}
/*
 * Description: Stores a customer and the calculated distance between that customer and the target customer.
 */
class CustomerDistance {
    Customer customer;
    double distance;
    /*
     *  Description: Creates a CustomerDistance object.
     *  Parameters: customer - the customer associated with the distance.
     *  distance - the calculated distance.
     */
    public CustomerDistance(Customer customer, double distance) {
        this.customer = customer;
        this.distance = distance;
    }
}

/*
 * Description: creates a minimum heap of CustomerDistance objects. The customer with the 
 * smallest distance is at the root. If two distances are equal, the customer whose name comes 
 * first alphabetically is given priority.
 */
class CustomerMinHeap {
    private CustomerDistance[] heap;
    private int size;
    /*
     * Description: Creates an empty minimum heap with the given capacity.
     * Parameters: capacity - the maximum number of elements the heap can hold.
     */
    public CustomerMinHeap(int capacity) {
        heap = new CustomerDistance[capacity];
        size = 0;
    }
    
    /*
     * Description: Initializes the minimum heap using bottom up heap construction. Copies the entries 
     * into the heap array and moves each non leaf entry down to restore the minimum heap property.
     * Parameters: entryArray - the entries to place in the heap.
     * numEntries - the number of entries in the array.
     */
    public void initialize(CustomerDistance[] entryArray, int numEntries) {
        /*
         * set the number of entries in the heap and then copies eeach entry into
         * the heap array if the heap has enough space. The next for loop starts at the last non leaf node and moves toward the 
         * root. Then restores the heap property for this node and its children using the moveDown method
         */
        size = numEntries;
        
        if (heap.length < numEntries) {
            heap = new CustomerDistance[numEntries];
        }
        
        for (int i = 0; i < numEntries; i++) {
            heap[i] = entryArray[i];
        }

        for (int position = size / 2 - 1; position >= 0; position--) {
            moveDown(position);
        }
    }

    /*
     *  Description: Checks whether the heap contains no elements.
     *  Returns: True if the heap is empty otherwise false.
     */
    public boolean isEmpty() {
        return size == 0;
    }
    /*
     * Description: Checks whether the heap has reached its capacity.
     * Returns: True if the heap is full otherwise false.
     */
    public boolean isFull() {
        return size == heap.length;
    }

    /*
     * Description: Determines which CustomerDistance object should come first in the minimum 
     * heap. Smaller distances have priority.If the distances are equal, names are compared 
     * alphabetically.
     * Parameters: a - the first CustomerDistance object.
     * b - the second CustomerDistance object.
     * Returns: True if a should come before b otherwise false.
     */
    private boolean comesFirst(CustomerDistance a, CustomerDistance b) {
        if (a.distance < b.distance) {
            return true;
        }

        if (a.distance > b.distance) {
            return false;
        }
        /*
         * if the order is tied it goes based on alphabetical order 
         */
        return a.customer.name.compareTo(b.customer.name) < 0;
    }
    /*
     * Description: Inserts a customer and their distance into the heap. The new element is moved 
     * upward until the minimum heap property is back.
     * Parameters: customer - the customer to insert.
     * distance - the distance associated with the customer.

     */
    public void insert(Customer customer, double distance) {
        /*
         * if the heap is full dont insert. Then palce the new element at the next available array 
         * position. Also it will track the pos (position) and the size
         */
        if (isFull()) {
            return;
        }

        heap[size] = new CustomerDistance(customer, distance);
        int pos = size;
        size++;
        /*
         * Compare the new element with its parent. If the new element should come first swap them.
         * Continue upward until the parent has priority or the root of the heap is reached. 
         */
        while (pos > 0) {
            int parent = (pos - 1) / 2;
            /*
             * uses comesFirst method to see if the element should come before its parent in the heap 
             * if not its in position 
             */
            if (comesFirst(heap[pos], heap[parent])) {
                CustomerDistance temp = heap[pos];
                heap[pos] = heap[parent];
                heap[parent] = temp;

                pos = parent;
            } else {
                break;
            }
        }
    }
    /*
     * Description: Returns the minimum element without removing it.
     * Returns: The root CustomerDistance object, or null if empty.
     */
    public CustomerDistance getMin() {
        if (isEmpty()) {
            return null;
        }

        return heap[0];
    }
    /*
     * Description: Removes and returns the minimum element.Moves the last element to the root and 
     * restores the heap using siftDown.
     * Returns: The removed minimum element or null if empty.
     */
    public CustomerDistance removeMin() {
        if (isEmpty()) {
            return null;
        }
        /*
         * save the minumum heat before removing it. then decrease the number of elements in the heap 
         * represented by size. If elements remain, move the last element to the root. Clear the 
         * old last position and restore the heap order. If the heap is now empty simply clear the root.
         */
        CustomerDistance minimum = heap[0];

        size--;

        if (size > 0) {
            heap[0] = heap[size];
            heap[size] = null;
            moveDown(0);
        } else {
            heap[0] = null;
        }
        /*
         * return element that was removed
         */
        return minimum;
    }
    /*
     *     Description: Restores the minimum heap property by moving an element downward until 
     *     it is no larger than its children.
     *     Parameters: position - the array index where the downward adjustment begins.
     */
    private void moveDown(int position) {
        /*
         *  Repeatedly find the smallest among the current element and its children. Swap with 
         *  the smallest child when needed. Stop when the current element is already the smallest.
         */
        while (true) {
            /*
             * calculate the array positions of the two children and initially assume the cirrent element
             * is the smallest
             */
            int left = 2 * position + 1;
            int right = 2 * position + 2;
            int smallest = position;
            /*
             * check whether the left child should come first. Then check for the right child then the 
             * stop if the current element is already the smallest 
             */
            if (left < size && comesFirst(heap[left], heap[smallest])) {
                smallest = left;
            }

            if (right < size && comesFirst(heap[right], heap[smallest])) {
                smallest = right;
            }

            if (smallest == position) {
                break;
            }
            /*
             * swap the current ellemt with its smallest child. contunue from the child previous positions 
             */
            CustomerDistance temp = heap[position];
            heap[position] = heap[smallest];
            heap[smallest] = temp;

            position = smallest;
        }
    }
}
/*
 * Description: Represents a customer in the music recommendation system. Stores the customers 
 * name and their ratings for 10 songs.
 */
class Customer{
    String name;
    int[] ratings;
    /*
     * Description: Creates a Customer object using the given name and song ratings.
     * Parameters: name - the customer's name.
     * ratings - the customer's array of 10 song ratings.

     */
    public Customer(String name, int[] ratings){
        this.name = name;
        this.ratings = ratings;
    }
    
}
