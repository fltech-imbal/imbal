/*
  Author: Parker Blue
  Email: pblue2025@my.fit.edu
  Course: CSE 2010
  Section: 001
  Description of this file: The point of this java file is to provide a music recommendation to a user 
    based on another user's similar music taste. This starts by reading an intitial ratings file
    and an actions file. For RecommendSongs, customers whose distance to the target can be computed is put
    into a heap-based priority queue keyed by distance. Customers are removed from the queue in order until
    one has liked songs that the target user has not yet rated. Those songs are then recommended.

    distance (x,y) = 1/|S| + (1/|S|) * sum_{s in S} |r_x(s) - r_y(s)|
    S = songs rated by both x and y
 */
 
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
 
public class HW4
{
    static final int MAX_CUSTOMERS = 100;
    static final int NUM_SONGS = 10;
    static final int LIKE_THRESHOLD = 4;
 
    // customers are sorted alphabetically by name.
    static Customer[] customers = new Customer[MAX_CUSTOMERS];
    static int numCustomers = 0;
    // the name of the customer who the song is being recommended.
    static String targetName;

    //  A customer: name plus 10 song ratings (0 = not rated, 1-5 otherwise).
    static class Customer
    {
        String name;
        int[] ratings;
 
        Customer(String name, int[] ratings)
        {
            this.name = name;
            this.ratings = ratings;
        }
    }
 
    /*
      Priority queue (min-heap) of (distance, customerName) entries.
      Extends the textbook HeapPriorityQueue and overrides compare() so that ties in distance are broken 
      alphabetically by their name. Adds getMin(), isFull() and a capacity check on insert.
    */
    static class CustomerPQ extends HeapPriorityQueue<Double, String>
    {
        static final double EPS = 1e-9;   // tolerance for equal distances
        private final int capacity;
 
        CustomerPQ(int capacity)
        {
            super();
            this.capacity = capacity;
        }
 
        // first compares by distance and will then compare by name
        @Override
        protected int compare(Entry<Double, String> a, Entry<Double, String> b)
        {
            double diff = a.getKey() - b.getKey();
            if (diff < -EPS) return -1;
            if (diff > EPS) return 1;
            return a.getValue().compareTo(b.getValue());
        }
 
        // insert (distance, name): this will fail if the queue is full
        @Override
        public Entry<Double, String> insert(Double key, String value)
        {
            if (isFull())
                throw new IllegalStateException("Priority queue is full");
            return super.insert(key, value);
        }
 
        // the entry with the smallest distance that isn't already removed, or null
        public Entry<Double, String> getMin()
        {
            return min();
        }
 
        // this is true of the queue holds capacity entries
        public boolean isFull()
        {
            return size() >= capacity;
        }
    }
 
    /*
      Inserts a customer while keeping the array in alphabetical order. If the name already exists, its
      ratings are replaced.
      name: customer name; ratings: 10 song ratings
    */
    static void insertCustomer(String name, int[] ratings)
    {
        int existing = findCustomer(name);
        if (existing >= 0)
        {
            customers[existing].ratings = ratings;
            return;
        }
        if (numCustomers >= MAX_CUSTOMERS)
            return;
 
        // shifts larger names one slot to the right, and then drops the new ones in
        int i = numCustomers - 1;
        while (i >= 0 && customers[i].name.compareTo(name) > 0)
        {
            customers[i + 1] = customers[i];
            i--;
        }
        customers[i + 1] = new Customer(name, ratings);
        numCustomers++;
    }
 
    /*
      Does a binary search for a customer by name. Returns the index in customers[], returns -1 though if
      not found.
    */
    static int findCustomer(String name)
    {
        int low = 0, high = numCustomers - 1;
        while (low <= high)
        {
            int mid = (low + high) / 2;
            int cmp = customers[mid].name.compareTo(name);
            if (cmp == 0) return mid;
            if (cmp < 0) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }
 
    
    //  The distance between customers x and y, and then returns -1 when they have no songs rated in common.
    static double distance(Customer x, Customer y)
    {
        int common = 0;
        int sumDiff = 0;
        for (int s = 0; s < NUM_SONGS; s++)
        {
            if (x.ratings[s] != 0 && y.ratings[s] != 0)
            {
                common++;
                sumDiff += Math.abs(x.ratings[s] - y.ratings[s]);
            }
        }
        if (common == 0)
            return -1;
        return (1.0 + sumDiff) / common;
    }
 
    /*
      Prints one row from the table: distance, name, ratings
      distText: text for the distance column
    */
    static void printRow(String distText, Customer c)
    {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-9s%-11s", distText, c.name));
        for (int s = 0; s < NUM_SONGS; s++)
        {
            if (s > 0) sb.append("  ");
            sb.append(c.ratings[s]);
        }
        System.out.println(sb);
    }
 
    /*
      Will print the target customer first, then all of the others alphabetically, each with its distance 
      to the target.
    */
    static void printCustomerDistanceRatings()
    {
        System.out.println("PrintCustomerDistanceRatings");
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
 
        Customer target = customers[findCustomer(targetName)];
        printRow("", target);
 
        for (int i = 0; i < numCustomers; i++)
        {
            Customer c = customers[i];
            if (c == target) continue;
            double d = distance(target, c);
            printRow(d < 0 ? "-----" : String.format("%.3f", d), c);
        }
    }
 
    /*
      Will build a priority queue of distances to the target, and then removes the closest customers one at
      a time until one has liked songs, that are greater than or equal to 4, the target has not rated.
      Prints those songs, or none.
    */
    static void recommendSongs()
    {
        Customer target = customers[findCustomer(targetName)];
 
        // will fill the priority queue with every customer whose distance exists.
        CustomerPQ pq = new CustomerPQ(MAX_CUSTOMERS);
        for (int i = 0; i < numCustomers; i++)
        {
            Customer c = customers[i];
            if (c == target) continue;
            double d = distance(target, c);
            if (d >= 0)
                pq.insert(d, c.name);
        }
 
        // tries customers from closest to farthest
        while (!pq.isEmpty())
        {
            Entry<Double, String> closest = pq.removeMin();
            Customer c = customers[findCustomer(closest.getValue())];
 
            StringBuilder songs = new StringBuilder();
            for (int s = 0; s < NUM_SONGS; s++)
            {
                if (target.ratings[s] == 0 && c.ratings[s] >= LIKE_THRESHOLD)
                    songs.append(" song").append(s).append(' ').append(c.ratings[s]);
            }
            if (songs.length() > 0)
            {
                System.out.println("RecommendSongs " + c.name + songs);
                return;
            }
        }
        System.out.println("RecommendSongs none");
    }
 
    //  Reads 10 ratings from tokens starting at index start
    static int[] parseRatings(String[] tokens, int start)
    {
        int[] ratings = new int[NUM_SONGS];
        for (int s = 0; s < NUM_SONGS; s++)
            ratings[s] = Integer.parseInt(tokens[start + s]);
        return ratings;
    }
 
    // args[0]: initial ratings file; args[1]: actions file
    public static void main(String[] args) throws FileNotFoundException
    {
        // reads the target name and initial customers
        Scanner ratingsIn = new Scanner(new File(args[0]));
        targetName = ratingsIn.nextLine().trim();
        while (ratingsIn.hasNextLine())
        {
            String line = ratingsIn.nextLine().trim();
            if (line.isEmpty()) continue;
            String[] tokens = line.split("\\s+");
            insertCustomer(tokens[0], parseRatings(tokens, 1));
        }
        ratingsIn.close();
 
        // processes each action, echoing it as required
        Scanner actionsIn = new Scanner(new File(args[1]));
        while (actionsIn.hasNextLine())
        {
            String line = actionsIn.nextLine().trim();
            if (line.isEmpty()) continue;
            String[] tokens = line.split("\\s+");
 
            if (tokens[0].equals("AddCustomer"))
            {
                int[] ratings = parseRatings(tokens, 2);
                insertCustomer(tokens[1], ratings);
                StringBuilder sb = new StringBuilder("AddCustomer " + tokens[1]);
                for (int r : ratings) sb.append(' ').append(r);
                System.out.println(sb);
            }
            else if (tokens[0].equals("RecommendSongs"))
            {
                recommendSongs();
            }
            else if (tokens[0].equals("PrintCustomerDistanceRatings"))
            {
                printCustomerDistanceRatings();
            }
        }
        actionsIn.close();
    }
}
