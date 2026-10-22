/*

  Author: Matthew Gomez
  Email: matthewgomez2025@fit.edu
  Course: CSE 2010
  Section:
  Description of this file: Recommends songs to a target customer using
  the ratings of the customer with the closest taste. Customers are kept
  in an alphabetically sorted array; a heap priority queue (CustomerPQ)
  finds the closest customer. Usage: java HW4 ratingsFile actionsFile

 */

import java.io.*;
import java.util.*;

public class HW4
{
    static final int MAX_CUSTOMERS = 100;
    static Customer[] customers = new Customer[MAX_CUSTOMERS]; // sorted by name
    static int numCustomers = 0;
    static String target;                                      // target customer name

    /*
      Inserts a customer so the array stays in alphabetical order.
      name: customer name, ratings: the 10 song ratings
      If the name already exists, its ratings are replaced.
    */
    static void insertCustomer(String name, int[] ratings)
    {
        int pos = findCustomer(name);
        if (pos >= 0) { customers[pos].setRatings(ratings); return; }
        if (numCustomers >= MAX_CUSTOMERS) return;
        int i = numCustomers - 1;
        while (i >= 0 && customers[i].getName().compareTo(name) > 0)
        {
            customers[i + 1] = customers[i];   // shift right
            i--;
        }
        customers[i + 1] = new Customer(name, ratings);
        numCustomers++;
    }

    /*
      Binary search for a customer by name.
      Returns the index in the array, or -1 if not found.
    */
    static int findCustomer(String name)
    {
        int lo = 0, hi = numCustomers - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) / 2;
            int c = customers[mid].getName().compareTo(name);
            if (c == 0) return mid;
            if (c < 0) lo = mid + 1; else hi = mid - 1;
        }
        return -1;
    }

    /*
      Distance between customers x and y: (1 + sum|rx - ry|) / |S|, where
      S is the set of songs rated by both. Returns -1 if S is empty.
    */
    static double distance(Customer x, Customer y)
    {
        int count = 0, sum = 0;
        for (int s = 0; s < Customer.NUM_SONGS; s++)
        {
            if (x.getRating(s) > 0 && y.getRating(s) > 0)
            {
                count++;
                sum += Math.abs(x.getRating(s) - y.getRating(s));
            }
        }
        if (count == 0) return -1;
        return (1.0 + sum) / count;
    }

    /*
      Builds the recommendation text for the target customer: puts every
      other customer in the priority queue, then removes customers from
      closest to farthest until one has a liked (rating >= 4) song the
      target has not rated. Returns "none" if nobody qualifies,
      otherwise "name song1 rating1 song2 rating2 ...".
    */
    static String recommend()
    {
        Customer t = customers[findCustomer(target)];
        CustomerPQ pq = new CustomerPQ();
        for (int i = 0; i < numCustomers; i++)
        {
            if (customers[i] == t) continue;
            double d = distance(t, customers[i]);
            if (d >= 0 && !pq.isFull()) pq.insert(d, customers[i].getName());
        }
        while (!pq.isEmpty())
        {
            String name = pq.removeMin().getValue();
            Customer c = customers[findCustomer(name)];
            StringBuilder sb = new StringBuilder(name);
            boolean found = false;
            for (int s = 0; s < Customer.NUM_SONGS; s++)
            {
                if (t.getRating(s) == 0 && c.getRating(s) >= 4)
                {
                    sb.append(" song").append(s).append(" ").append(c.getRating(s));
                    found = true;
                }
            }
            if (found) return sb.toString();
        }
        return "none";
    }

    /* Prints one table row: distance text, name, then the 10 ratings. */
    static void printRow(String dist, Customer c)
    {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%-9s", dist));        // distance starts at column 1
        sb.append(String.format("%-11s", c.getName())); // name starts at column 10
        for (int s = 0; s < Customer.NUM_SONGS; s++)   // s0 starts at column 21
        {
            if (s > 0) sb.append("  ");
            sb.append(c.getRating(s));
        }
        System.out.println(sb);
    }

    /* Prints the header, the target customer first, then the others alphabetically. */
    static void printTable()
    {
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");
        Customer t = customers[findCustomer(target)];
        printRow("", t);
        for (int i = 0; i < numCustomers; i++)
        {
            if (customers[i] == t) continue;
            double d = distance(t, customers[i]);
            printRow(d < 0 ? "-----" : String.format("%.3f", d), customers[i]);
        }
    }

    /* Parses 10 ratings from tokens[start..start+9]. */
    static int[] parseRatings(String[] tokens, int start)
    {
        int[] r = new int[Customer.NUM_SONGS];
        for (int i = 0; i < Customer.NUM_SONGS; i++)
            r[i] = Integer.parseInt(tokens[start + i]);
        return r;
    }

    /*
      args[0]: file of initial ratings (first line = target customer name)
      args[1]: file of actions
    */
    public static void main(String[] args) throws IOException
    {
        /* read the ratings file: first line is the target, then customers */
        BufferedReader in = new BufferedReader(new FileReader(args[0]));
        target = in.readLine().trim();
        String line;
        while ((line = in.readLine()) != null)
        {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] tk = line.split("\\s+");
            insertCustomer(tk[0], parseRatings(tk, 1));
        }
        in.close();

        /* read the actions file and perform each action, printing its output */
        in = new BufferedReader(new FileReader(args[1]));
        while ((line = in.readLine()) != null)
        {
            line = line.trim();
            if (line.isEmpty()) continue;
            String[] tk = line.split("\\s+");
            if (tk[0].equals("AddCustomer"))
            {
                insertCustomer(tk[1], parseRatings(tk, 2));
                System.out.println(String.join(" ", tk));
            }
            else if (tk[0].equals("RecommendSongs"))
            {
                System.out.println("RecommendSongs " + recommend());
            }
            else if (tk[0].equals("PrintCustomerDistanceRatings"))
            {
                System.out.println("PrintCustomerDistanceRatings");
                printTable();
            }
        }
        in.close();
    }
}
