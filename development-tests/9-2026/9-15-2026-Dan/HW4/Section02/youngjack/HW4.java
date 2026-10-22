/*
Author: Jack Young
Email: Jack2025@my.fit.edu
Course: Algorithms & Data Struct
Section: 01
Description of this file: A program that uses customer data to recommend music to new customers based on comparisons of a calculated rating.
*/
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.StringTokenizer;
public class HW4
{
    static Customer[] customers = new Customer[100];
    static int customerCount = 0;
    static String targetName;

    static class Customer
    {
        String name;
        int[] ratings = new int[10];

        Customer(String n, int[] r)
        {
            name = n;
            for (int i = 0; i < 10; i++)
            {
                ratings[i] = r[i];
            }
        }
    }

    static class DistanceKey implements Comparable<DistanceKey>
    {
        double distance;
        String name;

        DistanceKey(double d, String n)
        {
            distance = d;
            name = n;
        }

        public int compareTo(DistanceKey other)
        {
            if (distance < other.distance)
            {
                return -1;
            }
            if (distance > other.distance)
            {
                return 1;
            }
            return name.compareTo(other.name);
        }
    }

    public static int findCustomer(String name)
    {
        int low = 0;
        int high = customerCount - 1;

        while (low <= high)
        {
            int middle = (low + high) / 2;
            int result = customers[middle].name.compareTo(name);

            if (result == 0)
            {
                return middle;
            }
            else if (result < 0)
            {
                low = middle + 1;
            }
            else
            {
                high = middle - 1;
            }
        }
        return -1;
    }

    public static void insertCustomer(String name, int[] ratings)
    {
        int position = 0;
        while (position < customerCount && customers[position].name.compareTo(name) < 0)
        {
            position++;
        }

        if (position < customerCount && customers[position].name.equals(name))
        {
            customers[position] = new Customer(name, ratings);
        }
        else
        {
            for (int i = customerCount; i > position; i--)
            {
                customers[i] = customers[i - 1];
            }
            customers[position] = new Customer(name, ratings);
            customerCount++;
        }
    }

    public static Double getDistance(Customer first, Customer second)
    {
        int shared = 0;
        int difference = 0;

        for (int i = 0; i < 10; i++)
        {
            if (first.ratings[i] != 0 && second.ratings[i] != 0)
            {
                shared++;
                difference += Math.abs(first.ratings[i] - second.ratings[i]);
            }
        }

        if (shared == 0)
        {
            return null;
        }
        return 1.0 / shared + (double) difference / shared;
    }

    public static void printCustomerDistanceRatings()
    {
        System.out.println("PrintCustomerDistanceRatings");
        System.out.println("distance customer   s0 s1 s2 s3 s4 s5 s6 s7 s8 s9");

        int targetIndex = findCustomer(targetName);
        Customer target = customers[targetIndex];
    
        System.out.printf("%9s%-10s", "", target.name);
        for (int j = 0; j < 10; j++)
        {
            System.out.printf("%2d ", target.ratings[j]);
        }
        System.out.println();

        for (int i = 0; i < customerCount; i++)
        {
            if (!customers[i].name.equals(targetName))
            {
                Double distance = getDistance(target, customers[i]);
                if (distance == null)
                {
                    System.out.printf("%-5s    %-10s", "-----", customers[i].name);
                }
                else
                {
                    System.out.printf("%-5.3f    %-10s", distance, customers[i].name);
                }
                for (int j = 0; j < 10; j++)
                {
                    System.out.printf("%2d ", customers[i].ratings[j]);
                }
                System.out.println();
            }
        }
    }

    public static void recommendSongs()
    {
        Customer target = customers[findCustomer(targetName)];
        HeapPriorityQueue<DistanceKey, Customer> queue =
            new HeapPriorityQueue<DistanceKey, Customer>();

        for (int i = 0; i < customerCount; i++)
        {
            if (!customers[i].name.equals(targetName))
            {
                Double distance = getDistance(target, customers[i]);
                if (distance != null)
                {
                    queue.insert(new DistanceKey(distance, customers[i].name),
                                 customers[i]);
                }
            }
        }

        while (!queue.isEmpty())
        {
            Customer closest = queue.removeMin().getValue();
            boolean hasSong = false;

            for (int i = 0; i < 10; i++)
            {
                if (target.ratings[i] == 0 && closest.ratings[i] >= 4)
                {
                    hasSong = true;
                }
            }

            if (hasSong)
            {
                System.out.print("RecommendSongs " + closest.name);

                for (int i = 0; i < 10; i++)
                {
                    if (target.ratings[i] == 0 && closest.ratings[i] >= 4)
                    {
                        System.out.print(" song" + i + " " + closest.ratings[i]);
                    }
                }
                System.out.println();
                return;
            }
        }

        System.out.println("RecommendSongs none");
    }

    public static void main(String[] args) throws Exception
    {
        BufferedReader input = new BufferedReader(new FileReader(args[0]));
        targetName = input.readLine().trim();
        String line = input.readLine();

        while (line != null)
        {
            StringTokenizer parts = new StringTokenizer(line);
            String name = parts.nextToken();
            int[] ratings = new int[10];

            for (int i = 0; i < 10; i++)
            {
                ratings[i] = Integer.parseInt(parts.nextToken());
            }
            insertCustomer(name, ratings);
            line = input.readLine();
        }

        BufferedReader actions = new BufferedReader(new FileReader(args[1]));
        line = actions.readLine();

        while (line != null)
        {
            StringTokenizer parts = new StringTokenizer(line);
            String action = parts.nextToken();

            if (action.equals("AddCustomer"))
            {
                String name = parts.nextToken();
                int[] ratings = new int[10];

                for (int i = 0; i < 10; i++)
                {
                    ratings[i] = Integer.parseInt(parts.nextToken());
                }
                insertCustomer(name, ratings);
                System.out.println(line);
            }
            else if (action.equals("RecommendSongs"))
            {
                recommendSongs();
            }
            else if (action.equals("PrintCustomerDistanceRatings"))
            {
                printCustomerDistanceRatings();
            }
            line = actions.readLine();
        }
    }
}