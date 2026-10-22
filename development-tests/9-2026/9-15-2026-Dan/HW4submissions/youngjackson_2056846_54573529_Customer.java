/*
 * Author: Jackson Young
 * Email: young2025@my.fit.edu
 * Course: CSE2010
 * Section: 1
 * Customer class for holding a customer's name and ratings.
*/
public class Customer {

    // Field variables to hold a customer's name and ratings
    String name;
    int[] ratings;

    // Constructors
    public Customer() {
        this.name = "NoName";
        this.ratings = new int[10];
    }
    public Customer(String name, int[] ratings) {
        this.name = name;
        this.ratings = ratings;
    }

    // Public method to calculate the distance value of this customer from another specified customer
    // based on the similarity of their ratings on similar songs.
    // The method takes one parameter: another Customer object to compare this to.
    // The returned value is a double that represents the difference in song ratings
    // If the distance isn't calculatable, the method will return -1
    public double findDistance(Customer other) {
        int sum = 0; // A variable to hold the sum of the rating differences
        int similarities = 0; // A variabe to hold the number of songs rated by both customers
        // Loop through the rating array to evaluate each song's ratings
        for (int i = 0; i < 10; i++) {
            // Check if this song has been rated by both customers
            if (this.ratings[i] != 0 && other.ratings[i] != 0) {
                // Mark that a similarity has been found
                similarities++;
                // Find the difference of the ratings of the ith song
                int difference = Math.abs(this.ratings[i] - other.ratings[i]);
                // Add the difference to the sum
                sum += difference;
            }
        }
        
        // Check if distance is calculatable (don't divide by 0)
        if (similarities == 0) return -1; 
        // Divide the sum by the number of songs compared
        // then add the reciprocal of the similarly rated songs and return the result
        return (double) (1 + sum) / similarities;
    }

    // Overridden toString method that returns the name of the customer
    @Override
    public String toString() {
        return name;
    }
}