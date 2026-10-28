// Author: Adrian Santana
// Email: santanaa2025@my.fit.edu
// Course: CSE 2010
// Section: 1
// Description: The class shown below represents a customer with their name and their ratings for each song and is used in
// HW4.java to create customer objects.

public class Customer {
    // Create a String variable to store the customer's name
    private String name;
    // Create an array to store each customer's ratings for the songs
    private int[] ratings;
    // Make a constructor that takes in the customer's name (String name) and ratings (int[] ratings) and initializes them
    public Customer(String name, int[] ratings) {
        this.name = name;
        this.ratings = ratings;
    }
    // Use getter methods to access the customer names and ratings
    public String getName() {
        return name;
    }
    public int getRatings(int songIndex) {
        return ratings[songIndex];
    }
}
