/*
  Author: Matthew Gomez
  Email: matthewgomez2025@fit.edu
  Course: CSE 2010
  Section:
  Description of this file: A Customer stores a name and the ratings
  (0 = not rated, 1-5 = rating) for the 10 songs (song0 - song9).
 */

public class Customer
{
    public static final int NUM_SONGS = 10;

    private String name;     // customer name (at most 10 letters)
    private int[] ratings;   // ratings[i] is the rating of song i

    /*
      Creates a customer.
      name: customer name
      ratings: array of 10 ratings
    */
    public Customer(String name, int[] ratings)
    {
        this.name = name;
        this.ratings = ratings;
    }

    public String getName() { return name; }

    public int getRating(int song) { return ratings[song]; }

    public void setRatings(int[] r) { ratings = r; }
}
