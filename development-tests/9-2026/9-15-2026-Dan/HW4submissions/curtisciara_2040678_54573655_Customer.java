/* Author: Ciara Curtis
   Email: ccurtis2025@my.fit.edu
   Course: CSE2010
   Section: 4
   Description of this file: Customer class that hold the name
       of the name of the customer and an integer array of song 
       ratings. When compared to another Customer object, it can
       return a distance rating between the two customer objects. */

public class Customer {

  //Class Variables
  private String name;		//holds customer name
  private double distance;
  private int[] songRatings;    //holds customer songs

  //constructor that instantiates the class variable name
  //and the class integer array
  public Customer(String custName, int[] songs) {
    this.name = custName;
    this.songRatings = songs;
    this.distance = 0;
  }

  //empty contructor
  public Customer() {		}
  //-------------End of Constructors--------------//

  public String getName() { return this.name; }

  public double getDistance() { return this.distance; }

  public int[] getRatings() { return this.songRatings; }

  public int getRating(int index) { return this.songRatings[index]; }
  //-------------End of Get Methods---------------//

  public void setName(String custName) { this.name = custName; }

  public void setRatings(int[] ratings) { this.songRatings = ratings;}
  //-------------End of Set Methods---------------//
  
  /*Function that calculates and returns a distance between two
    customer objects by adding up each subtraction of two
    corresponding ratings from both customers, then multiplies
    the summation by one over the number of songs in the 
    songRatings integer array. Then one over the number of
    songs is added to the total and returned. 
    */
  public double setDistance(Customer otherCustomer) {
    double ratingSummation = 0;
    double bothRated = 0;
    for(int i = 0; i < 10; i++) {
      if(this.getRating(i) > 0 && otherCustomer.getRating(i) > 0) {
	double ratingSubtracted = this.getRating(i) - otherCustomer.getRating(i);
      	if(ratingSubtracted < 0) ratingSubtracted *= -1;
      	ratingSummation += ratingSubtracted;
	bothRated++;
      }
    }

    if(bothRated == 0) {
      this.distance = 5000;
      return this.distance;
    }

    this.distance = (1.0/bothRated + ((1.0/bothRated)*ratingSummation));
    return this.distance;
  }

}
