/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file: 
  This file defined the Customer class which handles information regarding customers and music
  they may like. It stores and manages the customer name and the list of ratings they have for
  each song. The class includes both getter and setter methods for both the customer name and
  list of song ratings.

 */

import java.util.ArrayList;
import java.util.List;

public class Customer
{
    // Instance variable declarations storing the name of the customer and an array list of their ratings for each song.
    private String name;
    private List<Integer> ratings = new ArrayList<>();
    
    /** Constructor
     * 
     *  @param name  the name of the customer
     *  @param ratings  the customer's list of ratings
     */
    public Customer(String name, List<Integer> ratings) {
        this.name = name;
        this.ratings = ratings;
    }
    
    // Getter methods
    /** Returns the name of the customer.
     *  @return  the customer's name
     */
    public String getCustomerName() {
        return this.name;
    }
    
    /** Returns the list of song ratings for a customer.
     *  @return  the ratings
     */
    public List<Integer> getRatings() {
        return this.ratings;
    }
    
    // Setter methods
    /** Sets a new name for the customer.
     *  @param name  the name of the customer
     */
    public void setCustomerName(String name) {
        this.name = name;
    }
    
    /** Sets a new list of song ratings for the customer.
     *  @param ratings  the ratings
     */
    public void setRatings(List<Integer> ratings) {
        this.ratings = ratings;
    }
    
    /** Checks if two customers are equal. They are equal if they have the same customer name.
     *  @param o  the object to be compared with the current object
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) { return true; }  // Returns true if the object is the exact same
        if (o == null) { return false; } // Returns false if the object does not exist
        
        // Checks if the object being compared is of the CustomerOnHold class
        if (o instanceof Customer) {
            Customer other = (Customer) o;
            
            // The names of both customers (to be compared)
            String thisName = this.getCustomerName();
            String otherName = other.getCustomerName();
            
            // Checks if the current name exists
            if (thisName == null) {
                return otherName == null;
            }
            
            // Returns true if both names match
            return thisName.equals(otherName);
        }
        
        // If the object to be compared is not a CustomerOnHold object
        return false;
    }
}