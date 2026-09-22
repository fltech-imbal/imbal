/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file: 
  This file defines the CustomerOnHold class which handles information regarding a customer being
  put on hold. It stores and manages the customer name and time they were put on hold. The class
  includes getter and setter methods for both the customer name and request time. The
  equals method has been overridden to compare customers on hold by the names of the customer.

 */
public class CustomerOnHold
{
    // Instance variable declarations storing the time a customer requested a chat session and the name of the customer.
    private int requestTime;
    private String name;
    
    /** Constructor
     * 
     *  @param requestTime  the time a customer requested a chat session
     *  @param name  the name of the customer
     */
    public CustomerOnHold(int requestTime, String name) {
        this.requestTime = requestTime;
        this.name = name;
    }
    
    // Getter methods
    /** Returns the time a customer requested a chat session.
     *  @return  the request time
     */
    public int getRequestTime() {
        return this.requestTime;
    }
    
    /** Returns the name of the customer.
     *  @return  the customer's name
     */
    public String getCustomerName() {
        return this.name;
    }
    
    // Setter methods
    /** Sets a new request time for the customer on hold.
     *  @param requestTime  the time a customer requested a chat session
     */
    public void setRequestTime(int requestTime) {
        this.requestTime = requestTime;
    }
    
    /** Sets a new name for the customer on hold.
     *  @param name  the name of the customer on hold
     */
    public void setCustomerName(String name) {
        this.name = name;
    }
    
    /** Checks if two customers on hold are equal. They are equal if they have the same customer name.
     *  @param o  the object to be compared with the current object
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) { return true; }  // Returns true if the object is the exact same
        if (o == null) { return false; } // Returns false if the object does not exist
        
        // Checks if the object being compared is of the CustomerOnHold class
        if (o instanceof CustomerOnHold) {
            CustomerOnHold other = (CustomerOnHold) o;
            
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