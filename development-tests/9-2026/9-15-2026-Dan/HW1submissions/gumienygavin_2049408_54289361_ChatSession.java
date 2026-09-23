/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file: 
  This file defines the ChatSession class which handles information regarding a chat session
  between a customer and representative. It stores and manages the names of both the customer
  and representative who are involved in a chat session. The class includes getter and
  setter methods for both the customer and representative name. The equals method
  has been overridden to compare sessions by the names of the customer and representative.

 */
public class ChatSession
{
    // Instance variable declarations storing the customer and representative name.
    private String customer;
    private String rep;

    /** Constructor
     * 
     *  @param customer  the name of the customer
     *  @param rep  the name of the representative
     */
    public ChatSession(String customer, String rep)
    {
        this.customer = customer;
        this.rep = rep;
    }

    // Getter methods
    /** Returns the customer's name in the chat session.
     *  @return  the customer's name
     */
    public String getCustomer() {
        return this.customer;
    }
    
    /** Returns the representative's name in the chat session
     *  @return  the representative's name
     */
    public String getRep() {
        return this.rep;
    }
    
    // Setter methods
    /** Sets a new name for the customer variable in the chat session.
     *  @param customer  the new name of the customer
     */
    public void setCustomer(String customer) {
        this.customer = customer;
    }
    
    /** Sets a new name for the representative variable in the chat session.
     *  @param rep  the new name of the representative
     */
    public void setRep(String rep) {
        this.rep = rep;
    }
    
    /** Checks if two chat sessions are equal. They are equal if they have the same customer and representative name.
     *  @param o  the object to be compared with the current object
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) { return true; }  // Returns true if the object is the exact same
        if (o == null) { return false; } // Returns false if the object does not exist
        
        // Checks if the object being compared is of the ChatSession class
        if (o instanceof ChatSession) {
            ChatSession other = (ChatSession) o;
            
            // The names of the customer in both chat sessions (to be compared)
            String thisCustomer = this.getCustomer();
            String otherCustomer = other.getCustomer();
            
            // Checks if both names match
            boolean customersMatch;
            if (thisCustomer == null) {
                customersMatch = (otherCustomer == null);
            }
            else {
                customersMatch = thisCustomer.equals(otherCustomer);
            }
            
            // The names of the representatives in both chat sessions (to be compared)
            String thisRep = this.getRep();
            String otherRep = other.getRep();
            
            // Checks if both reps match
            boolean repsMatch;
            if (thisRep == null) {
                repsMatch = (otherRep == null);
            }
            else {
                repsMatch = thisRep.equals(otherRep);
            }
            
            // Returns true if both customer and representative names match
            return customersMatch && repsMatch;
        }
        
        // If the object to be compared is not a ChatSession object
        return false;
    }
}