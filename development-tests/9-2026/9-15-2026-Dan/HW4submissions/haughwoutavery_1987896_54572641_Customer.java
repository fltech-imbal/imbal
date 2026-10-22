/*
Name: Avery Haughwout
Course: CSE2012
Section: 3:30
Description:
 */

public class Customer implements Comparable<Customer>{
    String name;
    int[] ratings = new int[10]; //Maximum of 10 songs
    public Customer(String thisName, int[] thisRatings){
        this.name = thisName;
        this.ratings = thisRatings;
    }

    //Getters
    public String getName(){
        return this.name;
    }
    public int[] getRatings(){
        return this.ratings;
    }
    //Setters
    public void setName(String setName){
        this.name = setName;
    }
    public void setRatings(int[] setRatings){
        this.ratings = setRatings;
    }
    @Override
    public int compareTo(Customer other){
        return this.name.compareTo(other.name);
    }
}