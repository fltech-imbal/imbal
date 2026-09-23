
/**
 * Stores the name of the requester and the time of the initiated request. Designed for HW1.java
 * Avery Haughwout
 * CSE2012
 * 8/27/26
 */
public class Request
{

    static String name;//Customer's name 
    static String time;// better be in the format HHMM or I'm going to riot
    /**
     * Constructor for objects of class Customer
     */
    public Request(String Name, String Time)
    {
        this.name = Name;
        this.time = Time;
    }

    public void setName(String Name){
        this.name = Name;
    }
    public void setTime(String Time){
        this.time = Time;
    }
    public String getName(){
        return Request.name;
    }
    public String getTime(){
        return Request.time;
    }
}
