import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException; 

//Author: Keidgiler Claimon 
//email: kclaimon2025@fit.edu 
//Course: CSE 2010 
//Section: 01 
//Description: A customer waiting line used to document the amount of time people waited for before they got a session or quit.

public class NEWERHW1 {
    
    public static class CustomerInfoObject { //a class called CustomerInfoObject, which will hold the startTime, the representative, and the customer
        public String startTime; 
        public String representative; 
        public String customer; 

        public CustomerInfoObject(String startTime, String representative, String customer) { //the parameter for the class
           this.startTime = startTime; 
           this.representative = representative; 
           this.customer = customer; 
        }  

        public String toString() { //This is used to return the string of a customer
            return customer;
        }

        @Override 
        public boolean equals(Object obj) { //the Object obj holds the data of the customer, so this will be changed in the function
            if (this == obj) { return true; } //to make sure that obj is correct when being pulled

            if (obj == null || getClass() != obj.getClass()) { //checking for obj is different from its class.
                return false;
            } 

            CustomerInfoObject other = (CustomerInfoObject) obj; //changes other into the object being used

            return this.customer.equals(other.customer); //compares the customer to see if they are the same
        }
    }
    private static SlinkedList<String> availableRepresentatives = new SlinkedList<String>(); //available representatives
    //The next two linkedLists need to be changed into CustomerInfoObject to hold the important parameters
    private static SlinkedList<CustomerInfoObject> customersOnHold = new SlinkedList<CustomerInfoObject>(); 
    private static SlinkedList<CustomerInfoObject> chatSessions = new SlinkedList<CustomerInfoObject>(); 
    
    public static int maxWaitTime = 0;

    private static void addRepresentatives() { //adds all representatives to the class before the main program starts
      availableRepresentatives.addFirst("Alice"); 
      availableRepresentatives.addLast("Bob"); 
      availableRepresentatives.addLast("Carol"); 
      availableRepresentatives.addLast("David"); 
      availableRepresentatives.addLast("Emily"); 
      
    }  

    public static int timeToMinutes(String requestTime) {//this changes time into minutes
        int time = Integer.parseInt(requestTime);//parseInt is used to change requestTime into an integer.
        int hours = time/100; 
        int minutes = time%100; 
        return hours*60+minutes;
    } 

    public static String minutesToTime(int minutesSinceMidnight) {//changes minutes to time
        int hours = minutesSinceMidnight/60; 
        int minutes = minutesSinceMidnight%60; 
        String hourString = Integer.toString(hours); 
        String minuteString = Integer.toString(minutes); 
        if (hourString.length() == 1) { //an if statement used to add a "0" in front of a single digit number
            hourString = "0" + hourString;
        } 
        if (minuteString.length() == 1) { //same thing
            minuteString = "0" + minuteString;
        } 
        return hourString + minuteString; //both of these combine into the time string in the terminal
    }


    public static void main(String[] args) {
        addRepresentatives(); //adds representatives at the start of the program
        
        Scanner scanner;
        
        int maxWaitTime = 0; //Will be used later for comparisons
        
        
        try { // a try catch exception for file use
            scanner = new Scanner(new File(args[0])); 
        } 
        catch(FileNotFoundException e) {
            System.out.println("File was not found"); 
            return;
        } 
        System.out.println(); 
        //holdCustomerInfo will be used to hold similar variables later
        CustomerInfoObject holdCustomerInfo = new CustomerInfoObject(null, null, null); 
        int differenceInTime = 0; //Will be used to hold a subtraction statement
        while(scanner.hasNext()) { //Every single case that will be gone over will be under "while" for unlimited use
            

            String ChatRequest = scanner.next(); //This will be used to check for the cases 

            switch(ChatRequest) { 
                
                case "ChatRequest": String requestTime = scanner.next(); //When ChatRequest is called.
                String customer = scanner.next(); 
                String  waitOrLater = scanner.next(); 

                System.out.println(ChatRequest + " " + requestTime + " " + customer + " " + waitOrLater); 
                if (availableRepresentatives.isEmpty()) { //first this checks if there are no representatives

                    if(waitOrLater.equals("wait")) { //will check for "wait"

                        System.out.println("PutOnHold" + " " + customer + " " + requestTime);
                        //customerAndTime will hold the customer and requestTime
                        CustomerInfoObject customerAndTime = new CustomerInfoObject(requestTime, null, customer);
                        customersOnHold.addLast(customerAndTime); 
                        


                    } else if (waitOrLater.equals("later")) { //checks for "later"

                        System.out.println("TryLater" + " " + customer + " " + requestTime); 

                    }
                    
                
                } else {
                    
                    String representative = availableRepresentatives.first(); //makes representative equal to the first one 
                    availableRepresentatives.removeFirst(); //removes the first representative to move the list
                    System.out.println("RepAssignment" + " " + customer + " " + representative + " " + requestTime); 
                    CustomerInfoObject session = new CustomerInfoObject(requestTime, representative, customer); //session will hold all 3 parameters
                    chatSessions.addLast(session);
                } 

                
                break;  

                case "QuitOnHold": //add all the methods onto the string
                    String quitOnHoldTime = scanner.next(); 
                    customer = scanner.next(); 
                    System.out.println("QuitOnHold" + " " + quitOnHoldTime + " " + customer); 
                    CustomerInfoObject quittingCustomer = new CustomerInfoObject(null, null, customer);  //quittingCustomer only has parameter customer
                    
                    if (customersOnHold.isEmpty()) {
                        System.out.println("No more customers!");
                    } else {
                        holdCustomerInfo = customersOnHold.remove(quittingCustomer); //holdCustomerInfo takes the quittingCustomer
                        
                    }
                    
                    //This takes the subtraction between the times in minutes
                    differenceInTime = timeToMinutes(quitOnHoldTime) - timeToMinutes(holdCustomerInfo.startTime);
                    
                    //this updates the maxWaitTime by comparing the total
                    if(differenceInTime > maxWaitTime) {
                        maxWaitTime = differenceInTime;
                    } 

                
                break;  

                case "ChatEnded": //when a chat session ends
                    
                    customer = scanner.next();
                    String representative = scanner.next();
                    requestTime = scanner.next();
                    CustomerInfoObject difference = new CustomerInfoObject(requestTime, representative, customer);
                    chatSessions.remove(difference);
                    System.out.println("ChatEnded" + " " + customer + " " + representative + " " + requestTime); 
                    
                    if(customersOnHold.isEmpty()) {
                        availableRepresentatives.addLast(representative);
                    } else {
                        //someone is on hold 
                        //representative is free 
                        
                        //requestTime - Original requestTime 
                        CustomerInfoObject holdCustomerInfo2 = customersOnHold.removeFirst();
                        //Update maxWaitTime by subtracting requestTime and quitOnHoldTime and comparing them 
                        differenceInTime = timeToMinutes(requestTime) - timeToMinutes(holdCustomerInfo2.startTime);
                        if(differenceInTime > maxWaitTime) {
                            maxWaitTime = differenceInTime;
                        } 
                        CustomerInfoObject session = new CustomerInfoObject(requestTime, representative, holdCustomerInfo2.customer); //session holds all parameters 
                        chatSessions.addLast(session); //added to chatSessions
                        System.out.println("RepAssignment" + " " + holdCustomerInfo2.customer + " " + representative + " " + requestTime);
                    } 
                    break; 

                    case "PrintAvailableRepList": //when AvailableRepList needs to be called.
                        requestTime = scanner.next();
                        if (availableRepresentatives.isEmpty() == false) { //this makes sure it can show all the representatives
                            System.out.println("AvailableRepList" + " " + requestTime + availableRepresentatives.toString()); 
                        } else {
                            System.out.println("AvailableRepList" + " " + requestTime);
                        }
                        
                    break; 

                    case "PrintMaxWaitTime": //when maxWaitTime needs to be called 
                        requestTime = scanner.next(); 
                        //make variable for maxWaitTime declared higher 
                         
                         
                        String maxWaitTimeSoFar = minutesToTime(maxWaitTime); //transfers maxWaitTime to maxWaitTimeSoFar to have a safer variable
                        
                         
                        //print out maxWaitTime requestTime maxWaitTimeSoFar 
                        System.out.println("PrintMaxWaitTime" + " " + requestTime + " " + maxWaitTimeSoFar); //prints maxWaitTimeSoFar and requestTime
                        
                        
                    break;
                    

                
                default: break;
                

            } 
        }
        scanner.close();
        
    }
}
