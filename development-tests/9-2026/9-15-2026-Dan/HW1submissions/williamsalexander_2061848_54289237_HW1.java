/*

  Author: Alexander Williams
  Email: alexanderwil2025@my.fit.edu
  Course: CSE2010
  Section: 
  Description of this file:

 */
import java.util.Scanner;
import java.io.File;
public class HW1
{

    /*
      Description of each method, including parameters 
    */
    static class Customer {  
    
    String name;
    int requestTime;
    
    Customer(string name, int requestTime){  //stores the name of the customer and the request time as a string and int
    this.name = name;
    this.requestTime = requestTime;
    }
}
    static class Representative {

    String name;
   
    Representative(string name) { //stores representative's name as a string
    this.name = name;
    }
}  
    static class chatSession {
    
    Customer customer;
    Representative representative;

    chatSession(Customer customer, Representative representative) { //connects a customer to a representative
    
    this.customer = customer;
    this.representative = representative;
}
    public static void main(String[] args) throws Exception
    {
        while (input.hasNextLine()) {  //loops while there is another line in the file;
        String line = input.nextLine; 
        String[] parts = line.split(" ");  //separates event line
        String event type = parts[0];
        if (eventType.equals("ChatRequest")) {
           int requestTime = Integer.parseInt([parts[1]);
           String customerName = parts[2];
           String waitOrLater = parts [3];

           Customer customer = new Customer(customerName, requestTime);

           if (!availableReps.isEmpty())
           {
             Representative rep = availableReps.removeFirst();
         
             ChatSession = session = new ChatSession(customer, rep);
             chatSessions.addLast(session);
           }
            else if (waitOrLater.equals("wait")) 
           {
            customerrsOnHold.addLast(customer);
           }  
            else if (waitOrLater.equals("later")) 
           {
            System.out.println("TryLater " + customerName + " " + requestTime);
          }
        if (eventType.equals("ChatEnded"))
          {
            String customerName = parts[1];
            String repName = parts[2];
            int endTime = Integer.parseInt(parts[3]);
           }
        }

        Scanner = new Scanner(new File(inputFile));  //opens the input file

        String inputFile = args[0];  //stores the input file name
   


	/* description of variables */
        SinglyLinkedList<Representative> availableReps =  //available representatives
        new SinglyLinkedList<>();

        SinglyLinkedList<Customer> customerOnHold = //customers on hold
        new SinglyLinkedList<>();       

        SinglyLinkedList<chatSession> chatSession = //customers in chat sessions
        new SinglyLinkedList<>();

	/* description of each block (around 5-10 lines) of instructions */
        //adds the names to the list of available representatives
        availableReps.addLast(new Representative("Alice"));
        availableReps.addLast(new Representative("Bob"));        
        availableReps.addLast(new Representative("Carol"));
        availableReps.addLast(new Representative("David"));
        availableReps.addLast(new Representative("Emily"));
    }

}
