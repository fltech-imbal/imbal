import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList; 
import java.util.Scanner; 
import java.util.Arrays;

public class hw4 {
    //I am working with heaps 
    //I need to make a heap class and use it. 

    

    public static void main(String[] args) {
       

        Scanner kb; 

        try { // a try catch exception for file use
            kb = new Scanner(new File(args[0])); 
        } 
        catch(FileNotFoundException e) {
            System.out.println("File was not found"); 
            return;
        } 

        ArrayList<customer> arrayOfCustomers = new ArrayList<>();

        String targetName = kb.next(); 
        customer targetCustomer = new customer(targetName, null);  
        while (kb.hasNext()) {
            String closestName = kb.next(); 
            int[] ratings = new int[10];
            for (int i = 0; i < ratings.length; i++) {
                int songs = kb.nextInt(); 
                ratings[i] = songs;
            } 
            customer closestCustomer = new customer(closestName, ratings); 
            arrayOfCustomers.add(closestCustomer); 
            
            if(closestName.equals(targetName)) {
                targetCustomer = closestCustomer;
            }

        } 

        kb.close(); 

        

        Scanner scanner; 
        
        try { // a try catch exception for file use
            scanner = new Scanner(new File(args[0])); 
        } 
        catch(FileNotFoundException e) {
            System.out.println("File was not found 2"); 
            return;
        }  

        while (scanner.hasNext()) {
            switch (scanner.next()) {
                case "AddCustomer":
                    System.out.print("AddCustomer "); 
                    String customerName = scanner.next(); 

                    break; 

                case "RecommendSongs": 
                    //make closestCustomer an integer of -1
                    int closestCustomer = -1; 
                    //make the variable double closestDistance
                    double closestDistance = Double.MAX_VALUE; 
                    //In a for loop from int i to arrayOfCustomers.size()
                    for (int i = 0; i < arrayOfCustomers.size(); i++) {
                        double distance = targetCustomer.getDistance(arrayOfCustomers.get(i));
                        if (distance >= 0 && distance < closestDistance) {
                            closestDistance = distance;
                            closestCustomer = i;
                        }
                    }
                    if (closestCustomer >= 0) {
                        targetCustomer.getDistance(arrayOfCustomers.get(closestCustomer)); 
                        String recommended; 
                        recommended = arrayOfCustomers.get(closestCustomer).getName() + " " + targetCustomer.recommendSongs(arrayOfCustomers.get(closestCustomer));
                    } 


                    break;
                case "PrintCustomerDistanceRatings": 
                    //create a string variable 
                    String customerDistanceRatings = ""; 
                    System.out.println("        " + targetCustomer.getName() + " " + Arrays.toString(targetCustomer.getRatings()));
                    //In a for loop: getDistance(), getName(), getRatings() and make them string. 
                    for(int i = 0; i < arrayOfCustomers.size(); i++) { 
                        double distance = targetCustomer.getDistance(arrayOfCustomers.get(i)); 
                        customerDistanceRatings = String.valueOf(distance) + " "; 
                        customerDistanceRatings += arrayOfCustomers.get(i).getName() + " "; 
                        customerDistanceRatings += Arrays.toString(arrayOfCustomers.get(i).getRatings()); 
                        System.out.println(customerDistanceRatings);


                    }


                    break;
                default:
                    break;
            }
        } 





        scanner.close();





        

        

        
    }
}