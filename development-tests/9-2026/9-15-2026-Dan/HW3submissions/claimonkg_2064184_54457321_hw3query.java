import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;

public class hw3query {
    public static Tree supervisor; 

    
     
    
    public static void main(String args[]){
        hw3Tree.main(args); 

        supervisor = hw3Tree.getTree();
        
        Scanner scanner; 

        try { // a try catch exception for file use
            scanner = new Scanner(new File("C:\\Users\\claim\\OneDrive\\Desktop\\java test\\hw3inputQueries1.txt")); 
        } 
        catch(FileNotFoundException e) {
            System.out.println("File was not found"); 
            return;
        } 

        String query = scanner.next(); 

        while (scanner.hasNext()) {
            switch (query) {
                //gets parent
                case "DirectSupervisor": 
                    String entity = scanner.next(); 
                    String directSupervisor = supervisor.getParent(entity); 
                    System.out.print("DirectSupervisor " + entity + " ");
                    if (directSupervisor == null) {
                        System.out.println("none"); 
                    } else { 
                        System.out.println(directSupervisor); 

                    } 
                break; 

                //gets children
                case "DirectSubordinates": 
                    entity = scanner.next(); 
                    String directSubordinates = supervisor.getChildren(entity); 
                    System.out.print("DirectSubordinates " + entity + " ");
                    
                    //System.out.println("[" + supervisor.getChildren(entity) + "]");

                    if (directSubordinates.equals("none")) {
                        System.out.println("none"); 
                    } else { 
                        String[] subordinateArray = directSubordinates.split(" ");  
                        Arrays.sort(subordinateArray);
                        System.out.println(String.join(" ", subordinateArray)); 

                    }  
                break; 

                //gets ancestors (parent of parent of parent of...)
                case "AllSupervisors": 
                    entity = scanner.next(); 
                    String allSupervisors = ""; 
                    System.out.print("AllSupervisors " + entity + " ");
                    while (!supervisor.getParent(entity).equals(null)) {
                        allSupervisors += supervisor.getParent(entity); 
                        entity = supervisor.getParent(entity);
                    }  
                    
                    
                    if (allSupervisors.equals("")) {
                        System.out.println("none"); 
                    } else { 
                        System.out.println(allSupervisors); 

                    } 
                break; 
                
                //gets grandchildren
                case "AllSubordinates": 
                    entity = scanner.next(); 
                    String allSubordinates = ""; 
                    System.out.print("AllSubordinates " + entity + " ");

                    allSubordinates = supervisor.getGrandChildren(supervisor.root.find(entity), "");

                    System.out.println(allSubordinates); 
                break; 

                case "NumberOfAllSupervisors": 
                    entity = scanner.next(); 
                    allSupervisors = "";
                    System.out.print("NumberOfAllSupervisors " + entity + " ");
                    while (supervisor.getParent(entity) != null) {
                        allSupervisors += supervisor.getParent(entity); 
                        entity = supervisor.getParent(entity);
                    }  

                    int number = allSupervisors.split(" ").length;  

                    System.out.println(number); 
                break; 

                case "NumberOfAllSubordinates": 
                    entity = scanner.next(); 
                    allSubordinates = ""; 
                    System.out.print("AllSubordinates " + entity + " ");

                    allSubordinates = supervisor.getGrandChildren(supervisor.root.find(entity), "");

                    number = allSubordinates.split(" ").length;  
                    System.out.println(number); 
                break; 

                case "IsSubordinate": 
                    entity = scanner.next(); 
                    String subordinate = scanner.next();
                    System.out.print("IsSubordinate " + entity + " " + subordinate + " ");
                    if (supervisor.root.find(entity).find(subordinate) != null) {
                        System.out.println("yes");
                    } else {
                        System.out.println("no"); 
                    }  
                break; 

                //case IsSupervisor 
                case "IsSupervisor": 
                    entity = scanner.next(); 
                    String supervisor1 = scanner.next(); 

                    System.out.print("IsSupervisor " + entity + " " + supervisor1 + " " ); 
                    if (supervisor.root.find(supervisor1).find(entity) != null) {
                        System.out.println("yes");
                    } else {
                        System.out.println("no"); 
                    }  
                break; 

                //case CompareRank 
                case "CompareRank": 
                    entity = scanner.next(); 
                    String entity2 = scanner.next();
                    System.out.print("CompareRank " + entity + " " + entity2 + " "); 

                    String result = "same";
                    String current = entity2; 
                    if (entity.equals(entity2)) {
                        result = "same"; 
                    } else {
                    while (supervisor.getParent(current) != null) {
                        current = supervisor.getParent(current); 
                        
                        if (current.equals(entity)) {
                            result = "higher";
                            break;
                        } 
                    } 


                        
                    }
                    
                    if (result.equals("same")) {
                        current = entity; 

                        while (supervisor.getParent(current) != null) {
                            current = supervisor.getParent(current); 
                        
                            if (current.equals(entity2)) {
                                result = "lower";
                                break;
                            }   
                        }
                    } 

                    System.out.println(result); 
                break; 

                //case ClosestCommonSupervisor
                case "ClosestCommonSupervisor": 
                    entity = scanner.next(); 
                    entity2 = scanner.next(); 

                    System.out.print("ClosestCommonSupervisor" + entity + " " + entity2 + " "); 

                    String current1 = entity; 
                    String common = null;

                    while (current1 != null) {
                        String current2 = entity2; 

                        while (current2 != null) {
                            if (current1.equals(current2)) {
                                
                                common = current1; 
                                break;
                            } 
                            current2 = supervisor.getParent(current2);
                        } 

                        if (common != null) {
                            break;
                        }
                        current1 = supervisor.getParent(current1);
                    } 
                    
                    if (common == null) {
                        System.out.println("none");
                    } else {
                        System.out.println(common);
                    } 
                break; 

                default: break;
                


            




                




            }
        }

      scanner.close();   
     }

}
