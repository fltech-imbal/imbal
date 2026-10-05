/*
 
  Author: Joseph Venech
  Email: jvenech2025@my.fit.edu
  Course: CSE 2010
  Section: 1
  Description: Reads an organization structure from a file and then stores it in a tree. 
    Then reads queriers from another file and performs them and outputs query and result 
*/

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW3 {

    public static void main(String[] args) throws FileNotFoundException {
        
        // Get the organization file and query file
        String organizationFile = args[0];
        String queryFile = args[1];

        Scanner organizationScanner = new Scanner(new File(organizationFile));
        
        String rootName = organizationScanner.nextLine().trim();
        
        //Create tree
        Tree tree = new Tree(rootName);
        
        // Read organization file and add each employee with appropriate supervisors and subordinates
        while (organizationScanner.hasNextLine()) {
            String line = organizationScanner.nextLine().trim();

            //Skip empty lines
            if (line.length() == 0) {
                continue;
            }

            Scanner lineScanner = new Scanner(line);

            String supervisor = lineScanner.next();
            String subordinate = lineScanner.next();
            
            //Add the subordinate under the supervisor
            tree.addChild(supervisor, subordinate);

            lineScanner.close();
        }

        organizationScanner.close();
        
        //Open queries file
        Scanner queryScanner = new Scanner(new File(queryFile));
        
        //Process each query
        while (queryScanner.hasNextLine()) {
            String line = queryScanner.nextLine().trim();
            
            //Skip empty lines
            if (line.length() == 0) {
                continue;
            }

            Scanner lineScanner = new Scanner(line);

            String query = lineScanner.next();
            
            //Find the direct supervisor of an entity
            if (query.equals("DirectSupervisor")) {

                String entity = lineScanner.next();

                System.out.println("DirectSupervisor " + entity + " " + tree.directSupervisor(entity));
            
                //Find the direct subordinates of an entity
            } else if (query.equals("DirectSubordinates")) {

                String entity = lineScanner.next();

                System.out.println("DirectSubordinates " + entity + " " + tree.directSubordinates(entity));
                
                //Find all supervisors above an entity
            } else if (query.equals("AllSupervisors")) {

                String entity = lineScanner.next();

                System.out.println("AllSupervisors " + entity + " " + tree.allSupervisors(entity));
                
                //Find all subordinates of an entity
            } else if (query.equals("AllSubordinates")) {

                String entity = lineScanner.next();

                System.out.println("AllSubordinates " + entity + " " + tree.allSubordinates(entity));
                
                //Find the total number of supervisors above an entity
            } else if (query.equals("NumberOfAllSupervisors")) {

                String entity = lineScanner.next();

                System.out.println("NumberOfAllSupervisors " + entity + " " + tree.numberOfAllSupervisors(entity));

                //Find the total number of subordinates below an entity
            } else if (query.equals("NumberOfAllSubordinates")) {

                String entity = lineScanner.next();

                System.out.println("NumberOfAllSubordinates " + entity + " " + tree.numberOfAllSubordinates(entity));

                //Check if one entity is a supervisor of another
            } else if (query.equals("IsSupervisor")) {

                String entity = lineScanner.next();
                String supervisor = lineScanner.next();
                String answer;
                
                //Determines if entity is supervisor
                if (tree.isSupervisor(entity, supervisor)) {
                    answer = "yes";
                } else {
                    answer = "no";
                }

                System.out.println("IsSupervisor " + entity + " " + supervisor + " " + answer);

                //Check if one entity is a subordinate of another
            } else if (query.equals("IsSubordinate")) {

                String entity = lineScanner.next();
                String subordinate = lineScanner.next();
                String answer;

                //Determines if entity is subordinate
                if (tree.isSubordinate(entity, subordinate)) {
                    answer = "yes";
                } else {
                    answer = "no";
                }

                System.out.println("IsSubordinate " + entity + " " + subordinate + " " + answer);

                //Compare the ranks of two entities
            } else if (query.equals("CompareRank")) {

                String entity1 = lineScanner.next();
                String entity2 = lineScanner.next();

                System.out.println("CompareRank " + entity1 + " " + entity2 + " " + tree.compareRank(entity1, entity2));
                
                //Find the closest supervisor shared by two entities
            } else if (query.equals("ClosestCommonSupervisor")) {

                String entity1 = lineScanner.next();
                String entity2 = lineScanner.next();

                System.out.println("ClosestCommonSupervisor " + entity1 + " " + entity2 + " " + tree.closestCommonSupervisor(entity1, entity2));
            }

            lineScanner.close();
        }
        
        //Close the query file
        queryScanner.close();
    }
}
