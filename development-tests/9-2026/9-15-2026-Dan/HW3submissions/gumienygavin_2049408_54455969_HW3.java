/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
  This file reads in a tree and outputs information regarding the tree given a list of
  queries. It takes two input file names as a command-line argument and reads each input
  line within each file. The first file is the data for the tree. It reads the first line,
  stores the root, and then reads each line afterwards to assign each subordinate (child) beneath
  its supervisor (parent). The second file is the list of queries to be called. 
  
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW3
{
    /** The main method (entry point) for the program. Adds the five representatives to their assigned list and reads inputs from a given file.
     * 
     *  @param args  an array of Strings passed into the program (in this case a two String elements, the data input file and the queries input file.).
     *  @throws FileNotFoundException  if the attempt to access the file through args fails.
     */
    public static void main(String[] args)
    {
        try {
            // Checks if there were any arguments passed into the program
            if (args.length < 2) {
                throw new FileNotFoundException("Incorrect file entry! Enter data file first followed by the queries file.");
            }
            
            // Scanner variable to read the tree data.
            Scanner scnrData = new Scanner(new File(args[0]));
            
            // Declare the tree
            Tree tree = new Tree();
            
            // Read the first line and store the root.
            if (scnrData.hasNextLine()) {
                String line = scnrData.nextLine();
                if (line != null && !line.trim().isEmpty()) {
                    tree.addRoot(line.trim());
                }
            }
            
            // Read each line from the tree data input file.
            while (scnrData.hasNextLine()) {
                // Puts an entire line into a string variable
                String line = scnrData.nextLine().trim();
                if (line.isEmpty()) continue;       // Nothing to read
                
                // Reads the text stored in each line
                Scanner lineScanner = new Scanner(line);
                
                // If the first words in a line exists, the word (the supervisor name)
                // is read and stored.
                if (lineScanner.hasNext()) {
                    // Stores the supervisor name
                    String supervisorName = lineScanner.next();
                    
                    // If the second word in a line exists, the word (the subordinate name)
                    // is read and stored.
                    if (lineScanner.hasNext()) {
                        // Stores the subordinate name
                        String subordinateName = lineScanner.next();
                                            
                        // Adds the child to the tree
                        tree.addChild(supervisorName, subordinateName);
                    }
                }
                lineScanner.close();
            }
            // Close first scanner.
            scnrData.close();
            
            // Open second scanner and read the tree query.
            Scanner scnrQuery = new Scanner(new File(args[1]));
            
            // Reads each line from the tree query input file.
            while (scnrQuery.hasNextLine()) {
                // Puts an entire line into a single variable.
                String line = scnrQuery.nextLine();
                if (line.isEmpty()) continue;       // Nothing to read
                
                // Reads the text stored in each line.
                Scanner lineScanner = new Scanner(line);
                
                // If the first word in a line exists, the word (the query)
                // is read and stored.
                if (lineScanner.hasNext()) {
                    // Stores the current query.
                    String query = lineScanner.next();
                    
                    // If the query to be called is "DirectSupervisor," the parameter entity is read and stored.
                    // Then the method directSupervisor is called.
                    if (query.equals("DirectSupervisor")) {
                        String entity = lineScanner.next();
                        tree.directSupervisor(entity);
                    }
                    // If the query to be called is "DirectSubordinates," the parameter entity is read and stored.
                    // Then the method directSubordinates is called.
                    else if (query.equals("DirectSubordinates")) {
                        String entity = lineScanner.next();
                        tree.directSubordinates(entity);
                    }
                    // If the query to be called is "AllSupervisors," the parameter entity is read and stored.
                    // Then the method allSupervisors is called.
                    else if (query.equals("AllSupervisors")) {
                        String entity = lineScanner.next();
                        tree.allSupervisors(entity);
                    }
                    // If the query to be called is "AllSubordinates," the parameter entity is read and stored.
                    // Then the method allSubordinates is called.
                    else if (query.equals("AllSubordinates")) {
                        String entity = lineScanner.next();
                        tree.allSubordinates(entity);
                    }
                    // If the query to be called is "NumberOfAllSupervisors," the parameter entity is read and stored.
                    // Then the method numberOfAllSupervisors is called.
                    else if (query.equals("NumberOfAllSupervisors")) {
                        String entity = lineScanner.next();
                        tree.numberOfAllSupervisors(entity);
                    }
                    // If the query to be called is "NumberOfAllSubordinates," the parameter entity is read and stored.
                    // Then the method numberOfAllSubordinates is called.
                    else if (query.equals("NumberOfAllSubordinates")) {
                        String entity = lineScanner.next();
                        tree.numberOfAllSubordinates(entity);
                    }
                    // If the query to be called is "IsSupervisor," the parameter entity and supervisor are read and stored.
                    // Then the method isSupervisor is called.
                    else if (query.equals("IsSupervisor")) {
                        String entity = lineScanner.next();
                        String supervisor = lineScanner.next();
                        tree.isSupervisor(entity, supervisor);
                    }
                    // If the query to be called is "IsSubordinate," the parameter entity and subordinate are read and stored.
                    // Then the method isSubordinate is called.
                    else if (query.equals("IsSubordinate")) {
                        String entity = lineScanner.next();
                        String subordinate = lineScanner.next();
                        tree.isSubordinate(entity, subordinate);
                    }
                    // If the query to be called is "CompareRank," the parameters entityOne and entityTwo are read and stored.
                    // Then the method compareRank is called.
                    else if (query.equals("CompareRank")) {
                        String entityOne = lineScanner.next();
                        String entityTwo = lineScanner.next();
                        tree.compareRank(entityOne, entityTwo);
                    }
                    // If the query to be called is "ClosestCommonSupervisor," the parameters entityOne and entityTwo are read and stored.
                    // Then the method closestCommonSupervisor is called.
                    else if (query.equals("ClosestCommonSupervisor")) {
                        String entityOne = lineScanner.next();
                        String entityTwo = lineScanner.next();
                        tree.closestCommonSupervisor(entityOne, entityTwo);
                    }
                }
                lineScanner.close();
            }
            // Close second scanner.
            scnrQuery.close();
        }
        // An exception that catches if the file to be read is not found.
        catch (FileNotFoundException ex){
            System.err.println("Error: File not found. " + ex.getMessage());
        }
    }
}
