/*
 * Author: Jackson Young
 * Email: young2025@my.fit.edu
 * Course: CSE 2010
 * Section: 9:30am lab
 * Description of this file: This is the driver class for my Tree data structure. It will take input from
 * two text files using Scanner objects and perform the necessary actions to the tree defined in the first
 * input file as determined by the second input file.
*/
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Scanner;
public class HW3 {
    public static void main(String[] args) {
        // Declare Scanner variables
        Scanner setupScanner;
        Scanner queryScanner;

        // Run a try-catch block to initialize setupScanner to a Scanner object reading a File object
        // and safely terminate the program if an error occurs during the creation of the input file
        try {
            setupScanner = new Scanner(new File(args[0])); // Tree Setup (first file)
            queryScanner = new Scanner(new File(args[1])); // Queries (second file)
        } catch (FileNotFoundException e) {
            System.out.println("FileNotFoundException: " + e + "\nAn Error Occurred With The Input Files.\nTerminating Program.");
            return;
        }

        // Initialize Tree
        Tree<String> root = new Tree(setupScanner.next());

        // Process all remaining input from setupScanner (first input file)
        while (setupScanner.hasNext()) {
            String supervisor = setupScanner.next();
            String subordinate = setupScanner.next();
            root.addChild(supervisor, subordinate);
        }

        // Process all input for queryScanner (second input file)
        while (queryScanner.hasNext()) {
            String action = queryScanner.next();
            // Using the input stored above, choose one of the following actions:

            /************************************\
             * DIRECT Supervisor / Subordinates *
            \************************************/
            if (action.equals("DirectSupervisor")) {
                // Get entity and supervisor
                String entity = queryScanner.next();
                Tree<String> supervisor = root.getParent(entity);

                // Print necessary text and "none" is no supervisor is found
                System.out.print("DirectSupervisor " + entity + " ");
                if (supervisor != null) System.out.println(supervisor.value);
                else System.out.println("none");

            } else if (action.equals("DirectSubordinates")) {
                // Get entity and list of subordinates
                String entity = queryScanner.next();
                ArrayList<Tree<String>> subordinates = root.getChildren(entity);
                // Sort subordinate list alphabetically using a comparator
                subordinates.sort(Comparator.comparing(child -> child.value));

                // Print necessary text and loop through subordinates list to get names
                System.out.print("DirectSubordinates " + entity + " ");
                if (!subordinates.isEmpty()) {
                    for (int i = 0; i < subordinates.size(); i++) {
                        System.out.print(subordinates.get(i).value);
                        if (i != subordinates.size() - 1) System.out.print(" ");
                    }
                    System.out.println(); // Line break
                }
                else System.out.println("none"); // Print "none" if there are no subordinates
            }

            /**********************************\
             * ALL Supervisors / Subordinates *
            \**********************************/
            if (action.equals("AllSupervisors")) {
                // Get entity
                String entity = queryScanner.next();
                // Make a list of supervisors
                ArrayList<Tree<String>> supervisorList = root.getAllParents(root.find(entity), new ArrayList());

                // Print necessary text and loop through supervisor list to get names
                System.out.print("AllSupervisors " + entity + " ");
                if (!supervisorList.isEmpty()) {
                    for (int i = 0; i < supervisorList.size(); i++) {
                        System.out.print(supervisorList.get(i).value);
                        if (i != supervisorList.size() - 1) System.out.print(" ");
                    }
                    System.out.println(); // Line break
                }
                else System.out.println("none"); // Print "none" if there are no supervisors

            } else if (action.equals("AllSubordinates")) {
                // Get entity
                String entity = queryScanner.next();
                // Make a list of subordinates
                ArrayList<Tree<String>> subordinateList = root.getAllChildrenStrings(root.find(entity), new ArrayList());

                // Print necessary text and loop through subordinates list to get names
                System.out.print("AllSubordinates " + entity + " ");
                if (!subordinateList.isEmpty()) {
                    for (int i = 0; i < subordinateList.size(); i++) {
                        System.out.print(subordinateList.get(i).value);
                        if (i != subordinateList.size() - 1) System.out.print(" ");
                    }
                    System.out.println(); // Line break
                }
                else System.out.println("none"); // Print "none" if there are no subordinates
            }

            /********************************************\
             * NUMBER Of All Supervisors / Subordinates *
            \********************************************/
            if (action.equals("NumberOfAllSupervisors")) {
                // Get entity
                String entity = queryScanner.next();
                // Make a list of supervisors
                ArrayList<Tree<String>> supervisorList = root.getAllParents(root.find(entity), new ArrayList());
                // Print necessary text and size of supervisorList
                System.out.println("NumberOfAllSupervisors " + entity + " " + supervisorList.size());

            } else if (action.equals("NumberOfAllSubordinates")) {
                // Get entity
                String entity = queryScanner.next();
                // Make a list of subordinates
                ArrayList<Tree<String>> subordinateList = root.getAllChildren(root.find(entity), new ArrayList());
                // Print necessary text and size of subordinateList
                System.out.println("NumberOfAllSubordinates " + entity + " " + subordinateList.size());
            }

            /*******************************\
             * IS Supervisor / Subordinate *
            \*******************************/
            if (action.equals("IsSupervisor")) {
                // Get entities
                String entity = queryScanner.next();
                String supervisor = queryScanner.next();
                // Print necessary text
                System.out.print("IsSupervisor " + entity + " " + supervisor + " ");
                // Check if supervisor is the parent of entity and print boolean
                if (root.getAllParents(root.find(entity), new ArrayList()).contains(root.find(supervisor))) System.out.println("yes");
                else System.out.println("no");

            } else if (action.equals("IsSubordinate")) {
                // Get entities
                String entity = queryScanner.next();
                String subordinate = queryScanner.next();
                // Print necessary text
                System.out.print("IsSubordinate " + entity + " " + subordinate + " ");
                // Check if subordinate is the parent of entity and print boolean
                if (root.getAllChildren(root.find(entity), new ArrayList()).contains(root.find(subordinate))) System.out.println("yes");
                else System.out.println("no");
            }

            /****************\
             * Compare Rank *
            \****************/
            if (action.equals("CompareRank")) {
                // Get entities
                String entity1 = queryScanner.next();
                String entity2 = queryScanner.next();
                // Print necessary text 
                System.out.print("CompareRank " + entity1 + " " + entity2 + " ");
                // Determine the difference in subtree depth and print
                Tree<String> node1 = root.find(entity1);
                Tree<String> node2 = root.find(entity2);
                if (node1.depth > node2.depth) System.out.println("lower"); // higher depth = lower rank
                else if (node1.depth < node2.depth) System.out.println("higher"); // lower depth = higher rank
                else System.out.println("same"); // equal rank
            }

            /*****************************\
             * Closest Common Supervisor *
            \*****************************/
            if (action.equals("ClosestCommonSupervisor")) {
                // Get entities
                String entity1 = queryScanner.next();
                String entity2 = queryScanner.next();
                // Print necessary text 
                System.out.print("ClosestCommonSupervisor " + entity1 + " " + entity2 + " ");

                // Find and print the closest common supervisor
                // by finding and comparing every supervisor of the two entities
                // until a match is found
                boolean found = false;
                Tree<String> supervisor1 = root.getParent(entity1);
                while (supervisor1 != null) {
                    Tree<String> supervisor2 = root.getParent(entity2);
                    while (supervisor2 != null) {
                        if (supervisor1.equals(supervisor2)) {
                            found = true;
                            break;
                        }
                        // Increment supervisor2 if no commonality found
                        supervisor2 = supervisor2.parent;
                    }
                    // Increment supervisor1 if no commonality found
                    if (found) break;
                    else supervisor1 = supervisor1.parent;
                }
                if (found && supervisor1 != null) System.out.println(supervisor1.value);
                else System.out.println("none");
            }
        }

        // Debug Line
        //root.print(root);
    }
}