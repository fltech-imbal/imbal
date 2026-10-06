// Name: Adrian Santana
// Email: santanaa2025@my.fit.edu
// Course: CSE 2010
// Description: The program presented below reads two input files and uses methods from Tree.java to output the hierarchical
// structure of the tree.
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.io.File;

public class HW3 {
    public static void main(String[] args) throws FileNotFoundException {
        // Make a scanner object that takes in the data input file
        File dataFile = new File(args[0]);
        Scanner scanner = new Scanner(dataFile);
        String rootData = scanner.nextLine();
        Tree tree = new Tree(rootData);

        // Create a while loop that reads each line in the input file
        while (scanner.hasNextLine()) {

            // Use the split line tool to break each line from the input into individual words
            String[] dataParts = scanner.nextLine().split(" ");
            tree.addChild(dataParts[0], dataParts[1]);
        }

        // Create a scanner object that takes in the queries input file
        File queriesFile = new File(args[1]);
        scanner = new Scanner(queriesFile);

        // Create a while loop that reads each line in the input file
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();

            // Use the line split tool to break each line in the queries input file into individual words
            String[] queriesParts = line.split(" ");
            String infoRequest = queriesParts[0];
            String entity = queriesParts[1];

            // If the request is "DirectSupervisor", use the getParent() method from Tree.java to find the parent and
            // the required message
            if (infoRequest.equals("DirectSupervisor")) {
                String supervisor = tree.getParent(entity);
                if (supervisor == null) {
                    System.out.println(infoRequest + " " + entity + " none");
                } else {
                    System.out.println(infoRequest + " " + entity + " " + supervisor);
                }
                // If the request is "DirectSubordinates", use the getChildren method in Tree.java to find the child
                // and print the message
            } else if (infoRequest.equals("DirectSubordinates")) {
                ArrayList<String> children = tree.getChildren(entity);
                System.out.print(infoRequest + " " + entity);
                if (children.isEmpty()) {
                    System.out.print(" none");
                } else {
                    for (String child : children) {
                        System.out.print(" " + child);
                    }
                }
                System.out.println();

                // If the request is "AllSupervisors", use the getAllSupervisors method in Tree.java to find all supervisors
                // and print message
            } else if (infoRequest.equals("AllSupervisors")) {
                ArrayList<String> supervisors = tree.getAllSupervisors(entity);
                System.out.print(infoRequest + " " + entity);
                if (supervisors.isEmpty()) {
                    System.out.print(" none");
                } else {
                    for (String supervisor : supervisors) {
                        System.out.print(" " + supervisor);
                    }
                }
                System.out.println();

                // If the request is "AllSubordinates", use the getAllSubordinates() method to find all subordinates and
                // print message
            } else if (infoRequest.equals("AllSubordinates")) {
                ArrayList<String> subordinates = tree.getAllSubordinates(entity);
                System.out.print(infoRequest + " " + entity);
                if (subordinates.isEmpty()) {
                    System.out.print(" none");
                } else {
                    for (String subordinate : subordinates) {
                        System.out.print(" " + subordinate);
                    }
                }
                System.out.println();

                // If request is "NumberOfAllSupervisors", use the getNumberOfAllSupervisors method to find all supervisors
                // and print the message
            } else if (infoRequest.equals("NumberOfAllSupervisors")) {
                int count = tree.getNumberOfAllSupervisors(entity);
                System.out.println(infoRequest + " " + entity + " " + count);

                // If request is "NumberOfAllSubordinates", use the getNumberOfAllSubordinates method to find all subordinates
                // and print message
            } else if (infoRequest.equals("NumberOfAllSubordinates")) {

                int count = tree.getNumberOfAllSubordinates(entity);
                System.out.println(infoRequest + " " + entity + " " + count);

                // If the request is "IsSupervisor", use the isSupervisor method to determine if the supervisor is actually
                // a supervisor of the entity and print message
            } else if (infoRequest.equals("IsSupervisor")) {
                if (tree.isSupervisor(queriesParts[1], queriesParts[2])) {
                    System.out.println(line + " yes");
                } else {
                    System.out.println(line + " no");
                }

                // If the request is "IsSubordinate", use the isSubordinate() method to determine if the subordinate is
                // a subordinate of the entity and print the required message
            } else if (infoRequest.equals("IsSubordinate")) {
                if (tree.isSubordinate(queriesParts[1], queriesParts[2])) {
                    System.out.println(line + " yes");
                } else {
                    System.out.println(line + " no");
                }

                // If the request is "CompareRank", use the compareRank method to determine the rank of the entities
                // and print the message
            } else if (infoRequest.equals("CompareRank")) {
                String entity2 = queriesParts[2];
                String rank = tree.compareRank(entity, entity2);
                System.out.println(infoRequest + " " + entity + " " + entity2 + " " + rank);

                // If request is "ClosestCommonSupervisor", use the closestCommonSupervisor method to determine
                // the entities' closest common supervisor and print the required message
            } else if (infoRequest.equals("ClosestCommonSupervisor")) {
                String entity2 = queriesParts[2];
                String closestSupervisor = tree.closestCommonSupervisor(entity, entity2);
                if (closestSupervisor == null) {
                    System.out.println(infoRequest + " " + entity + " " + entity2 + " " + " none");
                } else {
                    System.out.println(infoRequest + " " + entity + " " + entity2 + " " + closestSupervisor);
                }
            }
        }
    }
}