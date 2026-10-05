/*
Author: Jerye Demers-St.Hilaire
Email: jdemerssthil2025@my.fit.edu
Course: CSE2010
Section: 1
Description of this file: Organizing input based on rank and answer input questions based on organized input 
*/

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.LinkedList;

public class HW3 {
 public static void main(String[] args) throws FileNotFoundException {

 // Scanner for reading the first input file 
 Scanner scanner = new Scanner(new File(args[0]));

   int LineNum = 0;
   String topEntity = " ";

   Node top = null;
   Node supervisor = null;
   Node subordinate = null;

   LinkedList<Node> allNodes  = new LinkedList<>();

   Tree tree = new Tree();

 // Looping through the first input file until there is no more lines to read      
 while (scanner.hasNextLine()) {
   String input = scanner.nextLine();
   
 // The first line is always the top supervisor 
 if (LineNum == 0) {
    topEntity = input;
    top = new Node(topEntity);
    allNodes.add(top);
 }
  
 // If its not the first line split the supervisor and subordinate
 // Check if they already exist
 else {
    String[] inputSplit = input.split(" ");
    String suP = inputSplit[0];
    String suB = inputSplit[1];  
    Node duplicate = null;
    Node duplicateChild = null;
      
      // Looping through allNodes to find existing supervisor 
      // If supervisor already exsists store existing node
      for(int i = 0; i < allNodes.size(); i++) { 
       if (suP.equals(allNodes.get(i).name)) {
        duplicate = allNodes.get(i);
        supervisor = duplicate;
       }
      }  
      
      // Looping through allNodes fo find existing subordinate 
      // If subordinate already exsists store existing node
      for (int j = 0; j < allNodes.size(); j++) {
       if (suB.equals(allNodes.get(j).name)) {
        duplicateChild = allNodes.get(j);
        subordinate = duplicateChild;
       }   
      }
        
       // If supervisor is new, create and add it  
       if (duplicate == null) {
        supervisor = new Node(suP); 
        allNodes.add(supervisor);
       }
       
       // If subordinate is new, create and add it
       if (duplicateChild == null) {
        subordinate = new Node(suB);
        allNodes.add(subordinate);
       }
       
       // Add supervisor and subordinate to the tree
       tree.addChild(supervisor, subordinate);
     }
   LineNum++;
 }
 
   // Scanner to read the query file                     
   Scanner scannerQueries = new Scanner(new File(args[1]));
  
   // Looping through the second input file until there are no more lines to read
   while (scannerQueries.hasNextLine()) {
   
    // Storing and seperating the query input
    String inputQueries = scannerQueries.nextLine(); 
    String[] queriesInputSplit = inputQueries.split(" ");
     
     // Checking if the first input word is DirectSupervisor   
     if (queriesInputSplit[0].equals("DirectSupervisor")) {
       
       Node queryNode = null; 
       
        // Looping through allNodes to find input
        for (int i = 0; i < allNodes.size(); i++) {
         if (queriesInputSplit[1].equals(allNodes.get(i).name)) {
          queryNode = allNodes.get(i);
          break;
         }
        } 
        
         if (queryNode == null) {
          continue;
         }
     
      Node supervisorNode = tree.getParent(queryNode);
      
      // If there is no direct supervisor, output none
      // If there is a direct supervisor, output the direct supervisor
      if (supervisorNode == null) {
       System.out.println("DirectSupervisor " + queriesInputSplit[1] + " none"); 
      }
      else {
       System.out.println("DirectSupervisor " + queriesInputSplit[1] + " " + supervisorNode.name);
      }
     
     }
      
     // Checking if the first input word is DirectSubordinates 
     else if (queriesInputSplit[0].equals("DirectSubordinates")) {
      
      Node queryNode = null;
      
       // Looping through allNodes to find the input
       for (int i = 0; i < allNodes.size(); i++) {
        if (queriesInputSplit[1].equals(allNodes.get(i).name)) {
         queryNode = allNodes.get(i);
         break;
        }
       }
       
       if (queryNode == null) { 
        continue;
       }  
      
      // Getting a list of input supervisors subordinates
      LinkedList<Node> subordinatesNode = tree.getChildren(queryNode);
      
      // If there are subordinates output them
      // If there are no subordinates, output none
      if (!subordinatesNode.isEmpty()) {
      
      System.out.print("DirectSubordinates " + queriesInputSplit[1] + " ");
      
       for (int i = 0; i < subordinatesNode.size(); i++) {
        System.out.print(subordinatesNode.get(i).name + " ");
       }
       System.out.println();
      } 
      else {
       System.out.println("DirectSubordinates " + queriesInputSplit[1] + " none"); 
      }
     
     }
     
     // Checking if the first word is AllSupervisors or NumberOfAllSupervisors
     else if (queriesInputSplit[0].equals("AllSupervisors") || queriesInputSplit[0].equals("NumberOfAllSupervisors")) {
      
      Node queryNode = null; 
      int count = 0;
      
       // Looping through allNodes to find the input
       for (int i = 0; i < allNodes.size(); i++) {
        if (queriesInputSplit[1].equals(allNodes.get(i).name)) {
         queryNode = allNodes.get(i);
         break;
        }
       }
      
       if (queryNode == null) {
        continue;
       }
       
       Node temp = queryNode;

       if (queriesInputSplit[0].equals("AllSupervisors")) {
        System.out.print("AllSupervisors " + queriesInputSplit[1] + " ");
       }
      
       // Looping through each supervisor and their supervisor until the top entity is reached
       while (temp.parent != null) {
        
        temp = temp.parent;
        count++;
        
        // If the first word is AllSupervisors output all supervisors
        if (queriesInputSplit[0].equals("AllSupervisors")) {
        System.out.print(temp.name + " ");
        }
       } 
       
       System.out.println();
       
       // If the first word is NumberOfAllSupervisors output the amount of supervisors
       if (queriesInputSplit[0].equals("NumberOfAllSupervisors")) {
        System.out.println("NumberOfAllSupervisors " + queriesInputSplit[1] + " " + count);
       }
       
      }
     
     // Checking if the first word is AllSubordinates or NumberOfAllSubordinates
     else if (queriesInputSplit[0].equals("AllSubordinates") || queriesInputSplit[0].equals("NumberOfAllSubordinates")) {
      
      Node queryNode = null;
      int count = 0;
      
       // Looping through allNodes to find the input
       for (int i = 0; i < allNodes.size(); i++) {
        if (queriesInputSplit[1].equals(allNodes.get(i).name)) {
         queryNode = allNodes.get(i);
         break;
        }
       }
       
       if (queryNode == null) {
        continue;
       }
       
       
       Node temp = queryNode;
       
       // Creating list for of all the childNodes of the input 
       LinkedList<Node> childNodes = new LinkedList<>();
       
       if (queriesInputSplit[0].equals("AllSubordinates")) {
        System.out.print("AllSubordinates " + queriesInputSplit[1] + " ");
       }
       
       childNodes.addAll(tree.getChildren(queryNode));
       
       // Looping through childNodes until it is empty
       while (!childNodes.isEmpty()) {
       
        // Storing and removing the child Node at the end of the list
        temp = childNodes.removeLast();
        
        // Adding the temp children to the list in reverse order
        for (int i = tree.getChildren(temp).size() - 1; i >= 0; i--) {
         childNodes.add(tree.getChildren(temp).get(i));
        }
        
        count++;
        
        // If the first word is AllSubordinates output all of the input supervisors subordinates 
        if (queriesInputSplit[0].equals("AllSubordinates")) {
         System.out.print(temp.name + " ");
        }
       }
      
      // If the first word is AllSubordinates add a line for formatting purposes
      if (queriesInputSplit[0].equals("AllSubordinates")) {
       System.out.println();
      } 
       
       // If the first word is NumberOfAllSubordinates output the ammount of subordinates
       if (queriesInputSplit[0].equals("NumberOfAllSubordinates")) {
        System.out.println("NumberOfAllSubordinates " + queriesInputSplit[1] + " " + count);
       } 
       
     }  
     
     // Checking if the first word is IsSupervisor 
     else if (queriesInputSplit[0].equals("IsSupervisor")) {
      
      Node queryNode = null;
      
      // Looping through allNodes to find input
      for (int i = 0; i < allNodes.size(); i++) {
       if (queriesInputSplit[1].equals(allNodes.get(i).name)) {
        queryNode = allNodes.get(i);
        break;
       }
      }
     
      if (queryNode == null) {
       continue;
      }
      
      
      boolean isSuper = false;
      Node temp = queryNode;
      
      // Looping through each parent node until we reach the top entity 
      while (temp.parent != null) {
       
       temp = temp.parent;
       
        // Checking if the input supervisor is a supervisor of the input subordinate
        if (temp.name.equals(queriesInputSplit[2])) {
         isSuper = true;
        }
       } 
      
        // If the input supervisor is a supervisor of the input subordinate, output yes
        // If not, output no
        if (isSuper == true) {
         System.out.println("IsSupervisor " + queriesInputSplit[1] + " " + queriesInputSplit[2] + " yes");
        }
        else 
         System.out.println("IsSupervisor " + queriesInputSplit[1] + " " + queriesInputSplit[2] + " no"); 

     }
     
     // Checking if the first word is IsSubordinate
     else if (queriesInputSplit[0].equals("IsSubordinate")) {
      
      Node queryNode = null;
      
      // Looping through allNodes to find the input 
      for (int i = 0; i < allNodes.size(); i++) { 
       if (queriesInputSplit[1].equals(allNodes.get(i).name)) {
        queryNode = allNodes.get(i);
        break;
       } 
      }
      
      if (queryNode == null) {
       continue;
      }
      
      // Creating a list and adding the input supervisor child nodes to the list
      LinkedList<Node> childNodes = new LinkedList<>();
      childNodes.addAll(tree.getChildren(queryNode));
      
      boolean isSub = false;
      Node temp;
      
      // Looping through the list of child Nodes until the list is empty
      while (!childNodes.isEmpty()) {
      
       // Storing and removing the first child node from the list
       // Adding the child nodes of temp to the list of child nodes
       temp = childNodes.removeFirst();
       childNodes.addAll(tree.getChildren(temp));
       
       // Checking if the current node is the subordinate
       if (queriesInputSplit[2].equals(temp.name)) {
        isSub = true; 
       }
      }
      
      // If the subordinate was the found, output yes
      // If not, output no
      if (isSub == true) { 
       System.out.println("IsSubordinate " + queriesInputSplit[1] + " " + queriesInputSplit[2] + " yes");
       }  
      else 
       System.out.println("IsSubordinate " + queriesInputSplit[1] + " " + queriesInputSplit[2] + " no");
      }
     
     // Checking if the first word is CompareRank
     else if (queriesInputSplit[0].equals("CompareRank")) {
      
      Node queryInput1 = null;
      Node queryInput2 = null;
      
      // Looping through allNodes to find both input words
      for (int i = 0; i < allNodes.size(); i++) {
       if (queriesInputSplit[1].equals(allNodes.get(i).name)) {
        queryInput1 = allNodes.get(i);
       }                
       if (queriesInputSplit[2].equals(allNodes.get(i).name)) {
        queryInput2 = allNodes.get(i);
       }
      }
      
      int rank1 = 0;
      int rank2 = 0;
      Node temp1 = queryInput1;
      Node temp2 = queryInput2;
      
      // Looping through temps1 parents until reaching the top entity
      // Adding one each loop to the rank
      while (temp1.parent != null) {
       temp1 = temp1.parent;
       rank1++;
      }
      
      // Looping through temps2 parents until reaching the top entity
      // Adding one each loop to the rank
      while (temp2.parent != null) {
       temp2 = temp2.parent;
       rank2++;
      } 
      
      // If temps1 rank was a lower number , output higher
      // If temps2 rank was a lower number , output lower
      // If both temp1 and temp2 rank were the same number, output same
      if (rank1 < rank2) {
       System.out.print("CompareRank " + queriesInputSplit[1] + " " + queriesInputSplit[2] + " higher");
      } 
      if (rank1 > rank2) {
       System.out.print("CompareRank " + queriesInputSplit[1] + " " + queriesInputSplit[2] + " lower");
      }
      if (rank1 == rank2) {
       System.out.print("CompareRank " + queriesInputSplit[1] + " " + queriesInputSplit[2] + " same");
      }
      
     }
     
     // Checking if the first word is ClosestCommonSupervisor
     else if (queriesInputSplit[0].equals("ClosestCommonSupervisor")) {
      
      Node queryInput1 = null;
      Node queryInput2 = null;
      
      // Looping through allNodes to find both input words
      for(int i = 0; i < allNodes.size(); i++) { 
       if (queriesInputSplit[1].equals(allNodes.get(i).name)) {
        queryInput1 = allNodes.get(i);
       }
       if (queriesInputSplit[2].equals(allNodes.get(i).name)) {
        queryInput2 = allNodes.get(i);
       }
      }
      
      // Creating a list to store supervisors
      LinkedList<Node> supervisors = new LinkedList<>();
       
      Node temp = queryInput1;
      
      // Looping through each temp parent until reaching the top entity
      // Adding each supervisor to the list
      while (temp.parent != null) {
       temp = temp.parent;
       supervisors.add(temp);
      } 
      
      Node temp2 = queryInput2;
      Node ClosestCommonSupervisor = null;
      
      // Looping through temp2 parents until reaching the top entity
      while (temp2.parent != null) {
      
       temp2 = temp2.parent;
       
       // Comparing temp supervisors to temp2 supervisors
       // Storing the first matching supervisor
       for (int i = 0; i < supervisors.size(); i++) { 
        if (temp2 == supervisors.get(i)) {
         ClosestCommonSupervisor = temp2;
         break;
        } 
       }
       
       // If a common supervisor was found, break the loop
       if (ClosestCommonSupervisor != null) {
        break;
       } 
      }
      
      // If there was a matching supervisor, output the supervisor
      // If there was no common supervisor, output none
      if (ClosestCommonSupervisor != null) {
      System.out.print("ClosestCommonSupervisor " + queriesInputSplit[1] + queriesInputSplit[2] + " " + ClosestCommonSupervisor.name);
      }
      else 
       System.out.print("ClosestCommonSupervisor " + queriesInputSplit[1] + queriesInputSplit[2] + " none");
     
     }
    
   }
 
 }
}

//Creating a class to implement nodes
class Node {

 LinkedList<Node> childNodes = new LinkedList<>();

 String name;
 Node parent;
 
 Node(String name) {
  this.name = name;
  } 
 
}

// Creating a class to create addChild method
class Tree {
 
 public void addChild(Node node, Node childNode) {
 boolean done = false;
 boolean duplicate = false;
  for (int i = 0; i < node.childNodes.size(); i++) {
 
   // Checking if the child node already exist
   if (childNode.name.equals(node.childNodes.get(i).name)) {
    duplicate = true;
    break;
   } 
   
   // Storing the child nodes alphabetically
   if (childNode.name.compareTo(node.childNodes.get(i).name) < 0) {
    node.childNodes.add(i, childNode);
    done = true;
    break;
   }
 
  }
  
  // If the child node does not already exist and child node was not already stored 
  // Add child node to end of the list
  if (done == false && duplicate == false) {
   node.childNodes.add(childNode);
  }
  
  childNode.parent = node;
  
 } 
 
 // Returning the parent node when called
 public Node getParent(Node node) {
  return node.parent;
 }
 
 // Returns children nodes when called
 public LinkedList<Node> getChildren(Node node) {
  return node.childNodes;
 } 
}
