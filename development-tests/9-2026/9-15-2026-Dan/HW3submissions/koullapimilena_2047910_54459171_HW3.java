/*

  Author: Milena Koullapi 
  Email: mkoullapi2025@my.fit.edu
  Course: CSE 2010
  Section: 01
  Description of this file: Reads organizational hierarchy data from an input file to build a tree structure, 
    then processes query commands from a second file to print information about entity supervisors, subordinates, and ranks.

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;

public class HW3{

    /*
     * This method reads data and query filenames from command-line arguments, builds an organizational 
     * hierarchy tree from the data file, processes every line in the query file, and prints the corresponding output.
     * @param args - command-line arguments; args[0] is the path to the organizational data file and args[1] is the path to the query file.
    */
    public static void main(String[] args){
      /* 
      Filename store the name or path of the file that is going to be read.
      Thats information is received from args[0], which represents the first command-line argument 
      */
      String filename = args[0];

      Node root = null;
      Tree tree = null; 

      try ( Scanner scanner = new Scanner(new File(filename))){
        //Read specifically the first line of the file
        String firstline = scanner.nextLine();

        root = new Node(firstline);       
        tree = new Tree(root);       

        // Checks if there is another line of text left to read in the file. The loop will continue running as long as there is more lines.
        while(scanner.hasNextLine()){
          // Reads the next line of textand stores it in the variable called line.
          String line = scanner.nextLine();        
          String[] parts = line.split(" ");
          String supervisorName = parts[0];
          String subordinateName = parts[1];

          Node supervisorNode = Tree.findNode(root, supervisorName);
          Node subordinateNode = new Node (subordinateName); 

          Tree.addChild(supervisorNode, subordinateNode); 
        }

      }
      //This block only runs if the attempt to open the file specified by filename doesnt exists, cannot be opens or found in the specific path.
      catch (FileNotFoundException e) { 
        //It print "File not found" to the user
        System.out.println("File not found.");
        //it print the detailed technical details of the errors to help the developers understand where and why the error happened
        e.printStackTrace();
      }

      // Reads the query file line by line and processes each specific request
      try (Scanner queryScanner = new Scanner(new File(args[1]))) {
        while (queryScanner.hasNextLine()) {
          String queryLine = queryScanner.nextLine();
          String[] queryParts = queryLine.split(" ");

          // Checks if the entity has a direct supervisor and prints its name, or prints none if it does not have one
          if (queryParts[0].equals("DirectSupervisor")) {
            String entity = queryParts[1];
            
            Node entityNode = Tree.findNode(root, entity);
            
            if(entityNode != null && entityNode.getParent() != null){
              String supervisorName = entityNode.getParent().getData();
              System.out.println("DirectSupervisor " + entity + " " + supervisorName );
            }
            else{         
              System.out.println("DirectSupervisor " + entity + " none" );
            }

          }

          // Finds all direct subordinates of an entity and prints them in alphabetical order
          else if(queryParts[0].equals("DirectSubordinates")) {
            String entity = queryParts[1];
            
            Node entityNode = Tree.findNode(root, entity);
            
            if(entityNode != null && entityNode.getChildren().size() > 0){
              String result = "DirectSubordinates " + entity;

              for(Node child : entityNode.getChildren()){
                result += " " + child.getData();
              }

              System.out.println(result);

            } else {         
              System.out.println("DirectSubordinates " + entity + " none" );
            }

          }

          // Climbs up the hierarchy starting from the entity's parent all the way to the root to find all supervisors
          else if(queryParts[0].equals("AllSupervisors")) {
            String entity = queryParts[1];
            
            Node entityNode = Tree.findNode(root, entity);
            
            if(entityNode != null && entityNode.getParent() != null){
              String result = "AllSupervisors " + entity;

              Node current = entityNode.getParent();
              while(current != null){
                result += " " + current.getData();
                current = current.getParent();
              }

              System.out.println(result);

            } else {         
              System.out.println("AllSupervisors " + entity + " none" );
            }

          }

          // Performs a pre-order traversal starting from the given entity node to collect and print all subordinates
          else if(queryParts[0].equals("AllSubordinates")) {
            String entity = queryParts[1];
            Node entityNode = Tree.findNode(root, entity);
            
            if(entityNode != null && entityNode.getChildren().size() > 0){
              ArrayList<String> list = new ArrayList<>();
              Tree.collectSubordinates(entityNode, list);

              String result = "AllSubordinates " + entity;
              for(String subName : list ){
                result += " " + subName;
              }

              System.out.println(result);

            } else {         
              System.out.println("AllSubordinates " + entity + " none" );
            }
          }

          // Counts how many levels of supervisors exist above the given entity
          else if (queryParts[0].equals("NumberOfAllSupervisors")) {
            String entity = queryParts[1];
            Node entityNode = Tree.findNode(root, entity);

            int count = 0;
            if (entityNode != null) {
              Node current = entityNode.getParent();
              while (current != null) {
                count++;
                current = current.getParent();
              }
            }
            System.out.println("NumberOfAllSupervisors " + entity + " " + count);
          }

          // Counts the total number of subordinates under the given entity in the hierarchy
          else if (queryParts[0].equals("NumberOfAllSubordinates")) {
            String entity = queryParts[1];
            Node entityNode = Tree.findNode(root, entity);

            int count = 0;
            if (entityNode != null) {
              ArrayList<String> list = new ArrayList<>();
              Tree.collectSubordinates(entityNode, list);
              count = list.size();
            }
            System.out.println("NumberOfAllSubordinates " + entity + " " + count);
          }

          // Checks if the specified supervisor exists anywhere in the parent path above the entity
          else if (queryParts[0].equals("IsSupervisor")) {
            String entity = queryParts[1];
            String supervisor = queryParts[2];
            Node entityNode = Tree.findNode(root, entity);

            boolean found = false;
            if (entityNode != null) {
              Node current = entityNode.getParent();
              while (current != null) {
                if (current.getData().equals(supervisor)) {
                  found = true;
                  break;
                }
                current = current.getParent();
              }
            }

            if (found) {
              System.out.println("IsSupervisor " + entity + " " + supervisor + " yes");
            } else {
              System.out.println("IsSupervisor " + entity + " " + supervisor + " no");
            }
          }

          // Checks if the specified subordinate exists anywhere under the entity using pre-order traversal
          else if (queryParts[0].equals("IsSubordinate")) {
            String entity = queryParts[1];
            String subordinate = queryParts[2];
            Node entityNode = Tree.findNode(root, entity);

            boolean found = false;
            if (entityNode != null) {
              ArrayList<String> list = new ArrayList<>();
              Tree.collectSubordinates(entityNode, list);
              if (list.contains(subordinate)) {
                found = true;
              }
            }

            if (found) {
              System.out.println("IsSubordinate " + entity + " " + subordinate + " yes");
            } else {
              System.out.println("IsSubordinate " + entity + " " + subordinate + " no");
            }
          }

          // Compares the distance to the root for two entities to determine which one has a higher rank
          else if (queryParts[0].equals("CompareRank")) {
            String entity1 = queryParts[1];
            String entity2 = queryParts[2];

            Node node1 = Tree.findNode(root, entity1);
            Node node2 = Tree.findNode(root, entity2);

            int depth1 = 0;
            Node current = node1 != null ? node1.getParent() : null;
            
            while (current != null) {
              depth1++;
              current = current.getParent();
            }

            int depth2 = 0;
            current = node2 != null ? node2.getParent() : null;
            while (current != null) {
              depth2++;
              current = current.getParent();
            }

            if (depth1 < depth2) {
              System.out.println("CompareRank " + entity1 + " " + entity2 + " higher");
            } else if (depth1 > depth2) {
              System.out.println("CompareRank " + entity1 + " " + entity2 + " lower");
            } else {
              System.out.println("CompareRank " + entity1 + " " + entity2 + " same");
            }
          }

          // Finds the lowest common supervisor shared by both entity1 and entity2
          else if (queryParts[0].equals("ClosestCommonSupervisor")) {
            String entity1 = queryParts[1];
            String entity2 = queryParts[2];

            Node node1 = Tree.findNode(root, entity1);
            Node node2 = Tree.findNode(root, entity2);

            ArrayList<String> super1 = new ArrayList<>();
            Node current = node1 != null ? node1.getParent() : null;
            while (current != null) {
              super1.add(current.getData());
              current = current.getParent();
            }

            String common = "none";
            current = node2 != null ? node2.getParent() : null;
            while (current != null) {
              if (super1.contains(current.getData())) {
                common = current.getData();
                break; 
              }
              current = current.getParent();
            }

            System.out.println("ClosestCommonSupervisor " + entity1 + " " + entity2 + " " + common);
          } 

        }
      }

      //This block only runs if the attempt to open the file specified by filename doesnt exists, cannot be opens or found in the specific path.
      catch (FileNotFoundException e) { 
        //It print "File not found" to the user
        System.out.println("File not found.");
        //it print the detailed technical details of the errors to help the developers understand where and why the error happened
        e.printStackTrace();
      }


    }

    /*
     * Represents an individual node in the organizational tree hierarchy, holding its string data,
     * parent reference, and list of child nodes.
    */
    public static class Node {
      private String data;
      private Node parent;
      private ArrayList<Node> children;

      /*
       * Constructor to initialize a new Node with a given data name.
       * @param data - the entity name represented by this node
      */
      public Node(String data) {
        this.data = data;
        this.parent = null;
        this.children = new ArrayList<>();
      }

      /*
       * Returns the string name stored inside this node.
       * @return the node data string
      */
      public String getData() {
        return data;
      }

      /*
       * Returns the parent node of this node.
       * @return the parent Node reference
      */
      public Node getParent(){
        return parent;
      }

      /*
       * Returns the list of child nodes underneath this node.
       * @return an ArrayList of child Nodes
      */
      public ArrayList<Node> getChildren(){
        return children;
      } 

      /*
       * Sets the parent node reference for this node.
       * @param parent - the parent Node to set
      */
      public void setParent(Node parent){
        this.parent = parent;
      }
    }

  /*
   * Represents the tree structure containing static helper operations for adding children,
   * searching for nodes, and performing pre-order subordinate collection.
  */
  public static class Tree{
    private Node root;

    /*
     * Constructor to initialize the Tree with a given root node.
     * @param root - the root node of the organizational hierarchy
    */
    public Tree(Node root){
      this.root = root;
    }

    /*
     * Returns the root node of this tree.
     * @return the root Node
    */
    public Node getRoot() {
      return root;
    }

    /*
     * Adds a child node to a parent node while maintaining alphabetical order among children.
     * @param node - the supervisor node receiving the child
     * @param childNode - the subordinate node to add
    */
    static void addChild(Node node, Node childNode) {
      ArrayList<Node> children = node.getChildren();
      childNode.setParent(node);

      int i = 0;
      while (i < children.size()) {
        String existingName = children.get(i).getData();
        String newName = childNode.getData();
        

        if (newName.compareTo(existingName) < 0) {
            children.add(i, childNode);
            
            return;
        }
        i++;
      }

      children.add(childNode);
    
    }

    /*
     * Returns the list of child nodes for a specified node.
     * @param node - the node to retrieve children from
     * @return an ArrayList of child Nodes
    */
    static ArrayList<Node> getChildren(Node node) {
      return node.getChildren();
    }

    /*
     * Returns the parent node of a specified node.
     * @param node - the node to retrieve the parent from
     * @return the parent Node reference
    */
    static Node getParent(Node node) {
      return node.getParent();
    }

  
    /*
     * Recursively searches the tree starting from current node to find a node with targetName.
     * @param current - the starting node for the recursive search
     * @param targetName - the entity name to look for
     * @return the matching Node if found, or null if not found
    */
    static Node findNode(Node current, String targetName){
      if(current == null) return null;
      if(targetName.equals(current.getData())){
        return current;
      }

      ArrayList<Node> children = Tree.getChildren(current);

      for(int i = 0; i < children.size(); i++){
        Node result = findNode(children.get(i), targetName);
        if(result != null){
          return result;
        }
      }
      return null;
    }

    /*
     * Recursively collects all subordinate entity names under current node using a pre-order traversal.
     * @param current - the starting node for subordinate collection
     * @param list - the list collecting subordinate name strings
    */
    static void collectSubordinates(Node current, ArrayList<String> list) {
      if (current == null) return;
    
      for (Node child : current.getChildren()) {
        list.add(child.getData());               
        collectSubordinates(child, list);       
      }
    }

  }

}