/*
Author: Israel Caballero
Email: icaballeroes2025@my.fit.edu
Course:CSE2010-Algorithms and Data Structures
Section:01
Description of this file:Program that creates a tree from a workplace heirarchy(grand supervisors...supervisors and suburdinates)
                         and answers queries based on that structure.
*/
import java.util.Scanner;
import java.io.File;
import java.util.ArrayList;
public class HW3{
public static class LinkedTree{
//Node class that stores an element the parent and the children of the node as an arrayList
protected static class Node{
    private String element;
    private Node parent;
    private ArrayList<Node> children;

public Node(String e , Node above){
    element = e;
    parent = above;
    children = new ArrayList<>();
}
//Accessor Methods
public String getElement(){return element;}
public Node getParent(){return parent;}
public ArrayList<Node> getList(){return children;}
//Update Methods
public void setElement(String e){element= e;}
public void setParent(Node parentNode){parent = parentNode;}
public void setChildren(ArrayList<Node> childrenList){children = childrenList;}
}
//sets root and creates an ArrayList with everyNode
private Node root;
private ArrayList<Node> allNodes = new ArrayList<>();

public void setRoot(Node r){
    root = r; 
    allNodes.add(root);
}
//finds a node with the name given
public Node find(String name){
    for(int i = 0; i < allNodes.size(); i++){
        if(allNodes.get(i).getElement().equals(name)){
            return allNodes.get(i);
        }
    }
    return null;
}
//adds a child to a node by first setting its parent and adding it to allNodes
public void addChild(Node parent, Node child){
    ArrayList<Node> childList = parent.getList();
    child.setParent(parent);
    allNodes.add(child);
    //compares names in alphabetical order to see where to place it in the list
    for(int i = 0; i < childList.size(); i++){
        if(childList.get(i).getElement().compareTo(child.getElement()) > 0){
            childList.add(i,child);
            return;

        }
    }
    childList.add(child);
}

}

public static void main(String[] args) throws Exception{
    Scanner input = new Scanner(new File(args[0]));

    LinkedTree tree = new LinkedTree();
    //Sets the top supervisor of the tree.
    String root = input.nextLine();
    LinkedTree.Node rootName = new LinkedTree.Node(root, null);
    tree.setRoot(rootName);

    //while there is a next line to read
    while (input.hasNextLine()){
        //Makes each line into a string and then breaks the line into words after any white space and assigns those words to tokens
        String line = input.nextLine();
        String[] token = line.split("\\s+");
        //takes the first name in the line and makes the second name its child
        LinkedTree.Node supervisor = tree.find(token[0]);
        LinkedTree.Node subordinate = new LinkedTree.Node(token[1], null);
        tree.addChild(supervisor, subordinate);
    }
    Scanner inputQueries = new Scanner(new File(args[1]));

    //while there is a next line to read
    while(inputQueries.hasNextLine()){
        //splits line into words which are tokens
        String line = inputQueries.nextLine();
        String[] token = line.split("\\s+");
        //If the first word is "DirectSupervisor"
        if(token[0].equals("DirectSupervisor")){
            //set subordinate and supervisor variables
            LinkedTree.Node subordinate = tree.find(token[1]);
            LinkedTree.Node supervisor = subordinate.getParent();
            String result;
            //If there is no supervisor set result to none, otherwise get the supervisors name 
            if(supervisor == null){result = "none";}
            else result = supervisor.getElement();
            
            System.out.println(token[0] + " " + token[1] + " " + result);
        }
        //If the first word is "DirectSubordinates"
        if(token[0].equals("DirectSubordinates")){
            //finds supervisor in list, and creates an ArrayList of all the subordinates
            LinkedTree.Node supervisor = tree.find(token[1]);
            ArrayList<LinkedTree.Node> subordinateList = supervisor.getList();
            //if there are no subordinates print none
            if(subordinateList.isEmpty()){
                System.out.println(token[0] + " " + token[1] + " none");
            }
            //otherwise print all subordinates
            else{
                System.out.print(token[0] + " " + token[1]);
                for(int i = 0; i < subordinateList.size(); i++){
                    System.out.print(" " + subordinateList.get(i).getElement());
            }
            System.out.println();
            }
        }
        //If first word is "IsSupervisor"
        if(token[0].equals("IsSupervisor")){
            //Finds subordinate in the tree and sets the supervisor variable
            LinkedTree.Node subordinate = tree.find(token[1]);
            String supervisor = token[2];
            String decision = "no";
            //travereses through each parent to see if they are a supervisor to the element
            LinkedTree.Node current = subordinate.getParent();
            while(current != null){
                if(current.getElement().equals(supervisor)){
                    decision = "yes";
                    break;
                }
                current = current.getParent();
            }
            System.out.println(token[0] + " " + token[1] + " " + token[2] + " " + decision);

        }
        //If the first word is "IsSubordinate"
        if(token[0].equals("IsSubordinate")){
            //Finds supervisor in the tree and sets subordinate variable
            LinkedTree.Node supervisor = tree.find(token[1]);
            String subordinate = token[2];
            //creates an ArrayList of all children of the supervisor and their children
            ArrayList<LinkedTree.Node> toCheck = new ArrayList<>(supervisor.getList());
            String decision = "no";
            //traverses through list as long as there is elements in it
            while(toCheck.size() != 0){
                //if the name equals the given name from the list then make the decision yes
                LinkedTree.Node current = toCheck.remove(0);
                    if(current.getElement().equals(subordinate)){
                        decision = "yes";
                        break;
                    }
                    //adds the children to the list
                    toCheck.addAll(current.getList());
                }
            System.out.println(token[0] + " " + token[1] + " " + token[2] + " " + decision);

        }
        //If first word is "CompareRank"
        if(token[0].equals("CompareRank")){
            //set first entity and second and assign a depth to each of them
            LinkedTree.Node entity1 = tree.find(token[1]);
            int entity1Depth = 0; 
            LinkedTree.Node entity2 = tree.find(token[2]); 
            int entity2Depth = 0;
            String decision = "higher";
            //counts the depth of entity1
            LinkedTree.Node current1 = entity1;
            while(current1 != null){
                current1 = current1.getParent();
                entity1Depth += 1;
            }
            //counts depth of entity2
            LinkedTree.Node current2 = entity2;
            while(current2 != null){
                current2 = current2.getParent();
                entity2Depth += 1;
            }
            //if depth is the same, they are the same rank
            if(entity1Depth == entity2Depth){decision = "same";}
            //If entity one is higher than make decision lower
            if(entity1Depth > entity2Depth){decision = "lower";}

            System.out.println(token[0] + " " + token[1] + " " + token[2] + " " + decision);

        }
        //If first word is "ClosestCommonSupervisor"
        if(token[0].equals("ClosestCommonSupervisor")){
            //sets first and second entity and creates an arraylist of each of their supervisors
            LinkedTree.Node entity1 = tree.find(token[1]);
            ArrayList<LinkedTree.Node> entity1Supervisors = new ArrayList<>();
            LinkedTree.Node entity2 = tree.find(token[2]);
            ArrayList<LinkedTree.Node> entity2Supervisors = new ArrayList<>();
            String closestSupervisor = root;
            //Adds each supervisor of entity 1 to list1
            LinkedTree.Node current1 = entity1;
            while(current1 != null){
                entity1Supervisors.add(current1);
                current1 = current1.getParent();
            }
            //Adds each supervisorof entity 2 to list2
            LinkedTree.Node current2 = entity2;
            while(current2 != null){
                entity2Supervisors.add(current2);
                current2 = current2.getParent();
            }
            //Checks a supervisor of entity 1 against all supervisors of entity 2 does the same for every entity1 supervisor.
            //stops the loop when the closestcommonsupervisor is found
            boolean found = false;
            for(int i = 0; i < entity1Supervisors.size() && !found;i++){
                for(int j = 0; j < entity2Supervisors.size() && !found; j++){
                    if(entity1Supervisors.get(i).getElement().equals(entity2Supervisors.get(j).getElement())){
                        closestSupervisor = entity1Supervisors.get(i).getElement();
                        found = true;
                    }
                }
            }

            System.out.println(token[0] + " " + token[1] + " " + token[2] + " " + closestSupervisor);
        }
        //If first word is "NumberOfAllSupervisors"
        if(token[0].equals("NumberOfAllSupervisors")){
            //finds entity1 in the tree and creates an arraylist of supervisors
            LinkedTree.Node entity1 = tree.find(token[1]);
            ArrayList<LinkedTree.Node> entity1Supervisors = new ArrayList<>();
            //adds each supervisor of entity1 into entity1Supervisors and then prints the list size-1
            LinkedTree.Node current1 = entity1;
            while(current1 != null){
                entity1Supervisors.add(current1);
                current1 = current1.getParent();
            }
            
            System.out.println(token[0] + " " + token[1] + " " + (entity1Supervisors.size()-1) );
        }
        //If first word is "NumberOfAllSubordinates"
        if(token[0].equals("NumberOfAllSubordinates")){
            //finds entity1 in the tree and creates an arraylist of all its subordinates
            LinkedTree.Node entity1 = tree.find(token[1]);
            ArrayList<LinkedTree.Node> allSubordinates = new ArrayList<>(entity1.getList());
            //goes through allSubordinatesList, removing one by one and increasing count by one.
            int count = 0;
            while(allSubordinates.size() != 0){
                LinkedTree.Node current = allSubordinates.remove(0);
                count++;
                //adds the children of current to the list
                allSubordinates.addAll(current.getList());
                }
        
            System.out.println(token[0] + " " + token[1] + " " + count);
        }
        //If first word is "AllSubordinates"
        if(token[0].equals("AllSubordinates")){
            //finds entity1 in the tree and creates an array list of all subordinates
            LinkedTree.Node entity1 = tree.find(token[1]);
            ArrayList<LinkedTree.Node> allSubordinates = new ArrayList<>();
            String subordinates = "";
            //fills the list with the initial entities subordinates but in a backwards order so it prints correctly
            for(int i = entity1.getList().size()-1; i >=0;i--){
                allSubordinates.add(entity1.getList().get(i));
            }
            //while there is still subordinates in the list remove them one by one from the end of the list
            while(allSubordinates.size() != 0){
                LinkedTree.Node current = allSubordinates.remove(allSubordinates.size()-1);
                //adds the currents subordinates one by one in a backwards order so it prints correctly
                for(int i = current.getList().size()-1; i >= 0;i--){
                    allSubordinates.add(current.getList().get(i));
                }
                subordinates = subordinates.concat(" " + current.getElement());
            }
            //if there is no subordinates print none
            if(subordinates.equals("")){
                System.out.println(token[0] + " " + token[1] + " none");
            }
            // print subordinates list if there are subordinates
            else
                System.out.println(token[0] + " " + token[1] + subordinates);
        }
        //If first word is "AllSupervisors"
        if(token[0].equals("AllSupervisors")){
            //finds entity one in the tree
            LinkedTree.Node entity1 = tree.find(token[1]);
            String supervisors = "";
            //if there is no parent print none
            LinkedTree.Node current1 = entity1.getParent();
            if(current1 == null){
                System.out.println(token[0] + " " + token[1] + " none");
            }
            //concatinate supervisors with every parent there is until there are no parents left
            else{
            while(current1 != null){
                supervisors = supervisors.concat( " " + current1.getElement());
                current1 = current1.getParent();
            }
            System.out.println(token[0] + " " + token[1] + supervisors);
            }
        }
    }


}
}

