/*
Author: Avery Haughwout
Email: ahaughwout2024@my.fit.edu
Course: CSE 2012
Section: 3:30
File Description: Populates a tree based on a specifically formatted input file, then
computes relations between the nodes based on a formatted query file.
*/
//Imports :)
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class HW3{
    
    public static void main(String[] args) throws FileNotFoundException {
        Scanner scnr1 = new Scanner(new File(args[0]));
        Scanner scnr2 = new Scanner(new File(args[1]));
        Tree dataTree = new Tree();//makes the tree we're gonna use :)
        
        //DATA INPUT TIME
        dataTree.setRoot(scnr1.nextLine());//Declares the root :D
        
        while(scnr1.hasNextLine()){
            String input = scnr1.nextLine();
            String[] inputs = input.split(" ");
            dataTree.addLink(inputs[0], inputs[1]); //Puts all the data in
        }
        scnr1.close();

        while(scnr2.hasNextLine()){
            String input = scnr2.nextLine();
            String[] inputs = input.split(" ");
            switch(inputs[0]){
                case "DirectSupervisor": //DONE
                    DirectSupervisor(dataTree, inputs[1]);
                    break;
                case "DirectSubordinates": //DONE
                    DirectSubordinates(dataTree, inputs[1]);
                    break;
                case "AllSupervisors": //DONE
                    AllSupervisors(dataTree, inputs[1]);
                    break;
                case "AllSubordinates": //DONE
                    AllSubordinates(dataTree, inputs[1]);
                    break;
                case "NumberOfAllSupervisors": //DONE
                    NumberOfAllSupervisors(dataTree, inputs[1]);
                    break;
                case "NumberOfAllSubordinates": // DONE
                    NumberOfAllSubordinates(dataTree, inputs[1]);
                    break;
                case "IsSupervisor": //DONE
                    IsSupervisor(dataTree, inputs[1], inputs[2]);
                    break;
                case "IsSubordinate": //DONE
                    IsSubordinate(dataTree, inputs[1], inputs[2]);
                    break;
                case "CompareRank": //DONE
                    CompareRank(dataTree, inputs[1], inputs[2]);
                    break;
                case "ClosestCommonSupervisor": // DONE:
                    ClosestCommonSupervisor(dataTree, inputs[1], inputs[2]);
                    break;
                default:
                    break;
            }
        }
        scnr2.close();
    }
    public static void DirectSupervisor(Tree dataTree, String entity){
        /*Possible inputs:
        - the entity is the root
        - the entity is not on the list
        - the entity is on the list */
        if(dataTree.getRoot().getElement().equals(entity)){//quickly covers the case in which the root is called
            System.out.println("DirectSupervisor "+entity+" none");
        }
        else if(dataTree.search(dataTree.getRoot(), entity) == null){
            System.out.println("DirectSupervisor "+entity+" none");
        }
        else{
            System.out.println("DirectSupervisor "+entity+" "+dataTree.search(dataTree.getRoot(), entity).getParent().getElement());
        }
    }
    
    public static void DirectSubordinates(Tree dataTree, String entity){
        /*
        Possible inputs:
        The entity is not on the list
        The entity is on the list
        the entity has no kids
        */
        if(dataTree.search(dataTree.getRoot(), entity) == null){
            System.out.println("DirectSubordinates "+entity+" none");
        }
        else if(dataTree.search(dataTree.getRoot(), entity).getChildren().isEmpty()){
            System.out.println("DirectSubordinates "+entity+" none");
        }
        else{
            String response = "DirectSubordinates " + entity + " ";
            if(dataTree.search(dataTree.getRoot(), entity).getChildren().isEmpty()){
                response += "none";
            }
            else{
                for(int i = 0; i < dataTree.search(dataTree.getRoot(), entity).getChildren().size(); i++){
                    response += dataTree.search(dataTree.getRoot(), entity).getChildren().get(i).getElement() + " ";
                }
                System.out.println(response);
            }
        }
    }

    public static void AllSupervisors(Tree dataTree, String entity){
        /*Potential cases:
        The one called is the root
        The one called has parents :)
        How to go about this:
        greate a string which += each current node's getParent()*/
        if(dataTree.search(dataTree.getRoot(), entity).getParent()== null){
            System.out.println("AllSupervisors " +entity+ " none");
        }
        else{
            System.out.println("AllSupervisors "+entity+" "+getAllSupervisors(dataTree.search(dataTree.getRoot(), entity)));
        }
    }
    public static String getAllSupervisors(Tree.Node currentNode){
        if(currentNode.getParent()==null){ //Checks if it is the root
            return "";
        }
        else{
            return currentNode.getParent().getElement() + " " + getAllSupervisors(currentNode.getParent());
        }
    }

    public static void AllSubordinates(Tree dataTree, String entity){
        if(dataTree.search(dataTree.getRoot(), entity).getChildren().isEmpty()){
            System.out.println("AllSubordinates " +entity+ " none");
        }
        else{
            System.out.println("AllSubordinates " +getAllSubordinates(dataTree.search(dataTree.getRoot(), entity)));
        }
    }
    public static String getAllSubordinates(Tree.Node currentNode){
        /*
        Objective of this method:
        -Prints the first child of the currentNode, and all of that node's children
        once that is null, it  continues onward */
        if(currentNode.getChildren().isEmpty()){
            return currentNode.getElement(); // what happens when the end of the tree line is met
        }
        else{
            String s = currentNode.getElement();
            for(int i = 0; i < currentNode.getChildren().size(); i++){
                s += " " + getAllSubordinates(currentNode.getChildren().get(i));
            }
            return s;
        }
    }
    public static void NumberOfAllSupervisors(Tree dataTree, String entity){
        System.out.println("NumberOfAllSupervisors "+entity+" "+ numSupervisor(dataTree.search(dataTree.getRoot(), entity)));
    }
    public static int numSupervisor(Tree.Node currentNode){ // is also a rank getting method!!
        int i = 0;
        while(currentNode.getParent() != null){
            i++;
            currentNode = currentNode.getParent();
        }
        return i;
    }
    public static void NumberOfAllSubordinates(Tree dataTree, String entity){
        System.out.println("NumberOfAllSubordinates " +entity+ " " + numSubordinate(dataTree.search(dataTree.getRoot(), entity)));
    }
    public static int numSubordinate(Tree.Node currentNode){
        if(currentNode == null){
            return 0;
        }
        else{
            int count = 0;
            for(int i = 0; i < currentNode.getChildren().size(); i++){
                count += 1 + numSubordinate(currentNode.getChildren().get(i));
            }
            return count;
        }
    }

    public static void IsSupervisor(Tree dataTree, String entity, String supervisor){
        if(checkSup(dataTree.search(dataTree.getRoot(), entity), dataTree.search(dataTree.getRoot(), supervisor))){
            System.out.println("IsSupervisor " +entity+ " " +supervisor+ " yes" );
        }
        else{
            System.out.println("IsSupervisor " +entity+ " " +supervisor+ " no" );
        }
    }
    public static boolean checkSup(Tree.Node entity, Tree.Node supervisor){
        while(entity.getParent() != null){
            if(entity.getParent().getElement().equals(supervisor.getElement())){
                return true;
            }
            entity = entity.getParent();
        }
        return false;
    }

    public static void IsSubordinate(Tree dataTree, String entity, String subordinate){
        if(checkSub(dataTree.search(dataTree.getRoot(), entity), dataTree.search(dataTree.getRoot(), subordinate))){
            System.out.println("IsSubordinate " +entity+ " " +subordinate+ " yes" );
        }
        else{
            System.out.println("IsSubordinate " +entity+ " " +subordinate+ " no" );
        }
    }
    public static boolean checkSub(Tree.Node entity, Tree.Node subordinate) {
        if(entity == null) {
            return false;
        }
        for(int i = 0; i < entity.getChildren().size(); i++) {
            if(entity.getChildren().get(i).getElement().equals(subordinate.getElement())) {
                return true;
            }
            if(checkSub(entity.getChildren().get(i), subordinate)) {
                return true;
            }
        }
        return false;
    }

    public static void CompareRank(Tree dataTree, String entity1, String entity2){
        int rank1 = numSupervisor(dataTree.search(dataTree.getRoot(), entity1));
        int rank2 = numSupervisor(dataTree.search(dataTree.getRoot(), entity2));
        if(rank1 > rank2){ //1 is higher than 2
            System.out.println("CompareRank " +entity1 +" "+ entity2 + " lower");
        }
        else if(rank1 < rank2){
            System.out.println("CompareRank " +entity1 +" "+ entity2 + " higher");
        }
        else{
            System.out.println("CompareRank " +entity1 +" "+ entity2 + " same");
        }
    }
    public static void ClosestCommonSupervisor(Tree dataTree, String entity1, String entity2){
        /*
        Method:
        -takes the parents of one of them
        -if the parent is also the parent of entity2, returns this parent
        -otherwise, checks the parent of the parent */
        System.out.println("ClosestCommonSupervisor " + entity1 +" "+ entity2 + " " + findClosestCommonSupervisor(dataTree.search(dataTree.getRoot(), entity1),dataTree.search(dataTree.getRoot(), entity2)));
    }
    public static String findClosestCommonSupervisor(Tree.Node entity1, Tree.Node entity2){
        Tree.Node currParent = entity1.getParent();
        while(currParent != null){
            if(checkSub(currParent, entity2)){
                return currParent.getElement();
            }
            currParent = currParent.getParent();
        }
        return "none";
    }
}
