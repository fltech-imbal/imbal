/*
Author: Avery Haughwout
Email: ahaughowut2024@my.fit.edu
Course: CSE 2012
Section: Idk
File Description: TREE!
*/
import java.util.ArrayList;
import java.util.Collections;

public class Tree{
    //NODE DECLARATION
    public class Node implements Comparable<Node>{//For the love of god I hope every Node/node is porperly capitalized
        public ArrayList<Node> children;
        public Node parent;
        public String ele;
        public Node(ArrayList<Node> sub, Node sup, String newElement){
            this.children = sub;
            this.parent = sup;
            this.ele = newElement;
            
        }
            //METHODS
            public void addChild(Node newChild){
                this.children.add(newChild);
                Collections.sort(this.children);
            }
            public ArrayList<Node> getChildren(){
                Collections.sort(this.children);
                return this.children;
            }
            public Node getParent(){
                return this.parent;
            }
            public String getElement(){
                return this.ele;

            }
            public int compareTo(Node other){
                return this.ele.compareTo(other.ele);
            }
    }
    //TREE DECLARATION
    public Node root;

    public Tree(){
        this.root = null;
    }
    public void setRoot(String newRoot){
        Node addRoot = new Node(new ArrayList<Node>(), null, newRoot);
        this.root = addRoot;
    }

    public boolean isEmpty(){
        return this.root == null;
    }
    public void addLink(String parentNode, String childNode){
        Node newParent = search(this.root, parentNode);
        Node newChild = new Node(new ArrayList<Node>(), newParent, childNode);
        newParent.addChild(newChild);
    }
    public ArrayList<Node> getChildren(Node getNode){
        return getNode.getChildren();
    }
    public Node getParent(Node getNode){
        return getNode.getParent();
    }
    public Node getRoot(){
        return this.root;
    }

    public Node search(Node check, String entity){
        /*Search Method!
        -If the root's element is equal to the entity element, return the node
        Else, call it recursively for all of it's children
        if there are nochildren, return null*/
        if(check.getElement().equals(entity)){
            return check;
        }
        else{
            for(int i = 0; i < check.getChildren().size(); i++){
                Node result = search(check.getChildren().get(i), entity);
                if(result != null){
                    return result;
                }
            }
        }
        return null;//If you *really* can't find it
    }

    public void print(Node root){
        /*
        Objective of this method:
        -Print the root
            Node: -node-
            Children: -child1, child2, child3- OR "none"
            println()

        */
        
        System.out.println("Node: "+root.getElement());
        String childList = "Children: ";
        if(root.getChildren().isEmpty()){
            System.out.println(childList + "none");
        }
        else{
            for(int i = 0; i < root.getChildren().size(); i++)
            childList += root.getChildren().get(i).getElement() + ", ";
            System.out.println(childList);
        }
        for(int i = 0; i < root.getChildren().size(); i++){
            print(root.getChildren().get(i));
        }
        
    }
}