import java.util.ArrayList; 
import java.util.Arrays;

public class Tree { 
    public Node root;
    
    
    public static class Node {
        Node parent; 
        String value;
        
        ArrayList<Node> children = new ArrayList<Node>(); 

        Node(Node parent, String value) {
            this.parent = parent; 
            this.value = value; 
            if (parent != null) {
                parent.children.add(this); 
            } 
        } 

        public Node find(String value) {
            if(value.equals(this.value)) {
                return this;
            } else {
                for (int i = 0; i < children.size(); i++) {
                    Node foundChild = children.get(i);  
                    Node match = foundChild.find(value); 
                    if (match != null) {
                        return match;
                    }

                } 
            } 
            return null; 
            
        }
    } 

    public Tree(String rootData) {
        root = new Node(null, rootData);
    }


    //addChild(node, childNode) 
    public void addChild(String parentValue, String childValue) {
        Node parent = root.find(parentValue); 
        new Node(parent, childValue); 
    }

    //getChildren(node) 
    public String getChildren(String parentValue) {
        Node parent = root.find(parentValue); 
        String childrenString = ""; 
        if (parent.children.size() == 0) {
            return "none";
        } else {
          for (int i = 0; i < parent.children.size(); i++) {
            if (i == parent.children.size()-1) {
                childrenString += parent.children.get(i).value;
            } else {
               childrenString += parent.children.get(i).value + " "; 
            }
            
          }
          return childrenString;  
        } 

    } 

    public String getGrandChildren(Node parent, String childrenString) {
        
        for (Node child : parent.children) {
            for (Node grandchild : child.children ) {
                childrenString += grandchild.value + " ";
            } 
        } 
        String[] childrenArray = childrenString.trim().split(" "); 
        Arrays.sort(childrenArray); 
        
        return String.join(" ", childrenArray);
    }

    //getParent(node) 
    public String getParent(String childValue) {
        System.out.println("Looking for: " + childValue);
        Node child = root.find(childValue); 
        //DEBUG
        
        System.out.println("Found: " + child);
        
        if (child.parent == null) {
            return null;
        } else { 
            return child.parent.value;
        }

    }


}
