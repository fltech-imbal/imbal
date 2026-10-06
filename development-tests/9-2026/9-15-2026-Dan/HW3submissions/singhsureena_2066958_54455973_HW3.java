import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;


public class HW3{
    /*the HW3 class contains the main method and two classes- OrgData and Tree */
 public static void main(String[] args) {
        OrgData orgManager = new OrgData();
        //to ensure that the both files are provided
        if (args.length<2){
            System.out.println("provide the tree file and query file: ");
            return;
        }
        try{
            Scanner treeScanner =new Scanner(new File(args[0]));
            while (treeScanner.hasNextLine()){
            String line = treeScanner.nextLine().trim();
            if (line.isEmpty()) continue;
            String[] tokens = line.split("\\s+");
                if (tokens.length==1){
                    if (orgManager.tree.root ==null){
                        orgManager.tree.addRoot(tokens[0]);
                    }
                }else if (tokens.length >=2){
                    orgManager.tree.insert(tokens[0],tokens[1]);
                }
            } treeScanner.close();//closing the data file
            Scanner queryScanner = new Scanner(new File(args[1]));
            while (queryScanner.hasNextLine()){
                String line = queryScanner.nextLine().trim();
                if (line.isEmpty()) continue;
                String[] tokens = line.split("\\s+");
                String firstWord = tokens[0];
                /*Using switch to direct the input to the right method */
                switch (firstWord){
                    case "DirectSupervisor":
                        orgManager.DirectSupervisor(tokens[1]);
                        break;
                    case "DirectSubordinates":
                        orgManager.DirectSubordinates(tokens[1]);
                        break;
                    case "AllSupervisors":
                        orgManager.AllSupervisors(tokens[1]);
                        break;
                    case "NumberOfAllSupervisors":
                        orgManager.NumberOfAllSupervisors(tokens[1]);
                        break;
                    case "AllSubordinates":
                        orgManager.AllSubordinates(tokens[1]);
                        break;
                    case "NumberOfAllSubordinates":
                        orgManager.NumberOfAllSubordinates(tokens[1]);
                        break;
                    case "IsSupervisor":
                        orgManager.IsSupervisor(tokens[1],tokens[2]);
                        break;
                    case "IsSubordinate":
                        orgManager.IsSubordinate(tokens[1],tokens[2]);
                        break;
                    case "CompareRank":
                        orgManager.CompareRank(tokens[1],tokens[2]);
                        break;
                    case "ClosestCommonSupervisor":
                        orgManager.ClosestCommonSupervisor(tokens[1],tokens[2]);
                        break;
                    default:
                        break;
                }
            }
            queryScanner.close();//close the query file
        } catch (FileNotFoundException e){
            System.out.println("File not found: "+ e.getMessage());
        }

    }
 public static class OrgData{
    private Tree tree = new Tree();
    /*this method prints the direct supervisors of the given entity */
    public void DirectSupervisor(String entity){
        Tree.Node candidate =tree.find(entity);
        System.out.print("DirectSupervisor "+ entity+" ");
        if (candidate ==null || candidate.getParent()==null) System.out.println( "none");
        else {System.out.println(candidate.getParent().getElement());}
    }
    /*this method prints the direct subordinates of the given entity */
    public void DirectSubordinates(String entity){
        Tree.Node candidate = tree.find(entity);
        System.out.print("DirectSubordinates "+entity);
        if (candidate ==null){
            System.out.println(" none");
            return;
        }
        List<Tree.Node> children = tree.getChildren(candidate);
        if (children.isEmpty()){
            System.out.println(" none");
        }else{
         for (int i=0; i<children.size();i++){
             System.out.print(" "+children.get(i).getElement());
         }System.out.println();
        }
    }
    /*This method prints all the supervisors of a given entity */
    public void  AllSupervisors(String entity){
        Tree.Node current = tree.find(entity);
        System.out.print("AllSupervisors "+entity);
        if (current==null){
             System.out.println(" none");
             return;
        }
        if(current.getParent()==null){
            System.out.println(" none");
             return;
        }
        while(current.getParent()!=null){
            System.out.print(" "+current.getParent().getElement());
            current = current.getParent();
        }System.out.println();
    }
    /*the AllSubordinates uses a helper method to print all the descendants*/
    public void AllSubordinates(String entity){
        Tree.Node candidate = tree.find(entity);
        System.out.print("AllSubordinates "+ entity);
        if (candidate ==null|| candidate.getFirstChild()==null){
            System.out.println(" none");
            return;
        }
        printDescendants(candidate);
        System.out.println();
    }
    private void printDescendants(Tree.Node node){
        if (node==null) return;
        List<Tree.Node> children = tree.getChildren(node);
        for (int i=0; i<children.size();i++){
            Tree.Node child = children.get(i);
            System.out.print(" "+child.getElement());
            printDescendants(child);
        }
    }
    /*this method uses a helper method to count the number of all supervisors of a given entity */
    public void  NumberOfAllSupervisors(String entity){
        Tree.Node current = tree.find(entity);
        int count=0;
        
        while(current!=null&&current.getParent()!=null){
            count+=1;
            current = current.getParent();
        }System.out.println("NumberOfAllSupervisors "+entity+" "+count);
    }
    /*this method uses a helper method to count the number of all subordinates of a given entity */
    public void NumberOfAllSubordinates(String entity){Tree.Node candidate = tree.find(entity);
        System.out.print("NumberOfAllSubordinates "+ entity+" ");
        if (candidate ==null|| candidate.getFirstChild()==null){
            System.out.println("0");
            return;
        }
        int countofDescendants=countDescendants(candidate);
        System.out.println(countofDescendants);
    }
    /*this helper method counts the number of descendants using a for loop */
    private int countDescendants(Tree.Node node){
        if (node==null) return 0;
        List<Tree.Node> children = tree.getChildren(node);
        int count =0;
        for (int i=0; i<children.size();i++){
            Tree.Node child = children.get(i);
            count+=1;
            count+= countDescendants(child);
        }
        return count;
    }
    /*this method checks if the entity is the supervisor of the subordinate */
    public void  IsSupervisor(String entity, String subordinate){
        Tree.Node entNode = tree.find(entity);
        Tree.Node subNode = tree.find(subordinate);
        System.out.print("IsSupervisor "+ entity+" "+ subordinate+" ");
        if (subNode == null || entNode ==null) {
            System.out.println("no");
            return;
        }
        boolean result = false;
        Tree.Node current = entNode.getParent();
        while(current!= null){
            if (current.getElement().equals(subordinate)){
                result= true;
                break;
            }
            current = current.getParent();
        }
        if (result) System.out.println("yes");
        else System.out.println("no");
    }
    /*IsSubordinate method checks whether the subordinate's ancestor is the entity */
    public void IsSubordinate(String entity, String subordinate){
        Tree.Node entityNode = tree.find(entity);
        Tree.Node subNode = tree.find(subordinate);
        System.out.print("IsSubordinate "+entity+" "+subordinate+" ");
        if(entityNode ==null|| subNode==null){
            System.out.println("no");
            return;
        }
        boolean result = false;
         Tree.Node current = subNode.getParent();
         while(current != null){
            if (current.getElement().equals(entity)){
                result = true;
                break;
            }
            current = current.getParent();
         }
         if (result){ System.out.println("yes");}
         else{System.out.println("no");}
    }
    /* The CompareRank method uses a helper method depth to find the depth of the second entity with the first */
    public void CompareRank (String entity1, String entity2){
        Tree.Node node1 = tree.find(entity1);
        Tree.Node node2 = tree.find(entity2);
        System.out.print("CompareRank "+entity1+" "+entity2+" ");
        if (node1==null || node2 ==null){
            System.out.println("none");
            return;
        }
        int depth1 = getDepth(node1);
        int depth2 = getDepth(node2);
        if (depth1< depth2){
            System.out.println("higher");
        } else if (depth1>depth2){
            System.out.println("lower");
        }else System.out.println("same");
    }
    private int getDepth(Tree.Node node){
        int depth =0;
        Tree.Node current = node.getParent();
        while (current != null){
            depth++;
            current = current.getParent();
        }return depth;
    }/*ClosestCommonSupervisor method uses ArrayLists to store the ancestors of both entities and checks if one of the entities contains the other. */
    public void ClosestCommonSupervisor (String entity1, String entity2){
        Tree.Node node1 = tree.find(entity1);
        Tree.Node node2 = tree.find(entity2);
        System.out.print("ClosestCommonSupervisor "+entity1+" "+entity2+" "); 

        if (node1==null|| node2==null){
            System.out.println("none");
            return;
        }
        List<String> ancestors1 = new ArrayList<>();
        Tree.Node current1 = node1.getParent();
        while (current1 != null){
            ancestors1.add(current1.getElement());
            current1 = current1.getParent();
        }
        Tree.Node current2 = node2.getParent();
        while (current2!= null){
            if (ancestors1.contains(current2.getElement())){
                System.out.println(current2.getElement());
                return;
            }current2 = current2.getParent();
        }
        System.out.println("none");
    }
    
 }
 public static class Tree{
    /*this class creates a tree using nodes. It has nodes such as parent, firstChild, nextSibling and element.
    It has methods such as getChildren , addRoot, getParent, addChild, find etc */
    protected static class Node{
        private String element;
        private Node parent;
        private Node firstChild;
        private Node nextSibling;
        
        public Node(String e, Node above){
            this.element=e;
            this.parent= above;
            this.firstChild= null;
            this.nextSibling = null;
        }
        public String getElement(){return element;}
        public Node getParent(){return parent;}
        public Node getFirstChild(){return firstChild;}
        public Node getNextSibling(){return nextSibling;}

        public void setElement(String e){element =e;}
        public void setParent(Node parentNode){ parent = parentNode;}
        public void setFirstChild(Node node){firstChild = node;}
        public void setNextSibling(Node node){nextSibling = node;}


    }
    protected Node root = null;
    private int size =0;
    public Tree(){}

    public int size(){return size;}
    public boolean isEmpty(){return size ==0;}
    public Node root(){return root;}

    public Node addRoot(String e){
        if(!isEmpty()) throw new IllegalStateException("Tree is not empty");
        root = new Node(e,null);
        size =1; return root;
    }
    //this is a helper method that we use when we get string values from the input files and need to convert it into nodes before 
    //redirecting it to methods that require two values
    public void insert(String parentName, String childName){
        Node parentNode = find(parentName);
        if (parentNode!= null){
            Node childNode = new Node(childName, parentNode);
            addChild(parentNode, childNode);
        }
    }
    public void addChild(Node parentNode,Node childNode){
        if (parentNode ==null) throw new IllegalArgumentException("parent cannot be null.");
        childNode.setParent(parentNode);
        size++;
        String childName = childNode.getElement();
        //if parent has no children yet, or new child comes before the first child alphabetically
        if (parentNode.getFirstChild()==null || childName.compareTo(parentNode.getFirstChild().getElement())<0){
            childNode.setNextSibling(parentNode.getFirstChild());
            parentNode.setFirstChild(childNode);
        }else{
            Node current = parentNode.getFirstChild();
            while (current.getNextSibling()!=null &&childName.compareTo(current.getNextSibling().getElement())>0){
                current= current.getNextSibling();
            }
            childNode.setNextSibling(current.getNextSibling());
            current.setNextSibling(childNode);
        }
    }
    public List<Node> getChildren(Node node){
        List<Node> childrenList = new ArrayList<>();
        if (node==null) return childrenList;
        Node current = node.getFirstChild();
        while(current != null){
            childrenList.add(current);
            current = current.getNextSibling();
        } return childrenList;
    }
    public Node getParent(Node node){
        if (node== null) return null;
        return node.getParent();
    }
    public Node find(String name){
        return findRecursive(root,name);
    }
    private Node findRecursive(Node current, String name){
        if (current ==null) return null;
        if(current.getElement().equals(name)) return current;
        Node found = findRecursive(current.getFirstChild(), name);
        if (found!=null) return found;
        return findRecursive(current.getNextSibling(), name);
    }
    
    }

}