import java.util.Scanner;
import java.io.File; 
import java.io.FileNotFoundException;
import java.util.ArrayList;

/*
 * Author: Calli Campagna 
 * Email:ccampagna2025@code01.fit.edu
 * Course:CSE2010
 * Section:1-4
 * Description of this file: 
 * This program organizes data into a hierarchical structure using a tree structure. The first 
 * line is the root of the tree and then every input after is either a supervisor or a subordinate.
 * The program processes each query and prints information about the supervisors and subordinates.*/
public class HW3{
/*
 * The main method reads the organization and query files from the files and build 
 * the organizational tree. The organiation file helps creates TreeNode objects and connect 
 * the supervisors and subordinates. The query files is then read one at a time. Depeding on
 * query it says it depedns what method the main method calls to output the correct informations. 
 * parameter- the files in the command line 
 */   
    public static void main(String[] args)throws FileNotFoundException{
        /*
         * This first block takes the infroamtion from the organization file and creates a root 
         * node and then continues until the file is empty creating the supervisors and subordinates.
         * each pair is to creat the appropriate relationship parent and child relationship in 
         * the organizational tree.
         * variables: tree- tree stores the organizational hierarchy 
         * sc- scanner that gets the data from the organization file 
         * rootName- variable that holds the name from file that becomes the name of the root 
         * supervisorName- TreeNode that holds supervisor from the file 
         * subordinateNAme - TreeNode that holds subordinate from the file 
         * 
         */
        Tree tree = new Tree();
        Scanner sc = new Scanner(new File(args[0]));
        String rootName = sc.next();
        tree.root = new TreeNode(rootName);
        while(sc.hasNext()){
          String supervisorName = sc.next();
          String subordinateName = sc.next();
          /*
           * This block searches the tree for the supervisor and subordinate names
           * that were read from the organization file. If the subordinate has not
           * already been added to the tree, a new TreeNode is created for it.
           * The subordinate is then added as a child of the supervisor using the
           * addChild method. Then the sc scanner closes 
           */
          TreeNode supervisor = tree.findNode(tree.root, supervisorName);

          TreeNode subordinate = tree.findNode(tree.root, subordinateName);

           if (subordinate == null) {
               subordinate = new TreeNode(subordinateName);
           }

          tree.addChild(supervisor, subordinate);
          
          
        }
          
        sc.close();
        /*
         * after the organization file closes the quert file opens . The program reads one query at
         * a time until there are no more queries. In this file the first word decides the oepration 
         * that is performed. The words after the first word depedn on the query and uses the tree 
         * method to output the answer
         */
        Scanner scanner = new Scanner(new File(args[1]));
        while(scanner.hasNext()){
            String query = scanner.next();
            /*
             * if the first word in the command is DirectSupervisor the program finds the entitys immediate parent in the tree.
             * Evert node except for the root has exactyl one direct supervisor. The getParent method is used to retrieve this parent.
             * if the entity is the root the parent is null, so it prints none. Else the name of the direct supervisor is printed 
             * after the entity name. 
             */
            if (query.equals("DirectSupervisor")){
                String entityName = scanner.next();

                TreeNode entity = tree.findNode(tree.root, entityName);

                TreeNode supervisor = tree.getParent(entity);
                
                if (supervisor == null) {
                    System.out.println("DirectSupervisor " + entity.name + " none");
                } else {
                    System.out.println("DirectSupervisor " + entity.name + " " + supervisor.name);
                }
                /*
                 * DirectSubordinates finds all of the entity's immediate children.
                 * The getChildren method returns the ArrayList containing those children.
                 * Since children are stored in alphabetical order when they are added. If the entity has no children 
                 * it prints none
                 */
            }else if(query.equals("DirectSubordinates")){
                String entityName = scanner.next();

                TreeNode entity = tree.findNode(tree.root, entityName);

                 ArrayList<TreeNode> children= tree.getChildren(entity);
                 System.out.print("DirectSubordinates" + " " + entity.name);
                 if (children.size() == 0){
                     System.out.print(" none");
                 }else{
                 for (int i =0; i < children.size(); i++){
                     System.out.print( " " + children.get(i).name);
                 }
                }
                System.out.println();
                
                /*
                 * AllSupervisors finds every supervisor above the selected entity. The getAllSupervisors method begins with 
                 * the entitys parent and continues moving upward through the parent references until reaching the root.
                 * The supervisors are stored from the closest to the farthest and if the entity is the root the resulting list 
                 * is empty and none is printed
                 * 
                 */
            }else if( query.equals("AllSupervisors")){
                String entityName = scanner.next();

                TreeNode entity = tree.findNode(tree.root, entityName);
                ArrayList<TreeNode> supervisors = tree.getAllSupervisors(entity);
                
                System.out.print("AllSupervisors " + entity.name);
                
                if (supervisors.size() == 0){
                   System.out.print(" none");
                   
               }else{
                   for (int i =0; i < supervisors.size(); i++){
                       System.out.print(" " + supervisors.get(i).name);
                   }
               }
               System.out.println();
               /*
                * AllSubordinates finds every subordinate below the selected entity. The getAllSubordinates method uses recursion 
                * to visit each child and each child decendents, creating a preorder traversal of the organizational tree.
                * if the entity has no subordinates the program prints none.
                */
            }else if(query.equals("AllSubordinates")){
                String entityName = scanner.next();
                TreeNode entity = tree.findNode(tree.root, entityName);
                
                ArrayList<TreeNode> subordinates = tree.getAllSubordinates(entity);
                
                System.out.print("AllSubordinates " + entity.name);
                
                if (subordinates.size() == 0){
                   System.out.print(" none");
                   
               }else{
                   for (int i =0; i < subordinates.size(); i++){
                       System.out.print(" " + subordinates.get(i).name);
                   }
               }
               System.out.println();
               /*
                * NumberOfAllSupervisors counts the number of supervisors above an entity. First the program gets the complete
                * lsit of supervisors using getAllSupervisors. Then it uses a loop to count the number of elements in that ArrayList.
                * The final count represents how many levels of supervivors are above the selected entity
                */
            }else if (query.equals("NumberOfAllSupervisors")){
                String entityName = scanner.next();
                TreeNode entity = tree.findNode(tree.root, entityName);
                
                ArrayList<TreeNode> supervisors = tree.getAllSupervisors(entity);
                int count = 0;
                for (int i =0; i< supervisors.size(); i++){
                    count +=1; 
                }
                System.out.println("NumberOfAllSupervisors" + " " + entity.name + " " + count);
                /*
                 * NumberOfAllSubordinates counts every subordinate below the entity. The program first calls getAllSubordinates 
                 * to create a list containing direct and inderect subordinates. A loop counts the elements in that list
                 * the resulting number is printed along with the name of entity. 
                 */
            }else if (query.equals("NumberOfAllSubordinates")){
                String entityName = scanner.next();
                TreeNode entity = tree.findNode(tree.root, entityName);
                
                ArrayList<TreeNode> subordinates = tree.getAllSubordinates(entity);
                int count = 0;
                for (int i =0; i< subordinates.size(); i++){
                     count +=1; 
                }
                System.out.println("NumberOfAllSubordinates" + " " + entity.name + " " + count);
                
                /*
                 * IsSupervisor determines whether one entity is a supervisor of another.It starts with the first entitys parent 
                 * and moved upward through the tree. If the maybe supervisor is foudn then the method returns true lese it 
                 * until the root is reached it returns false. The false is ouputed as no and true is yes
                 * 
                 */
            }else if(query.equals("IsSupervisor")){
                String entityName = scanner.next();
                String supervisorName = scanner.next();
                
                TreeNode entity = tree.findNode(tree.root, entityName);
                TreeNode maybeSupervisor = tree.findNode(tree.root, supervisorName);
                
                boolean result = tree.isSupervisor(entity, maybeSupervisor);
                if (result == true){
                    System.out.println("IsSupervisor " + entity.name + " " + maybeSupervisor.name + " yes");
                }else{
                    System.out.println("IsSupervisor " + entity.name + " " + maybeSupervisor.name + " no");
                }
                
                /*
                 *IsSubordinate checks whether one entity is a subordinate of another entity.
                 *The program reads the names of the entity and the possible subordinate
                 *from the query file. It then finds both entities in the tree using findNode.
                 *The isSubordinate method is called to check whether the second entity
                 *appears anywhere below the first entity then method returns true if the 
                 *subordinate is found and false otherwise. The progrma prints yes when true and no when false
                 *
                 */
            
            }else if(query.equals("IsSubordinate")){
                String entityName = scanner.next();
                String subordinateName = scanner.next();
                
                TreeNode entity = tree.findNode(tree.root, entityName);
                TreeNode maybeSubordinate = tree.findNode(tree.root, subordinateName);
                
                boolean  result = tree.isSubordinate(entity, maybeSubordinate);
                
                if (result == true){
                    System.out.println("IsSubordinate " + entity.name + " " + maybeSubordinate.name + " yes");
                }else{
                    System.out.println("IsSubordinate " + entity.name + " " + maybeSubordinate.name + " no");
                }
                /*
                 * ComoareRank comapres the position of two entities in the organization. It gets both 
                 * of them then calcualtes the fepth of each using calculateDepth method, which finds
                 * how far it is from the root. Then it comapares the depth and if the first rank has a 
                 * higher depth then it is higher, if its smaller its lower, or their the same. 
                 */
            }else if (query.equals("CompareRank")){
                String entityName1 = scanner.next();
                String entityName2 = scanner.next();
                
                TreeNode entity1 = tree.findNode(tree.root, entityName1);
                TreeNode entity2 = tree.findNode(tree.root, entityName2);
                
                int depth1 = tree.calculateDepth(entity1);
                
                int depth2 = tree.calculateDepth(entity2);
                
                
                if (depth1 < depth2){
                    System.out.println("CompareRank " + entity1.name + " " + entity2.name + " higher");
                }else if(depth1 > depth2){
                System.out.println("CompareRank " + entity1.name + " " + entity2.name  + " lower");
               }else{
                   System.out.println("CompareRank " + entity1.name + " " + entity2.name + " same");
               }
               /*
                * ClosestCommonSupervisor finds the closest supervisor shared by two entities. 
                * The commonSupervisor method creates a list of supervisors for each entity and compares
                * the two lists using a double for loop. Since the lists are on order from closest supervisor to 
                * the root the first mathcing supervisor is the closest. If none are foudn the program
                * prints none 
                */
            }else if (query.equals("ClosestCommonSupervisor")){
                String entityName1 = scanner.next();
                String entityName2 = scanner.next();
                
                TreeNode entity1 = tree.findNode(tree.root, entityName1);
                TreeNode entity2 = tree.findNode(tree.root, entityName2);
                
                TreeNode common = tree.commonSupervisor(entity1, entity2);
                
                if (common == null){
                     System.out.println("ClosestCommonSupervisor " + entity1.name + " " + entity2.name + " none");
                }else{
                     System.out.println("ClosestCommonSupervisor " + entity1.name + " " + entity2.name + " " + common.name);
                }
                
                
           }
        }
}
}
/*
 * The tree class represents the organizational hierarchy as a tree structure. Each TreeNode created
 * represents an entity and stores its name, parent and a list of subordinates. The tree class also conencts t
 * to the methods used in the main method for specific queries. 
 */
  class Tree{
       /*
        * The root variable stores the first entity the top of the organizational tree. Its 
        * used as the starting point for searching the rest of the entities. The constructor method 
        * creates a new tree object and sets root to null becasue the entity has not been read from file yet
        */
       TreeNode root;
        
        public Tree(){
            root = null;
        }
        /*
         * Adds the subordinates to the supervisors lsit of children also in alphabetical order. The method 
         * finds the correct position for the child and then adds it and sets the parent to the specified 
         * supervisor 
         * parameters: 
         * parent - the TreeNode will be the supervisor of the child 
         * child - the TreeNOde that will be added as a subordinate of the parent 
         */
        public void addChild(TreeNode parent, TreeNode child){
            int i = 0;

            while (i < parent.children.size() && parent.children.get(i).name.compareTo(child.name) < 0) {
              i++;
            }
            parent.children.add(i, child);
            child.parent = parent;
        }
        /*
         * gets the direct supervisor of the specified entity by returning the paretn stored in that entitys 
         * TreeNode
         * Parameters:
         * node - the TreeNode whose supervisor is wanted 
         * return - the TreeNode representing that wnated supervisor 
         */
        public TreeNode getParent(TreeNode node){
          return node.parent;
        }
        /*
         * gets direct subordinates of the specified entity by returning the arrayList containing 
         * that entitys children 
         * parameters:
         * node- the TreeNode whose direct subordinates are being requested 
         * returns - arrayList containing the subordinates of the node 
         */
        public ArrayList<TreeNode>getChildren(TreeNode node){
            return node.children;
        }
        /*
         * finda the closest supervisor shared by two entities. The method gets the list of supervisors 
         * for each entity and comapres the two lsits suing a double for loop. The first matching supervisor 
         * is returned 
         * Parameters- entity1- the first TreeNode being compared
         * entity2 - the second TreeNode being compared
         */
        public TreeNode commonSupervisor(TreeNode entity1, TreeNode entity2){
            ArrayList<TreeNode> supervisors1 = getAllSupervisors(entity1);
            ArrayList<TreeNode> supervisors2 = getAllSupervisors(entity2);
            
            for(int i = 0; i< supervisors1.size(); i++){
                    for(int j = 0; j< supervisors2.size();j++){
                    if (supervisors1.get(i).equals(supervisors2.get(j))){
                        return supervisors1.get(i);
                    }
                   }
                 }
                 return null;
        }
        /* 
         * calculates how far the entity is from the root of the tree. The method starts atvthe given 
         * node and follows its parent references until it reaches the root counting each level along the way
         * parameter- the TreeNode whose depth in the tree is being calculated 
         * returns- the integer that represents the level between the node and the root, the depth 
         */
        public int calculateDepth(TreeNode node){
            int count = 0;
            TreeNode current = node;
                
            while(current.parent != null){
                    count +=1;
                    current = current.parent; 
                }
            return count; 
        }
        /*
         * Finds all supervisors above a specified entity. The method starts with the entitys direct 
         * supervisor and follows parent upward until it reaches the root. Stored closest to farthest 
         * parameter- node- the TreeNode whose supervisors are being found 
         * return- an arrayList containing all supervisors of the node 
         */
        public ArrayList<TreeNode>getAllSupervisors(TreeNode node){
            ArrayList<TreeNode> supervisors = new ArrayList<>();
            
            TreeNode current = node.parent;
            
            while (current !=null){
                supervisors.add(current);
                current = current.parent;
            }
            return supervisors; 
        }
        /*
         * Checks whether one entity is a supervisor of another entity. The method
         * starts with the first entity's direct supervisor and follows parent references upward. if 
         * found the method returns true otherwise its false 
         * parameters: entity- The TreeNode whose supervisors are being checked 
         * returns- true or false dependign on if maybeSupervisor is a supervisor 
         */
        public boolean isSupervisor(TreeNode entity, TreeNode maybeSupervisor){
            TreeNode current = entity.parent; 
            while (current != null){
                if(current == maybeSupervisor){
                    return true;
                }
                current = current.parent;
            }
            return false; 
        }
        /*
         * Checks whether one entity is a subordinate of another entity.The method checks the direct 
         * subordinates and recursively searches their subordinates and if found below its true else its false 
         * parameter- entity- the TreeNode whose subordinates are being searched 
         * returns- true or false dependign on if maybeSubordinates is a subordinate 
         */
        public boolean isSubordinate(TreeNode entity, TreeNode maybeSubordinate){
            for (int i = 0; i < entity.children.size(); i++) {
                 TreeNode current = entity.children.get(i);

                if (current == maybeSubordinate) {
                    return true;
                }

                if (isSubordinate(current, maybeSubordinate)) {
                     return true;
                 }
             }
            return false;
        }
        /*
         * finds all subordinates below a specific entity and the method checks each direct subordinates
         * and recursevley searches the subordinates subordinates. Each one is added to an arrayList
         * parameters- the TreeNode whose subordinates are being found 
         * returns - the arrayList contaaining all subordinates 
         */
        public ArrayList<TreeNode>getAllSubordinates(TreeNode node){
            ArrayList<TreeNode> subordinates = new ArrayList<>();
            
            for(int i = 0; i < node.children.size(); i++){
                TreeNode child = node.children.get(i);
                subordinates.add(child);
                subordinates.addAll(getAllSubordinates(child));
            }
            return subordinates; 
        }
        /* 
         * searches the organizational tree for an entity with a specified name. It checks the current node 
         * and if its not it, it recusively searches each of its subordinates until its found 
         * parameters: current - the TreeNode where the search begins 
         * name - the name of the entity being searched for 
         * returns: the TreeNode with the specified name or null
         */
          public TreeNode findNode(TreeNode current, String name){
            if (current.name.equals(name)) {
           return current;
          }

           for (int i = 0; i < current.children.size(); i++) {

            TreeNode result = findNode(current.children.get(i), name);

            if (result != null) {
              return result;
           }
          }

          return null;

        }
        
}
/*
 * creates a new TreeNode for an entity in  the organizational tree and the node stores the entitys name 
 * starts with null as parent and creates an empty arrayList for subordinates 
 * paramters: name - the name of the entity
 * 
 */
class TreeNode{
        String name;
        TreeNode parent;
        ArrayList<TreeNode>children;
        
        public TreeNode(String name){
         this.name = name;
         this.parent = null;
         this.children = new ArrayList<>();
        }
        
}
