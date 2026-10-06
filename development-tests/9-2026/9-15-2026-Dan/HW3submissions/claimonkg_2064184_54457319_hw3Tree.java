import java.util.Scanner; 
import java.io.File; 
import java.io.FileNotFoundException;

public class hw3Tree { 
    public static Tree supervisor; 

    public static Tree getTree() {
        return supervisor;
    } 

    
    //This assignment is asking to make a tree of the supervisor heritate, and then ask questions about if that director is there
       public static void main(String args[]){
        
        Scanner kb; 

        try { // a try catch exception for file use
            kb = new Scanner(new File("C:\\Users\\claim\\OneDrive\\Desktop\\java test\\hw3input1.txt")); 
        } 
        catch(FileNotFoundException e) {
            System.out.println("File was not found"); 
            return;
        } 

        String worker = kb.next();  

        String result = "";

        supervisor = new Tree(worker); 

        System.out.println("[" + supervisor.getChildren(worker) + "]");

        while (kb.hasNext()) {
            supervisor.addChild(kb.next(), kb.next()); 
        }  
        
        System.out.println(supervisor.getChildren(worker)); 
        System.out.println(supervisor.getGrandChildren(supervisor.root.find(worker), result));

        kb.close();
    }

}