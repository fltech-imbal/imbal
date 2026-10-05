/*
	Author: Ciara Curtis
	Email: ccurtis2025@my.fit.edu
	Course: CSE2010
	Section: 4
	Description of this file: Homework 3 submission that uses the 
		scanner class to read two files: the first containing the names
		of supervisors and subordinates that needed to be sorted into a
		traversable tree, and the second file that contains inquiries
		about the tree made from the first file. Output of the program
		is based on the second file and determining if the information
		given by the first file is correct.
*/

import java.util.Scanner;
import java.io.File;

public class HW3 {

  public static void main(String[] args) throws Exception {
    Scanner treeInput = new Scanner(new File(args[0]));
    Tree workerTree = new Tree(treeInput.next());

    while(treeInput.hasNext()) {
	workerTree.addChild(treeInput.next(), treeInput.next());
    }
    treeInput.close();


    Scanner queries = new Scanner(new File(args[1]));
    while(queries.hasNext()) {
      String queryType = queries.next();
      String worker = queries.next();

      if(queryType.equals("DirectSupervisor")) {
	System.out.println("DirectSupervisor " + worker + " " + workerTree.findParent(worker));
      }

      else if(queryType.equals("DirectSubordinates")) {
	System.out.println("DirectSubordinates " + worker + " " + workerTree.listChildren(worker));
      }

      else if(queryType.equals("AllSupervisors")) { 
        System.out.println("AllSupervisors " + worker + " " + workerTree.allParents(worker));
      }

      else if(queryType.equals("AllSubordinates")) {
        System.out.println("AllSubordinates " + worker + " " + workerTree.allChildren(worker));
      }

      else if(queryType.equals("NumberOfAllSupervisors")) {
	System.out.println("NumberOfAllSupervisors " + worker + " " + workerTree.countParents(worker));
      }

      else if(queryType.equals("NumberOfAllSubordinates")) {
	System.out.println("NumberOfAllSubordinates " + worker + " " + workerTree.countChildren(worker));
      }

      else if(queryType.equals("IsSupervisor")) {
        String worker2 = queries.next();
	System.out.print("IsSupervisor " + worker + " " + worker2 + " ");
	if(workerTree.findParent(worker).equals(worker2)) { System.out.println("yes"); }
	else System.out.println("no");
      }

      else if(queryType.equals("IsSubordinate")) {
	String worker2 = queries.next();
	System.out.print("IsSubordinate " + worker + " " + worker2 + " ");
        if(workerTree.findParent(worker2).equals(worker)) System.out.println("yes");
        else System.out.println("no");
      }

      else if(queryType.equals("CompareRank")) {
	String worker2 = queries.next();
        System.out.print("CompareRank " + worker + " " + worker2 + " ");
	if(workerTree.countParents(worker) > workerTree.countParents(worker2))
	  System.out.println("lower");
        else if(workerTree.countParents(worker) < workerTree.countParents(worker2))
	  System.out.println("higher");
        else System.out.println("same");
      }

      else if(queryType.equals("ClosestCommonSupervisor")) {
	String worker2 = queries.next();
	System.out.println(queryType + " " + worker + " " + worker2 + " " + workerTree.commonParent(worker, worker2));
      }

    }

  }

}
