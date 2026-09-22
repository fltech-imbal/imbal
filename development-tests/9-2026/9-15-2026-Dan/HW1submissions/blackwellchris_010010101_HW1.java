/*

  Author: Chris B
  Email: cblackwell2025@my.fit.edu
  Course: cse 2010
  Section: 05
  Description of this file: simulates chat requests and assignments

 */
import java.util.Scanner;
import java.io.File;
import java.io.IOException;


public class HW1
{
	private static SinglyLinkedList availablereps = new SinglyLinkedList();
	private static SinglyLinkedList occupiedreps = new SinglyLinkedList();
	private static SinglyLinkedList onhold = new SinglyLinkedList();
	private static SinglyLinkedList sessions = new SinglyLinkedList();
	private static void CR(String a){
		System.out.println(a);
		String[] words= a.split("\\s+");
		if(!availablereps.isEmpty()){
			RA(words);
			//add(words[2]);
		}else if(words[3].equals("wait")){
			POH(words);
		}else TL(words);
	}
	
	private static void CE(){
		availablereps.addFirst(occupiedreps.removeFirst());//this will NOT work with hw1out2. fix this when you get the chance!
		System.out.println("ChatEnded");
	}
	
	private static void QOH(){
		System.out.println("QuitOnHold");
	}
	
	private static void PARL(String a){
		//String[] words= a.split("\\s+");
		//System.out.print("AvailableRepList "+words[1]);
		//while(availablereps.getElement()!=null){
		//	System.out.print(" "+availablereps.getNext());
		//}
		System.out.println("fixlater");
	}
	
	private static void PMWT(){
		System.out.println("PMWT");
	}
	
	private static void RA(String[] a){
		occupiedreps.addFirst(availablereps.removeFirst());
		System.out.println("RepAssignment "+a[2]+" "+occupiedreps.first()+" "+a[1]);
	}
	
	private static void POH(String[] a){
		System.out.println("POH");
	}
	
	private static void TL(String[] a){
		System.out.println("TL");
	}
	
	private static Scanner in;
    /*
      Description of each method, including parameters 
    */
    public static void main(String[] args)
    {
	/*check if file is inputed via command line, exiting if no file found. if file is found, import it*/
	if (args.length==0){
		System.out.println("no input");
		System.exit(1);
	}
	Scanner reqs=null;
	try{
		File in=new File(args[0]);
		reqs=new Scanner(in);
	} catch (IOException ioException){
		System.err.println("can't open file");
		System.exit(1);
	}
	availablereps.addLast("Alice");
	availablereps.addLast("Bob");
	availablereps.addLast("Carol");
	availablereps.addLast("David");
	availablereps.addLast("Emily");
	String line="";
	while(reqs.hasNextLine()){
		line=reqs.nextLine();
		if(line.contains("ChatRequest")){
			CR(line);
		}else if(line.contains("ChatEnded")){
			CE();
		}else if(line.contains("QuitOnHold")){
			QOH();
		}else if(line.contains("PrintAvailableRepList")){
			PARL(line);
		}else if(line.contains("PrintMaxWaitTime")){
			PMWT();
		}else{
			System.err.println("unrecognized instruction:" + reqs.nextLine());
			System.exit(1);
		}
			
	}
		//read input line i;
		//call appropiate function;
	/* description of each block (around 5-10 lines) of instructions */
    }

}
