/* 
	Author: Ciara Curtis
	Email: ccurtis2025@my.fit.edu
	Course: CSE2010
	Section: 4
	Description of this file: Homework 2 submission that uses the scanner class to read a file,
		finds palindromes with a desired phrase length, and then spits it back out to the
		command line in alphabetical order.
*/


import java.util.Scanner;
import java.io.File;

public class HW2 {
    public static SinglyLinkedList2 palindromes = new SinglyLinkedList2();
    //Holds and sorts the final palindromes

    //Checks to make sure a given string is a palindrome (ignores spaces)
    public static boolean palindromeCheck(int low, int high, String phrase) {
        if(low > high) return true;
        /* Base case: if you have gotten to this point, then the word is a palindrome because
           the function has recursively returned true for (begin == end) */

        char begin = phrase.charAt(low);                        //Stores the starting character
        char end = phrase.charAt(high);                         //Stores the ending character

        if(begin == ' ') begin = phrase.charAt(++low);
        //if begin character equals space, go up to next index
        if(end == ' ') end = phrase.charAt(--high);
        //if end character equals space, go down to next index


        if(begin != end) return false;
        else return palindromeCheck(++low, --high, phrase);
        /*Decomposition: if begin equals end, then the index of low and high are updated
          and the new characters are compared. */
    }
    
    //creates the palindromes by combining words into a string that is checked by the palindromeCheck method
    public static void palindromeMaker(int length, SinglyLinkedList2 sorted, SinglyLinkedList2 unsorted) {
	SinglyLinkedList2 smallUnsorted = unsorted;           //A smaller unsorted list
	for (int i = 1; i <= unsorted.size(); i++) {          //iterates over every element in the unsorted list
	    sorted.addLast(smallUnsorted.removeFirst());      //chooses the first element in the unsorted list
	    if(length == 1) {                                 //if all the elements needed have been chosen, then PalindromeCheck will check the string
		String toCheck = sorted.toString();
		if(palindromeCheck(0,toCheck.length() - 1, toCheck)) palindromes.addElement(toCheck);
	    }
	    else palindromeMaker(length - 1, sorted, smallUnsorted);
            smallUnsorted.addLast(sorted.removeLast());
	}
    }

    public static void main(String[] args) throws Exception {
        Scanner scanner = new Scanner(new File(args[0]));                       //Reads a file from the command line
	int phraseLength = scanner.nextInt();
	SinglyLinkedList2 wordList = new SinglyLinkedList2();                   //Holds the initial word list from file
	SinglyLinkedList2 sorted = new SinglyLinkedList2();		        //empty list for palindromeMaker method

	while(scanner.hasNext()) wordList.addLast(scanner.next());              //while loop to move words from file to linked list
        scanner.close();                                                        

	palindromeMaker(phraseLength, sorted, wordList);
	palindromes.toPrint();
    }
}
