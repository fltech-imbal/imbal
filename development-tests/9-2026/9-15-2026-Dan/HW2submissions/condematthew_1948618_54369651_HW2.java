/*

  Author: Matthew Conde
  Email: mconde2022@my.fit.edu
  Course: CSE 2010
  Section: 2
  Description of this file:	palindrome checker, takes a word or combination of words and determines if it can be spelled the same forward and backward
				this excludes spaces and punctuation. uses recursive functions to iterate through strings from the outside in. for phrases,
				takes the length of the phrase and a list of words and prints all unique permutations of words that are palindromes (excluding 
				spaces and punctuation). phrase palindrome check uses word palindrome check in recursive calls.

 */

import java.io.File;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Arrays;

public class HW2
{

    /*
	method checks if a string is a palindrome, base case if string empty or one character, automatically counts as palindrome
	if not, compare first and last letter of string for equality until reach the middle letter (base case) or no letters left (base case)
    */

    public static boolean palindromeCheck (String word){
	
	String checkword = word;
	String afterPhrase= null;
	if (checkword.length() == 0){return true;}	// base cases

	if (checkword.length() == 1){
	    return true;
	}

	int index = checkword.indexOf(" ");		// see where first instance of a space is, iterate until no spaces left in string (can apply to punctuation easily)
        if (index >= 0){

            afterPhrase = checkword.substring(0,index) + checkword.substring(index+1,checkword.length());
            return palindromeCheck(afterPhrase);
        }

	if (checkword.charAt(0) == checkword.charAt(checkword.length() - 1)){
	    return palindromeCheck(checkword.substring(1, checkword.length()-1));		// main comparison recursive call, check first & last characters, 
	}											// return recursive call of substring excluding first & last
	return false;										// if base case not reached, return false
    }

	/* 
	    recursive function to find a palindrome phrase given a list of words, and length of phrase. iterate through each word and test every permutation of other strings
	    boolean tracker used to make sure all words in a sentence/phrase are unique. if the phrase reaches the max size (L) test if it is a palindrome using above function
	*/

    public static void palindromePhrase(String[] words, ArrayList<String> currentPhrase, boolean[] used,int L){
		

        if(currentPhrase.size() == L){
								// if reach size L in phrase, join list into one string (" " delimiters) then pass to palindromeCheck
	    String phrase = String.join(" ", currentPhrase);

            if(palindromeCheck(phrase)){
                System.out.println(phrase);			// if palindrome, print phrase (with spaces included)
            }
            return;
         }       

        for (int i = 0; i<words.length; i++){		// main loop through words, pick a word, recursive call to pick second,third...Lth, repeat until all permutations attempted
            if (!used[i]){
                used[i] = true;
                currentPhrase.add(words[i]);

                palindromePhrase(words, currentPhrase, used, L);
		
                currentPhrase.remove(currentPhrase.size()-1);		// remove word from the phrase once all permutations reached with "x" word in "i" spot
                used[i] = false;					// mark index for word as not used, unique word to next permutation
            }

	}
      

    }

    public static void main(String[] args) throws Exception
    {

	File inputFile = new File(args[0]);
	Scanner scnr = new Scanner(inputFile);  

//	System.out.print(palindromeCheck("frorf"));
//	System.out.print(palindromeCheck("taco"));		//test for palindromeCheck with simple strings 

	int L = scnr.nextInt();					// pull length of phrase from input file
            
        ArrayList<String> wordList = new ArrayList<String>();		// create array list to hold strings, using array list for ease of add/remove (dynamic list size)
        while (scnr.hasNext()) {
            wordList.add(scnr.next());				// add next word to list from input
        }
        scnr.close();
	
	String[] words = wordList.toArray(new String[0]);	// convert array list to array
        boolean[] used = new boolean[words.length];		// create boolean list to make sure all words unique in a permutation
	Arrays.sort(words);					// sort so output alphabetical
        ArrayList<String> currentPhrase = new ArrayList<>();	// arraylist to dynamically add/remove words from phrase

        palindromePhrase(words, currentPhrase, used, L);	// start search for plalindrome phrase


    }

}
