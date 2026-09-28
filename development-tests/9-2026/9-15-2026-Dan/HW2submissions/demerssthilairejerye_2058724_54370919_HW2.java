/*
Author: Jerye Demers-St.Hilaire
Email: jdemerssthil2025@my.fit.edu
Course: CSE2010
Section: 1
Description of this file: Evaluting the input using a recursive algorithim to test if the string is a plaindrome 
*/

import java.util.Scanner;
import java.util.LinkedList;
import java.io.File;
import java.io.FileNotFoundException;

public class HW2 {

  static LinkedList<String> possiblePalindromes = new LinkedList<>();
  static LinkedList<String> palindromes = new LinkedList<>();
  static LinkedList<String> palindromePhrase = new LinkedList<>();
  static int palPhraseLength;

public static void main(String[] args) throws FileNotFoundException {

 Scanner scanner = new Scanner(new File(args[0]));

  String input = scanner.nextLine();
  possiblePalindromes.add(input);

  // Turning the first line into an int
  // Storing it as palPhraseLength 
  // Removing it from possible palindromes  
  palPhraseLength = Integer.parseInt(possiblePalindromes.get(0));
  possiblePalindromes.remove(0);
  
  while (scanner.hasNextLine()) {       
     input = scanner.nextLine();
     possiblePalindromes.add(input);
  }
  
  findPalindromes(); 
 
   //Compares the letters of the palindromes to sort alphabetically
   for (int i = 0; i < palindromes.size() - 1; i++) {
    for (int j = i + 1; j < palindromes.size(); j++) {
     if (palindromes.get(i).compareTo(palindromes.get(j)) > 0) {
      String temp = palindromes.get(i);
      palindromes.set(i, palindromes.get(j));
      palindromes.set(j, temp);
     }
    }
   }


  //Prints the output
  for (int i = 0; i < palindromes.size(); i++) {
      System.out.println(palindromes.get(i));
  } 
}       
   
   // Checking if the palindrome phrase is the input length
   // Combining the palindromes into one phrase
   // Checking if phrase is a palindrome
   static void findPalindromes() {
    if (palindromePhrase.size() == palPhraseLength) {
        String phrase = "";
    for (int i = 0; i < palindromePhrase.size(); i++) {
        phrase = phrase + palindromePhrase.get(i) + " ";
    }
    if (isPalindrome(phrase)) {
        palindromes.add(phrase);
     }
     return;
     }
    
                                              
     // Checking if the palindrome is a duplicate
     for (int i = 0; i < possiblePalindromes.size(); i++) {
        String posP = possiblePalindromes.get(i);
        boolean duplicate = false;
     for (int j = 0 ; j < palindromePhrase.size(); j++) {
       if (posP.equals(palindromePhrase.get(j))) {
           duplicate = true;
       }
     }
       if (duplicate == false) {
          palindromePhrase.add(posP);
          findPalindromes();
          palindromePhrase.removeLast();
       }   
     }
    
                                                                                                                      }
   // Removing all spaces and punctionation from the word
   // Checking if the loop is in the middle of the word
   // Checking if the next letters match
   // If they dont, return its not a palindrome 
   // If true repeat the process 
   static boolean isPalindrome(String word) {
    
    String wordNoSP = word.replace(" ", "" );
    String NSP = wordNoSP.replaceAll("[^a-zA-Z]", "" );
    
    if (NSP.length() == 1 || NSP.length() == 0) {
     return true;
    }
    
    String f = NSP.substring(0,1);
    String e = NSP.substring(NSP.length()-1);
    
    if (!e.equals(f)) {
     return false;
    }
    else {
     String sS = NSP.substring(1,(NSP.length()-1));
     return isPalindrome(sS);    
     }
    }
 
 }

