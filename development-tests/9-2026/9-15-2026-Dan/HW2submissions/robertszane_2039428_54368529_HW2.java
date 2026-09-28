/*
Author: Zane Roberts
Email: zroberts2025@fit.edu
Course: CSE2010
Section: Section 3
Description of this file: A recursive program that sorts and checks palindromes of a certain phrase length
*/

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class HW2 {
    
    /* checks if a word or phrase is a plaindrome by seeing if the first and last char equal each other then
       calling the method again with the first and last char removed until the string is less than or equal
       to 1. Only parameter is str which stores the string for each word */
    private static boolean Palindrome(String str) {
       if (str.length() <= 1) {
         return true;
       } 
       if (str.charAt(0) != str.charAt(str.length() - 1)) {
         return false;
       }
       return Palindrome(str.substring(1, str.length() - 1));  
    }
    public static void main(String[] args) {
       /* checks if the file was passed in the terminal */
       if (args.length == 0) {
        return;
       }
       
       /* Read the filename from the first command-line argument */
       String input = args[0];
       
       /* creates an array with the availible words */
       List<String> wordList = new ArrayList<>();
       int palPhraseLength = 0;
       
       /* scans the input file for the plaPhraseLength number */
       try (Scanner scan = new Scanner(new File(input))) {
         if (scan.hasNextInt()) {
            palPhraseLength = scan.nextInt();
         }
         while (scan.hasNext()) {
            wordList.add(scan.next());
         }
          /* catch for is it can't find a file in the aurgument in the terminal */
       } catch (FileNotFoundException e) {
           System.err.println("Error: File not found - " + input);
           return;
       }
       
        /* uses collections to sort the availible word list */
       Collections.sort(wordList);
       
        /* makes another matching array for boolean tracking */
       boolean[] used = new boolean[wordList.size()];
       List<String> results = new ArrayList<>();
       
        /* starts recursion */
       uniquePhrases(wordList, palPhraseLength, new ArrayList<>(), used, results);
       
        /* prints results */
       System.out.println("Palindromes found: " + results);
    }
    
    /* this method removes the gaps between words so Palindrome method can properly check if it's a palindrome, 
       forms every possible cobination of words in the list, and checks if a word has already been used so it
       can't be repeated in the same phrase. */
    private static void uniquePhrases(
          /* This parameter contains all avaible words from the input file */
          List<String> wordList,
          /* This parameter stores what the phrase length of each palindrome should be*/
          int palPhraseLength,
          /* This parameter stores each phrase that matches the number of words for palPhraseLength*/
          List<String> currentPhrase,
          /* This parameter stores words that were already used so the same word isn't used twice in the same Phrase*/
          boolean[] used,
          /* This parameter stores completed palindromes*/
          List<String> results
    ) {
       
       /* removes spaces from the phrase then calls Palindrome to check if it is a palindrome
          then adds the spaces back so the results look neat */
       if (currentPhrase.size() == palPhraseLength) {
           String combined = String.join("", currentPhrase);
           if (Palindrome(combined)) {
               results.add(String.join(" ", currentPhrase));            
           }
           return;
       }

       /* a for loop that checks if a word has been used and if it hasent it slots it in to the phrase
          after removing the old one to make a new phrase */
       for (int i = 0; i < wordList.size(); i++) {

           if (used[i]) {
              continue;
           }

           if (i > 0 && wordList.get(i).equals(wordList.get(i - 1)) && !used[i - 1]) {
              continue;
           }

           used[i] = true;
           currentPhrase.add(wordList.get(i));

           uniquePhrases(wordList, palPhraseLength, currentPhrase, used, results);

           currentPhrase.remove(currentPhrase.size() - 1);
           used[i] = false;
       }
        
    }
}
