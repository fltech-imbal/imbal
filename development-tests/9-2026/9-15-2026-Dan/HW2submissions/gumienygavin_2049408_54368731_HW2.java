/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
  This file recursively finds multi-word palindromes given an input file containing the required words in a phrase
  followed by the list of words. It does this by finding all possible combinations of phrases with a given length
  of words and checks each combination to see if it reads the same forward and backwards (palindrome). The file takes an
  input file name as a command-line argument and reads each input line within the file.

 */
import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.io.File;
import java.io.FileNotFoundException;

public class HW2
{
    // List to store the found palindromes.
    private static List<String> foundPalindromes = new ArrayList<>();
    
    /** Recursively checks each character in the string to see if they match.
     *  @param phrase  the word being checked
     *  @param low  the first character in the word
     *  @param high  the last character in the word
     *  @return  true if there are no mismatches
     */
    public static boolean isPalindrome(String phrase, int low, int high) {
        // If phrase is null return false.
        if (phrase == null) { return false; }
        
        // Returns true if phrase is one word or less.
        if (phrase.length() <= 1) { return true; }
        
        // Base case: low and high are equal or cross over
        if (low >= high) {
            return true;
        }
        // Compares characters from opposite sides of the phrase
        else if (phrase.charAt(low) != phrase.charAt(high)) {
            return false;
        }
        // Recursive call: the current indexes of the string match so we check
        // the next indexes
        else {
            return isPalindrome(phrase, low + 1, high - 1);
        }
    }
    
    /** Recursively find all of the possible phrases that can be put together
     *  and checked to see if they are palindromes.
     *  @param palPhraseLength  the required length of each possible phrase
     *  @param assigned  the list of chosen words
     *  @param unassigned  the list of unused words
     * 
     */
    public static void findPalindromes(int palPhraseLength, List<String> assigned, List<String> unassigned) {
        // Base case: we reach the end of our phrase
        // Takes each word in the unassigned list and assigns it as a possibility for the next call
        for (int i = 0; i < unassigned.size(); i++) {
            String word = unassigned.get(i);
            assigned.add(word);
            unassigned.remove(i);
            
            // Base case: The phrase is long enough
            if (palPhraseLength == 1) {
                // Combines all of the words in the phrase into a single string
                StringBuilder sb = new StringBuilder();
                for (String p : assigned) {
                    // Adds a space between each word
                    if (sb.length() > 0) {
                        sb.append(" ");
                    }
                    
                    sb.append(p);
                }
                // The phrase we will output if it is a palindrome
                String phrase = sb.toString();
                
                // The phrase without spaces between words so we can properly check
                // if it is a palindrome
                String phraseNoSpace = phrase.replace(" ", "");
                
                // After turning our phrase into a string, we check if it is a palindrome
                if (isPalindrome(phraseNoSpace, 0, phraseNoSpace.length() - 1)) {
                    foundPalindromes.add(phrase); // adds a found palindrome to the found list
                }
            }
            // Recursive call: The phrase is not long enough so we call again 
            // to add another word to the phrase
            else {
                findPalindromes(palPhraseLength - 1, assigned, unassigned);
            }
            // The word that has been used gets put back into the unassigned list
            unassigned.add(i, word);
            assigned.remove(assigned.size() - 1);
        }
    }
   
    /** The main method (entry point) for the program. Reads inputs from a given file. The first input read is the number of required words in a phrase
     *  and the rest of the inputs read are words.
     * 
     *  @param args  an array of Strings passed into the program (in this case a single String element, the input file name).
     *  @throws FileNotFoundException  if the attempt to access the file through args fails.
     */
    public static void main(String[] args)
    {
        try {
            // Checks if there were any arguments passed into the program
            if (args.length == 0) {
                throw new FileNotFoundException("Enter a file name!");
            }
            
            // Scanner variable to handle the input file.
            Scanner s = new Scanner(new File(args[0]));

            // Stores the number of required words for the palindrome phrase.
            int palPhraseLength = s.nextInt();
            
            // List to store words provided by the input.
            List<String> wordList = new ArrayList<>();
           
            // Checks if there are lines after the phrase length number.
            if (s.hasNextLine()) {
                s.nextLine();      // Shifts to next line
                
                // Reads the rest of the lines from the input file.
                while (s.hasNextLine()) {
                    String word = s.nextLine().trim();
                    // Adds the word if a word has been retrieved and the word does not exist in the list
                    if (!word.isEmpty() && !wordList.contains(word)) {
                        wordList.add(word);
                    }
                }
            }
            
            // Finds all possible phrases (of a given length) that are palindromes (read the same forward and backward).
            findPalindromes(palPhraseLength, new ArrayList<>(), wordList);
            
            // Sorts the list of words read from the input file.
            Collections.sort(foundPalindromes);
            
            // Prints out each phrase that was found
            for (String phrase : foundPalindromes) {
                System.out.println(phrase);
            }
        }
        // An exception that catches if the file to be read is not found.
        catch (FileNotFoundException ex){
            System.err.println("Error: File not found. " + ex.getMessage());
        }
    }
}
