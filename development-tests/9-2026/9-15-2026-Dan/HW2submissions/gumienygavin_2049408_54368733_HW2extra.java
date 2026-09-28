/*

  Author: Gavin Gumieny
  Email: ggumieny2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
  This file finds multi-word palindromes (non-recursively) given an input file containing the required words in a phrase
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

public class HW2extra
{
    // List to store the found palindromes.
    private static List<String> foundPalindromes = new ArrayList<>();
    
    /** Checks each character in the string to see if they match.
     *  @param phrase  the word being checked
     *  @return  true if there are no mismatches
     */
    public static boolean isPalindrome(String phrase) {
        // If phrase is null return false.
        if (phrase == null) { return false; }
        
        // Returns true if phrase is one word or less.
        if (phrase.length() <= 1) { return true; }
        
        // Low and high variables to keep track of indexes we want to compare.
        int low = 0;
        int high = phrase.length() - 1;
        
        // While the low index is less than the high index we walk through the
        // low and high indexes and compare their characters
        while (low < high) {
            // Compares characters from opposite sides of the phrase
            if (phrase.charAt(low) != phrase.charAt(high)) {
                return false;
            }
            
            // Increases low and high by one (we look to the next index)
            low++;
            high--;
        }
        
        // Returns true if every index matches
        return true;
    }
    
    /** Recursively find all of the possible phrases that can be put together
     *  and checked to see if they are palindromes.
     *  @param palPhraseLength  the required length of each possible phrase
     *  @param unassigned  the list of unused words
     * 
     */
    public static void findPalindromes(int palPhraseLength, List<String> unassigned) {
        // A stack to store possible combinations of assigned words.
        Stack<List<String>> assignedWords = new Stack<>();
        
        // Push something on the stack to start the loop.
        assignedWords.push(new ArrayList<>());
        
        // While the stack is not empty 
        while (!assignedWords.isEmpty()) {
            List<String> assigned = assignedWords.pop();
            
            // The phrase is long enough
            if (assigned.size() == palPhraseLength) {
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
                if (isPalindrome(phraseNoSpace)) {
                    foundPalindromes.add(phrase); // adds a found palindrome to the found list
                }
            }
            // The phrase is not long enough so we must add more words to the phrase
            else {
                for (String word : unassigned) {
                    // Checks if each word in the unassigned list has been added to the assigned list
                    if (!assigned.contains(word)) {
                        // Creates a duplicate of the current assigned list and adds the new word to the assigned list
                        List<String> nextAssigned = new ArrayList<>(assigned);
                        nextAssigned.add(word);
                        assignedWords.push(nextAssigned);
                    }
                }
            }
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
            findPalindromes(palPhraseLength, wordList);
            
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
