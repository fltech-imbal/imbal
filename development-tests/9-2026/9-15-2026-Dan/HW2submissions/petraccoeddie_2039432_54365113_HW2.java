
/**
 * Eddie Petracco III
 * epetracco2025@my.fit.edu
 * CSE 2010
 * Section 3
 * This is checking if an inputted list of words has combination phrases that are palindromes  
 */

import java.util.Scanner;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
public class HW2
{
    // Method for checking if the phrase is a palindrome
    public static boolean IsPalindrome(String word, int high, int low) {
        if (high > low) {
            // checking each character to equal each other 
            if (word.substring(low - 1, low).equals(word.substring(high - 1, high))) {
                return IsPalindrome(word, high - 1, low + 1); 
            }
            // If not, the base case concludes it is not a palindrome
            return false;
        }
        // other base case for successfully making it through the whole phrase
        return true;
    }
    
    // Method for making the phrases
    public static void PhraseMaker(ArrayList<String> initialWords, int targetLength, boolean[] used, ArrayList<String> currentPhrase) {
        // Base case: phrase reaches targetLength
        if (currentPhrase.size() == targetLength) {
            // creates the phrase as one big string
            StringBuilder raw = new StringBuilder();
            for (String word : currentPhrase) {
                raw.append(word);
            }

            String combined = raw.toString();
            
            // if the word is a palindrome, it prints it out in proper spaced out format
            if (IsPalindrome(combined, combined.length(), 1)) {
                System.out.println(String.join(" ", currentPhrase));
            }
            return; // otherwise return because the word is not a palindrome and go back through the phrase making portion.
        }

        // Trys every word in the inital words list
        for (int i = 0; i < initialWords.size(); i++) {
            if (!used[i]) {
                // sets the word to being used and gets the first word from the list
                used[i] = true;
                currentPhrase.add(initialWords.get(i));

                // recursive call
                PhraseMaker(initialWords, targetLength, used, currentPhrase);

                // after the phrases are made from the first word, backtracks to reset the list and move onto the next word
                currentPhrase.remove(currentPhrase.size() - 1);
                used[i] = false;
            }
        }
    }
    
    public static void main(String[] args) {
        String filePath = args[0];
        ArrayList<String> initialWords = new ArrayList<>();
        
        // file reader to take the inputted words
        try (Scanner fileScanner = new Scanner(new File(filePath))) {
            int phraseLength = fileScanner.nextInt();
            // makes a an arrayList of the given words in the input file
            while (fileScanner.hasNext()) {
                String word = fileScanner.next();
                initialWords.add(word);
            }
            // sorts the words in alphabetical order for convience
            Collections.sort(initialWords);
            
            // Creating a list of booleans to tell what words are used already
            boolean[] used = new boolean[initialWords.size()];
            ArrayList<String> currentPhrase = new ArrayList<>();
            
            // main recursive call for making the phrases and then outputting them as palindromes
            PhraseMaker(initialWords, phraseLength, used, currentPhrase);
        }
        catch (FileNotFoundException e) {
            System.err.println("File name: " + filePath + " was not found!"); 
        }
    }
}