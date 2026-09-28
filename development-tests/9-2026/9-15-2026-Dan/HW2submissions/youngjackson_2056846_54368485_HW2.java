/*
* Author: Jackson Young
* Email: young2025@my.fit.edu
* Course: CSE 2010
* Section: 1
* Description of this file: This program will take an input text file and output all possible palindromes
*   of a specified size recursively using the words in the input file. Palindromes exclude spaces.
*/
import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
public class HW2 {
    // Global variables
    static int palPhraseLength; // Keeps track of the number of words in a valid palindrome
    // ArrayList to hold all input words
    static ArrayList<String> wordList = new ArrayList<>();
    // ArrayList to hold all found palindromes
    static ArrayList<String> foundPals = new ArrayList<>();

    // Main method wil process input via Scanner and print the necessary output
    public static void main(String[] args) {
        // Safely declare Scanner to the input file using the File class
        Scanner scanner;
        try {
            // Attempt to create new Scanner and File objects
            scanner = new Scanner(new File(args[0]));
        } catch (FileNotFoundException e) {
            // If file path is incorrect, throw an error and terminate the program
            System.out.println("ERROR: FileNotFoundException:\n" + e.getMessage() + "\nProgram Terminated.");
            return;
        }

        // Read first input to be the number of words in a valid palindrome
        //  then assign the integer to global variable palPhraseLength
        palPhraseLength = scanner.nextInt();

        // Main loop to proccess input tokens and put them in the wordList
        while (scanner.hasNext()) {
            wordList.add(scanner.next());
        }

        // Find all possible palindromic phrases
        findPalindromes(new String[palPhraseLength], 0);

        // Alphabetize foundPals
        foundPals.sort(null);
        // Print foundPals
        for (int i = 0; i < foundPals.size(); i++) {
            System.out.println(foundPals.get(i));
        }
    }

    // Static method that recursively checks if a String is a palindrome
    public static boolean checkPalindrome(String str, int buffer) {
        // Remove spaces
        if (str.contains(" ")) str = str.replace(" ", "");
        // Base case: If the unchecked string's length is one or zero, it is automatically a palindrome
        if (buffer >= str.length() / 2) {
            return true;
        } else {
            // Recursive Case: Check if the char at the front is the same as the char in the back
            //  if it is, check again with a string excluding the first and last chars
            //  otherwise, the string is not a palindrome
            if (str.charAt(buffer) == str.charAt(str.length() - buffer - 1)) {
                // str could be a palindrome
                return checkPalindrome(str, buffer + 1);
            } else {
                // str is NOT a palindrome
                return false;
            }
        }
    }

    // Static method that recursively checks for palindromes of size palPhraseLength using the provided input
    //  Puts found palindromes into the foundPals ArrayList to be printed
    public static void findPalindromes(String[] usedWords, int index) {
        // Get each word in the input word list
        for (int i = 0; i < wordList.size(); i++) {
            // Check if the word is already selected
            if (!contains(usedWords, wordList.get(i))) {
                // Base Case: index == selectedWords.length-1
                // This is the last index, so there is no need to create an entirely new array
                // Instead, test the final cases here
                if (index == usedWords.length - 1) {
                    // Add word to usedWords array
                    usedWords[index] = wordList.get(i);

                    // Merge words to single String
                    String palindrome = "";
                    for (int j = 0; j < usedWords.length; j++) {
                        palindrome += usedWords[j];
                        if (j < usedWords.length - 1) palindrome += " ";
                    }
                    
                    // Check if this completed phrase is a palindrome
                    if (checkPalindrome(palindrome, 0)) {
                        foundPals.add(palindrome);
                    }
                }

                // Recursive Case: index < array.length-1
                // There are more words to be added to the phrase
                // Create a new usedWords list with the selected word and call the function again with incremented index
                else {
                    // Preserve original array
                    String[] newSelectedWords = usedWords.clone();
                    // Add selected word to array
                    newSelectedWords[index] = wordList.get(i);
                    // Check for more palindromes using the selected word
                    findPalindromes(newSelectedWords, index + 1);
                }
            }
        }
    }

    // Simple linear search method for finding an element in a given array
    public static <E> boolean contains(E[] array, E item) {
        for (int i = 0; i < array.length; i++) {
            if (!(array[i] == null) && array[i].equals(item)) {
                return true;
            }
        }
        return false;
    }
}