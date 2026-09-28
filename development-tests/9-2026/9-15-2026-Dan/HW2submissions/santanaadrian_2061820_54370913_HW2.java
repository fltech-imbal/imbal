// Author: Adrian Santana
// Email: santanaa2025@my.fit.edu
// Course: CSE 2010
// Section: 1
// Description: The program shown below evaluates the words provided by an input file and uses puzzle solve to output the
// combinations that form palindromes.
import java.util.Scanner;
import java.util.ArrayList;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Collections;

public class HW2 {
    public static void main(String[] args) throws FileNotFoundException {
        // Make a scanner object that takes in the input file
        File inputFile = new File(args[0]);
        Scanner scanner = new Scanner(inputFile);
        int palPhraseLength = scanner.nextInt();

        // Create an empty list that stores the words provided by the scanner
        ArrayList<String> inputWords = new ArrayList<>();

        // Create a while loop that reads the words from the input file and adds them to the inputWords list
        while (scanner.hasNext()) {
            inputWords.add(scanner.next());
        }
        // Create an empty list that stores the palindromes found
        ArrayList<String> foundPalindromes = new ArrayList<>();
        findPalindromes(palPhraseLength, new ArrayList<String>(), inputWords, foundPalindromes);

        // Sort the foundPalindromes list
        Collections.sort(foundPalindromes);

        // Make a for loop that takes each palindrome phrase store in foundPalindromes and prints them line by line
        for (String phrase : foundPalindromes) {
            System.out.println(phrase);
        }
    }
    // Create a boolean method that takes a String (word) as a parameter and returns true of the String is a palindrome
    // or false otherwise
    public static boolean isPalindrome(String word) {
        // If the word is less than 2 letters, it is a palindrome
        if (word.length() < 2) {
            return true;

            // If the first character is a space, recall the method without the first character
        } else if (word.charAt(0) == ' ') {
            return isPalindrome(word.substring(1));

            // If the last character is a space, recall the method without the last character
        } else if (word.charAt(word.length() - 1) == ' ') {
            return isPalindrome(word.substring(0, word.length() - 1));
    } else {
            if (word.charAt(0) == word.charAt(word.length() - 1)) {
                return isPalindrome(word.substring(1, word.length() - 1));
            } else {
                return false;
            }
        }

    }
    // Create a void method that takes an int (wordsRemaining) and ArrayLists (currentPhrase, availableWords, palindromesFound)
    // and uses puzzle solve to determine the word combinations that from palindromes
    public static void findPalindromes(int wordsRemaining, ArrayList<String> currentPhrase, ArrayList<String> availableWords, ArrayList<String> palindromesFound) {
        for (int i = 0; i < availableWords.size(); i++) {
            String word = availableWords.get(i);
            // Make a copy of currentPhrase and availableWords so that they are not changed when running the code
            ArrayList<String> updatedCurrentPhrase = new ArrayList<>(currentPhrase);
            ArrayList<String> updatedAvailableWords = new ArrayList<>(availableWords);

            // Add the word to the updatedCurrentPhrase ArrayList and remove it from the updatedAvailableWords ArrayList
            updatedCurrentPhrase.add(word);
            updatedAvailableWords.remove(i);

            // If there is only one word left in the wordsRemaining ArrayList
            if (wordsRemaining == 1) {
                String phrase = String.join(" ", updatedCurrentPhrase);

                // If the phrase is a palindrome, add it to the palindromesFound ArrayList
                if (isPalindrome(phrase)) {
                    palindromesFound.add(phrase);
                }
                // If wordsRemaining is not 1 and the phrase is not a palindrome, recall the method with the updated parameters
            } else {
                findPalindromes(wordsRemaining - 1, updatedCurrentPhrase, updatedAvailableWords, palindromesFound);
            }

        }
    }
}