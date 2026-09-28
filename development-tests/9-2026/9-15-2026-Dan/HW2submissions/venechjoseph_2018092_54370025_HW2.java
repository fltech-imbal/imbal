/*
 * Author: Joseph Venech
 * Email: jvenech2025@my.fit.edu
 * Course: CSE 2010
 * Section: 1
 * Description: This program uses recursion to find unique palindromes from a list of words
 *       and prints them in alphabetical order.
 * 
 */
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class HW2 {

    // Method to check if a string is a palindrome
    public static boolean isPalindrome(String str, int left, int right) {

        // Base case - crossed or reached the middle
        if (left >= right) {
            return true;
        }

        // If characters don't match then it is not a palindrome
        if (str.charAt(left) != str.charAt(right)) {
            return false;
        }

        
        return isPalindrome(str, left + 1, right - 1);
    }

    // Method to find palindromes
    public static void findPalindromes(String[] words, boolean[] used, String[] current, int depth, int palPhraseLength) {

        // Base case
        if (depth == palPhraseLength) {

            // Build the phrase
            StringBuilder phrase = new StringBuilder();

            for (int i = 0; i < palPhraseLength; i++) {
                phrase.append(current[i]);
            }

            // Check if the phrase is a palindrome
            String phraseString = phrase.toString();

            if (isPalindrome(phraseString, 0, phraseString.length() - 1)) {

                // Print the words with spaces
                for (int i = 0; i < palPhraseLength; i++) {
                    if (i > 0) {
                        System.out.print(" ");
                    }
                    System.out.print(current[i]);
                }

                System.out.println();
            }

            return;
        }

        // Try every word
        for (int i = 0; i < words.length; i++) {

            // Make sure each word is only used once
            if (!used[i]) {

                used[i] = true;
                current[depth] = words[i];

                // Select the next word with recursion
                findPalindromes(words, used, current, depth + 1, palPhraseLength);

                // Backtrack
                used[i] = false;
            }
        }
    }

    public static void main(String[] args) throws FileNotFoundException {

        // Get filename from command line
        String filename = args[0];

        Scanner input = new Scanner(new File(filename));

        // First line = number of words in palindrome
        int palPhraseLength = input.nextInt();

        // Read remaining words
        String[] words = new String[0];

        while (input.hasNext()) {
            String word = input.next();

            words = Arrays.copyOf(words, words.length + 1);
            words[words.length - 1] = word;
        }

        input.close();

        // Sort words for lexicographical order
        Arrays.sort(words);

        // Keep track of which words have been used
        boolean[] used = new boolean[words.length];

        // Stores the current phrase
        String[] current = new String[palPhraseLength];

        // Start search
        findPalindromes(words, used, current, 0, palPhraseLength);
    }
}