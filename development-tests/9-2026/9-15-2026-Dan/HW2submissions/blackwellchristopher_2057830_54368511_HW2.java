/*

  Author: Chris B
  Email: cblackwell2025@my.fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file: find and print number of palindrome sentences of a given length using an input file

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class HW2
{
	public static boolean isPalindrome(String str) {
        return isPalindrome(str, 0, str.length() - 1);
    }

    private static boolean isPalindrome(String str, int left, int right) {

        // Base case
        if (left >= right) {
            return true;
        }
        // Ignore spaces and punctuation on the left
        if (!Character.isLetterOrDigit(str.charAt(left))) {
            return isPalindrome(str, left + 1, right);
        }
        // Ignore spaces and punctuation on the right
        if (!Character.isLetterOrDigit(str.charAt(right))) {
            return isPalindrome(str, left, right - 1);
        }
        if (Character.toLowerCase(str.charAt(left))
                != Character.toLowerCase(str.charAt(right))) {
            return false;
        }
        return isPalindrome(str, left + 1, right - 1);
    }
	public static void findPalindromes(
            ArrayList<String> words,
            int palindromeLength,
            ArrayList<String> current,
            boolean[] used,
            ArrayList<String> answers) {
        if (current.size() == palindromeLength) {

            StringBuilder phrase = new StringBuilder();

            for (String word : current) {
                phrase.append(word);
            }

            if (isPalindrome(phrase.toString())) {
                StringBuilder output = new StringBuilder();

                for (int i = 0; i < current.size(); i++) {
                    if (i > 0) {
                        output.append(" ");
                    }
                    output.append(current.get(i));
                }

                answers.add(output.toString());
            }

            return;
        }
        for (int i = 0; i < words.size(); i++) {

            if (!used[i]) {

                used[i] = true;
                current.add(words.get(i));

                findPalindromes(
                        words,
                        palindromeLength,
                        current,
                        used,
                        answers
                );
                current.remove(current.size() - 1);
                used[i] = false;
            }
        }
    }
	public static void main(String[] args) {
        //Check if input is provided
		if (args.length != 1) {
			return;
        }
		//Setup of variables and scanner
        ArrayList<String> words = new ArrayList<>();

        try {
            Scanner input = new Scanner(new File(args[0]));

            int palindromeLength = input.nextInt();
            input.nextLine();

            while (input.hasNextLine()) {
                String word = input.nextLine().trim();

                if (!word.isEmpty()) {
                    words.add(word);
                }
            }

            input.close();

            // Sort words so generated output is easier to order
            Collections.sort(words);

            ArrayList<String> current = new ArrayList<>();
            boolean[] used = new boolean[words.size()];
            ArrayList<String> answers = new ArrayList<>();

            findPalindromes(
                    words,
                    palindromeLength,
                    current,
                    used,
                    answers
            );

            // Required alphabetical / lexicographical ordering
            Collections.sort(answers);

            for (String answer : answers) {
                System.out.println(answer);
            }

        } catch (FileNotFoundException e) {
            return;
        }
    }
}
