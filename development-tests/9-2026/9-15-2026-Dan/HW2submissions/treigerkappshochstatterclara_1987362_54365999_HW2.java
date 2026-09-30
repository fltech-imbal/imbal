/*

  Author: Clara Treiger
  Email: mtreigerkapp2024@my.fit.edu
  Course: CSE2010
  Section: 01
  Description of this file: HW2 Programming

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Collections;

public class HW2
{
    public static void main(String[] args)
    {
    /*
      Variables:
      filename stores the name of the input file.
      palPhraseLength stores the number of words required
      in each palindrome phrase.
      words stores all of the words read from the input file.
      used keeps track of which words have been selected.
      palindromes stores all valid palindrome phrases.
    */
        String filename = args[0];
    /*
      This block opens the input file using the filename
      provided through the command line. It reads the
      required palindrome phrase length from the first
      line of the file and then creates an ArrayList to
      store the words that follow.
    */
        try
        {
            Scanner input = new Scanner(new File(filename));

            int palPhraseLength = input.nextInt();

            ArrayList<String> words = new ArrayList<String>();
            /*
              This block reads each word from the input file
              and adds it to the words ArrayList. After all
              words have been read, the input file is closed.
            */
            while (input.hasNext())
            {
                words.add(input.next());
            }

            input.close();
            /*
              This block creates an array to track which words
              have been used and an ArrayList to store the
              palindrome phrases found by the recursive search.
              The recursive method is then called to begin
              generating phrases.
            */
            boolean[] used = new boolean[words.size()];

            ArrayList<String> palindromes =
                new ArrayList<String>();

            findPalindromes(words, used,
                            new ArrayList<String>(),
                            palPhraseLength, palindromes);
            /*
              This block sorts all palindrome phrases in
              lexicographical order and then prints each
              palindrome on its own line.
            */
            Collections.sort(palindromes);

            for (String palindrome : palindromes)
            {
                System.out.println(palindrome);
            }
        }
        catch (FileNotFoundException e)
        {
            System.out.println("File not found.");
        }
    }
    /*
      Recursively generates all possible phrases using unique
      words. words contains the input words, used tracks which
      words have already been selected, current stores the
      phrase being built, palPhraseLength is the required
      number of words, and palindromes stores valid results.
    */
    public static void findPalindromes(
        ArrayList<String> words,
        boolean[] used,
        ArrayList<String> current,
        int palPhraseLength,
        ArrayList<String> palindromes)
    {
        /*
          This is the base case of the recursive search. When
          the current phrase contains the required number of
          words, the words are combined into one string and
          checked to determine whether the phrase is a
          palindrome.
        */
        if (current.size() == palPhraseLength)
        {
            /*
              This block combines the words currently stored
              in current into one phrase. A space is placed
              between consecutive words so that the resulting
              string has the same format as the required output.
            */
            String phrase = "";

            for (int i = 0; i < current.size(); i++)
            {
                if (i > 0)
                {
                    phrase += " ";
                }

                phrase += current.get(i);
            }
            /*
              This block checks the completed phrase using the
              recursive palindrome method. If the phrase is a
              palindrome, it is added to the list of results.
              The return statement ends this recursive branch.
            */
            if (isPalindrome(phrase))
            {
                palindromes.add(phrase);
            }

            return;
        }
        /*
          This block tries each word that has not already been
          used. The selected word is added to the current phrase,
          and the method recursively continues building the
          phrase. After returning from recursion, the word is
          removed and marked unused so another possibility can
          be explored.
        */
        for (int i = 0; i < words.size(); i++)
        {
            if (!used[i])
            {
                used[i] = true;
                current.add(words.get(i));

                findPalindromes(words, used, current,
                                palPhraseLength, palindromes);

                current.remove(current.size() - 1);
                used[i] = false;
            }
        }
    }
    /*
      Checks whether a phrase is a palindrome. The phrase is
      cleaned by removing spaces and punctuation and converting
      letters to lowercase. The cleaned string is then passed
      to the recursive palindrome-checking method.
    */
    public static boolean isPalindrome(String phrase)
    {
        /*
          This block removes spaces and punctuation and converts
          the phrase to lowercase so that the palindrome check
          compares only the characters that matter.
        */
        String cleaned = phrase.replaceAll("[^a-zA-Z0-9]", "")
                               .toLowerCase();

        return isPalindromeRecursive(
            cleaned, 0, cleaned.length() - 1);
    }
    /*
      Recursively checks whether the characters at the left and
      right positions match. If they do, the method moves both
      positions toward the center and checks again. The method
      returns false when characters do not match and true when
      the middle of the string is reached.
    */
    public static boolean isPalindromeRecursive(
        String phrase, int left, int right)
    {
        /*
          This is the base case. When the left and right
          positions meet or cross, all necessary character
          comparisons have been completed, so the string is
          a palindrome.
        */
        if (left >= right)
        {
            return true;
        }
        /*
          This block compares the characters at the two ends
          of the current portion of the string. If they are
          different, the string cannot be a palindrome.
        */
        if (phrase.charAt(left) != phrase.charAt(right))
        {
            return false;
        }
        /*
          This recursive call moves the left position one
          character forward and the right position one
          character backward, continuing the palindrome check
          toward the center of the string.
        */

        return isPalindromeRecursive(
            phrase, left + 1, right - 1);
    }
}
