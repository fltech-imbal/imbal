
/*

  Author:THE CARSON JAMES ROCHA
  Email:crocha2025@my.fit.edu
  Course: CSE 2010
  Section:3 (made me check coursicle for that)
  Description of this file:
  This program reads words from a file and finds multi-word palindromes.
  It uses recursion to make the phrases and recursion to check palindromes.
  and btw i strongly dislike recusrion

 */


import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class HW2
{
    /*
      main reads the file gets  phrase length stores the words
      sorts  words and starts  recursive search.
    */
    public static void main(String[] args)
    {
        if (args.length != 1)
        {
            System.out.println("Usage: java HW2 filename");
            return;
        }

        ArrayList<String> words = new ArrayList<String>();
        int palPhraseLength;

        try
        {
            Scanner input = new Scanner(new File(args[0]));

            palPhraseLength = input.nextInt();

            while (input.hasNext())
            {
                words.add(input.next());
            }

            input.close();
        }
        catch (FileNotFoundException e)
        {
            System.out.println("File not found.");
            return;
        }

        Collections.sort(words);

        boolean[] used = new boolean[words.size()];

        findPalindromes(words, used, "", 0, palPhraseLength);
    }

    /*
      findPalindromes makes every possible phrase of the needed length.
      A word can only be used once in the phrase. When a phrase is complete,
      it checks whether the phrase is or isnt a  palindrome.
    */
    public static void findPalindromes(ArrayList<String> words, boolean[] used,  String phrase, int count, int length)
    {
        if (count == length)
        {
            if (isPalindrome(phrase, 0,   phrase.length() - 1))
            {
                System.out.println(phrase);
            }
            return;
        }

        for (int i = 0; i < words.size(); i++)
        {
            if (used[i] == false)
            {
                used[i ] = true;

                if (count == 0)
                {
                    findPalindromes(words, used, words.get(i), count + 1, length);
                }
                else
                {
                    findPalindromes(words, used, phrase + " " + words.get(i),
                                    count + 1, length);
                }

                used [i] = false;
            }
        }
    }

    /*
      isPalindrome recursively compares the characters at the phrases two ends.
      Spaces and punctuation are skipped. If all matching characters are
      the same, the string would be a palindrome.
    */
    public static boolean isPalindrome(String text, int left, int right)
    {
        if (left >= right)
        {
            return true;
        }

        if (Character.isLetterOrDigit(text.charAt(left)) == false)
        {
            return isPalindrome(text, left + 1, right);
        }

        if (Character.isLetterOrDigit(text.charAt(right)) == false)
        {
            return isPalindrome(text, left, right - 1);
        }

        if (Character.toLowerCase(text.charAt(left)) !=
            Character.toLowerCase(text.charAt(right)))
        {
            return false;
        }

        return isPalindrome(text, left + 1, right - 1);
    }
}
