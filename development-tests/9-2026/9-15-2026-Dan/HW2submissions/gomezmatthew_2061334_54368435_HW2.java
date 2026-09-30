/*

  Author: Matthew Gomez
  Email: matthewgomez2025@fit.edu
  Course: CSE 2010
  Section: 4
  Description of this file:
    Reads a list of words and a phrase length (palPhraseLength) from an
    input file, then recursively finds every multi-word phrase of that
    length (using each word at most once) whose letters, read left to
    right with the spaces removed, form a palindrome. Matching phrases
    are printed in alphabetical order, one per line.

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;

public class HW2
{
    /*
      isPalindrome(String s, int left, int right)
      Recursively checks whether s reads the same forward and backward
      between index "left" and index "right" (inclusive).
      Parameters:
        s     - the string being checked
        left  - current index moving in from the front
        right - current index moving in from the back
      Returns true if the substring s[left..right] is a palindrome.
    */
    public static boolean isPalindrome(String s, int left, int right)
    {
	/* Base case: the two pointers met or crossed, so every pair of
	   characters checked so far matched -> it is a palindrome. */
	if (left >= right)
	{
	    return true;
	}

	/* If the outer characters do not match, it cannot be a palindrome. */
	if (Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right)))
	{
	    return false;
	}

	/* Recursive case: the outer characters matched, so move both
	   pointers inward and check the rest of the string. */
	return isPalindrome(s, left + 1, right - 1);
    }

    /*
      isPalindrome(String s)
      Convenience wrapper that starts the recursive check across the
      whole string s.
    */
    public static boolean isPalindrome(String s)
    {
	return isPalindrome(s, 0, s.length() - 1);
    }

    /*
      findPhrases(String[] words, boolean[] used, String[] currentPhrase,
                  int depth, int palPhraseLength, List<String> results)
      Recursively builds every ordered phrase of palPhraseLength unique
      words chosen from words[], and records the phrase (with spaces
      between words, for output) whenever the concatenated letters form
      a palindrome.
      Parameters:
        words           - the full list of available words
        used            - tracks which words are already part of the
                          phrase currently being built
        currentPhrase   - the words chosen so far, in order
        depth           - how many words have been placed so far
        palPhraseLength - target number of words in a phrase
        results         - collects every palindromic phrase found
    */
    public static void findPhrases(String[] words, boolean[] used, String[] currentPhrase,
				    int depth, int palPhraseLength, List<String> results)
    {
	/* Base case: a full-length phrase has been assembled.
	   Build the printable phrase (words separated by spaces) and the
	   concatenated phrase (no spaces) used for the palindrome test. */
	if (depth == palPhraseLength)
	{
	    StringBuilder phrase = new StringBuilder();
	    StringBuilder concatenated = new StringBuilder();
	    for (int i = 0; i < palPhraseLength; i++)
	    {
		if (i > 0)
		{
		    phrase.append(" ");
		}
		phrase.append(currentPhrase[i]);
		concatenated.append(currentPhrase[i]);
	    }

	    if (isPalindrome(concatenated.toString()))
	    {
		results.add(phrase.toString());
	    }
	    return;
	}

	/* Recursive case: try every not-yet-used word in the next slot
	   of the phrase, recurse to fill the remaining slots, then undo
	   the choice (backtrack) so the word is free for other phrases. */
	for (int i = 0; i < words.length; i++)
	{
	    if (!used[i])
	    {
		used[i] = true;
		currentPhrase[depth] = words[i];

		findPhrases(words, used, currentPhrase, depth + 1, palPhraseLength, results);

		used[i] = false;
	    }
	}
    }

    public static void main(String[] args)
    {
	/* description of variables
	   fileName        - input file given as a command-line argument
	   scanner         - reads the input file line by line
	   palPhraseLength - number of words required in each phrase (first line of input)
	   wordList        - every candidate word read from the input file
	   words           - wordList converted to an array for recursion
	   used            - parallel array marking words already placed in the phrase being built
	   currentPhrase   - holds the phrase currently being assembled
	   results         - every palindromic phrase that was found
	*/

	if (args.length < 1)
	{
	    System.out.println("Usage: java HW2 <input file>");
	    return;
	}

	String fileName = args[0];
	List<String> wordList = new ArrayList<String>();
	int palPhraseLength = 0;

	/* Read palPhraseLength from the first line, then read one word
	   per line until the end of the file. */
	try
	{
	    Scanner scanner = new Scanner(new File(fileName));

	    palPhraseLength = Integer.parseInt(scanner.nextLine().trim());

	    while (scanner.hasNextLine())
	    {
		String word = scanner.nextLine().trim();
		if (!word.isEmpty())
		{
		    wordList.add(word);
		}
	    }
	    scanner.close();
	}
	catch (FileNotFoundException e)
	{
	    System.out.println("File not found: " + fileName);
	    return;
	}

	/* Convert the word list to an array and set up the bookkeeping
	   arrays needed by the recursive phrase builder. */
	String[] words = wordList.toArray(new String[0]);
	boolean[] used = new boolean[words.length];
	String[] currentPhrase = new String[palPhraseLength];
	List<String> results = new ArrayList<String>();

	findPhrases(words, used, currentPhrase, 0, palPhraseLength, results);

	/* Sort the palindromic phrases alphabetically and print them,
	   one per line. */
	Collections.sort(results);
	for (String phrase : results)
	{
	    System.out.println(phrase);
	}
    }
}
