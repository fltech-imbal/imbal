/*

  Author: Sean Delate
  Email: sdelate2025@my.fit.edu
  Course: CSE2010
  Section: 01
  Description of this file:
  Searches a list of a bunch of words in order to find palindromes inside.
  This program uses a brute force method so  it is pretty slow.
  In addition the program will also find all palendromes of a specified character length.
  */
import java.io.File; // Needed to read files
import java.io.FileNotFoundException; // Needed to handle errors
import java.util.ArrayList; // Needed for text input
import java.util.Scanner; // Needed to use as a changable array for inputs
public class HW2
{
    private int targetLength; // is used to find the needed length for the palendrome

    /*  
    This is the recursive algorithm that will find the paendromes of a specified length 
    it does this through the se of a swap program to find every permutation of length specified
    and then checks to see if all needed lengths will work as a palendrome.
    */
    public void findPalendromesRec(String[] arr, String phrase, int first, int last, int wordCount) {
        if (wordCount == targetLength) {
            if (isPalindromeRec(phrase.replace(" ", ""), 0, phrase.replace(" ", "").length() - 1)) {
                System.out.println(phrase);
            }
            return;
        }

        if (first > last || wordCount > targetLength) {
            return;
        }

        for (int i = first; i <= last; i++) {
            swap(arr, first, i);
            String nextPhrase = phrase.isEmpty() ? arr[first] : phrase + " " + arr[first];
            findPalendromesRec(arr, nextPhrase, first + 1, last, wordCount + 1);
            swap(arr, first, i);
        }
    }

    // Swaps string at index a with the string at index b
    public String[] swap(String[] arr, int a, int b) {
        String temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
        return arr;
    }

    // This is the recursive program that checks a string to see if it is a palindrome
    public boolean isPalindromeRec(String s, int first, int last) {
        if (first >= last) {
            return true;
        }
        if (s.charAt(first) != s.charAt(last)) {
            return false;
        }
        return isPalindromeRec(s, first + 1, last - 1);
    }


    public static void main(String[] args)
    {
        HW2 commands = new HW2(); // Used to run methods

		File inputFile = new File(args[0]); // Takes the command line arguement for the file and then uses it to make a new file object to be read later

        ArrayList<String> words = new ArrayList<>(); // Will contain all inputs from the file. Is used instead of array because I dont know how long the file will be.

        try (Scanner scan = new Scanner(inputFile)) {
            commands.targetLength = Integer.parseInt(scan.nextLine());

            while (scan.hasNextLine()) {
                String word = scan.nextLine();
                if (!word.isEmpty()) {
                    words.add(word);
                }
            }
        } catch (FileNotFoundException e) {
            System.out.println("Error: file not found");
            e.printStackTrace();
            return;
        }

        String[] wordsArray = new String[words.size()]; // String array to move arraylist words into
        for (int i = 0; i < words.size(); i++) {
            wordsArray[i] = words.get(i);
        }

        commands.findPalendromesRec(wordsArray, "", 0, wordsArray.length - 1, 0);
    }
}
