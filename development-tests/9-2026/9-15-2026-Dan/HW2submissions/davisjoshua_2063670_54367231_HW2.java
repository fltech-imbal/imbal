/**
 Author: Joshua Davis
 Email: joshuadavis2025@my.fit.edu
 Course: CSE 2010 - Algorithms & Data Structures
 Section: 4

 Description:
 This Program takes an input argument at Program run
 that given a phrase length and a set of words
 Finds every permutation of words that is the phrase length
 and checks if that phrase is a palindrome
 */

import java.io.FileNotFoundException;
import java.io.File;
import java.util.Scanner;
import java.util.ArrayList;

public class HW2 {
    public static void main(String[] args) throws FileNotFoundException {

        //This takes the first argument as an input file
        File input = new File(args[0]);
        Scanner reader = new Scanner(input);

        //The first line of the input file is the phrase length this stores it in a variable
        int palPhraseLength = Integer.parseInt(reader.nextLine());


        //Here 2 Array lists are created, one for the words given with the input.
        // The second for the list of found palindromes
        ArrayList<String> words = new ArrayList<>();
        ArrayList<String> palindromes = new ArrayList<>();

        //This adds all the input words into the created array list
        while (reader.hasNextLine()) {
            String line = reader.nextLine();
            words.add(line);
        }

        //This initializes the phrase to check if it is a palindrome to empty before the method runs.
        //Then the handleWordPermutations, given the given words, phrase length, our empty current phrase, and our
        //initially empty palindrome list, finds all the palindromes and puts them in the list
        String currentPhrase = "";
        handleWordPermutations(words, palPhraseLength, currentPhrase, palindromes);

        //This sorts the palindromes found into the required alphabetical/lexicographical format
        palindromes.sort(null);

        //Once the palindromes are found this prints all of them as output
        for (String palindrome : palindromes) {
            System.out.println(palindrome);
        }
    }
    //This is the palindrome checker the parameter is a string
    public static boolean isPalindrome(String str) {
        //if the string is 1 or less characters return true
        if (str.length() <= 1) {
            return true;
        }
        //if not then check if the first and last characters match, if true then
        //check if the substring between the two letters is a palindrome otherwise return false
        else  {
            if(str.charAt(0) == str.charAt(str.length()-1)) {
                return isPalindrome(str.substring(1, str.length()-1));
            }
            else {
                return false;
            }
        }
    }
    //this method recursively finds all the permutations given words, phrase length, a currentPhrase variable, and the palindrome list
    //if the phrase length is 0 then check if the phrase is a palindrome, if it is add the phrase to the palindrome list
    //otherwise it will move to the else block and select more words until the phrase length is 0.
    public static void handleWordPermutations(ArrayList<String> words, int phraseLength, String currentPhrase, ArrayList<String> palindromes) {
        if (phraseLength == 0) {
            String palindromeCheck = currentPhrase.replace(" ", "");
            if(isPalindrome(palindromeCheck)) {
                palindromes.add(currentPhrase);
            }
        }
        //the loop picks a word, adds it to the current phrase variable, then removes it from the available words temporarily
        //then calls itself with the available words left after removal, subtracts 1 from the phrase length, and passes the built phrase to continue building
        //as well as passing through the palindrome list
        else {
            for (int i = 0; i < words.size(); i++) {
                String wordChosen = words.get(i);
                String newBuiltPhrase = currentPhrase + wordChosen + " ";
                words.remove(wordChosen);
                handleWordPermutations(words, phraseLength - 1, newBuiltPhrase, palindromes);
                words.add(i, wordChosen);
            }
        }
    }
}
