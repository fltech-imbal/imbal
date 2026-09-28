/*
Author: Calli Campagna 
Email:ccampagna2025@code01.fit.edu
Course:CSE2010
Section:1-4
Description of this file: 
This program finds multi-word palindromes from a list of words.
The program uses recursion to check whether a string is a palindrome
and recursion with backtracking to create possible multi-word phrases.
The palindromes found are printed in alphabetical order.
*/
import java.util.Scanner;
import java.util.ArrayList;
import java.io.File; 
import java.io.FileNotFoundException;
public class HW2{
   /*
    The checkPalindrome method recursively checks whether a string
    is a palindrome. It compares the first and last characters.
    If they are different, the string is not a palindrome.
    If they are the same, the method recursively checks the
    smaller string between those two characters.
    Parameters:word - the string being checked to determine if it is a palindrome
    Returns:true if the string is a palindrome, false otherwise.
    */   
   public static boolean checkPalindrome(String word){
       /* Base case: an empty string or one character string is a palindrome.
        */
       if (word.length() == 1 || word.length() == 0){
           return true;
       }
       /*
       If the first character is a space or punctuation, skip it when checking the palindrome.
      */
       if (!Character.isLetterOrDigit(word.charAt(0)))
       {
        return checkPalindrome(word.substring(1));
       }

       /*
       If the last character is a space or punctuation, skip it when checking the palindrome.
       */
       if (!Character.isLetterOrDigit(word.charAt(word.length() - 1)))
       {
        return checkPalindrome(word.substring(0, word.length() - 1));
       }

       /*Start at the first character of the string
        */
       int i = 0;
       char currentChar = word.charAt(i);
       /*
        Compare the first and last characters. If they are the same,
        recursively check the rest of the string between them.
       */
       if (currentChar == word.charAt(word.length() -1)){     
           return checkPalindrome(word.substring(i+1, word.length() - (i+1)));
                   
        }else{
            /*
             *If the first and last characters are different, the string is not a palindrome.
             */
            return false;
        }
       }
       
   /*
    The createWord method recursively creates phrases using the words
    provided in the input file. It adds one unused word to the current
    phrase at a time. When the phrase reaches the required number of words,
    the phrase is checked using the checkPalindrome method. If it is a
    palindrome it is added to the list of palindromes. The method then
    goes back by removing the last word and tries another word.
    Parameters: words - array containing all of the words from the input file
    palPhraseLength - number of words required in each phrase
    currentPhrase - list containing the words currently being used
    palindromes - list used to store the palindromes that are found
    */
   public void createWord(String[] words, int palPhraseLength,ArrayList<String> currentPhrase,ArrayList<String> palindromes) {
     /* 
      When the current phrase contains the required number of words
      create a single string from the words in the phrase. The spaces
      are included so that the original phrase can be printed.
      The completed phrase is then checked to determine whether it is
      a palindrome. If it is it is added to the list of results.
      */
     if (currentPhrase.size() == palPhraseLength) {

        String phrase = "";

        for (int i = 0; i < currentPhrase.size(); i++) {
            phrase += currentPhrase.get(i);

            if (i < currentPhrase.size() - 1) {
                phrase += " ";
            }
        }
        /*
         Check whether the completed phrase is a palindrome
         */
        if (checkPalindrome(phrase)) {
            palindromes.add(phrase);
        }

        return;
     }
     /*
       Try each word in the words array. A word can only be added if it
       has not already been used in the current phrase. After adding a
       word recursively call createWord to choose the next word. Once
       the recursive call finishes remove the word that was just added.
       This is the backtracking step that allows the program to try
       different combinations of words.
      */
     for (int i = 0; i < words.length; i++) {

        if (!currentPhrase.contains(words[i])) {

            currentPhrase.add(words[i]);

            createWord(words, palPhraseLength,
                       currentPhrase, palindromes);

            currentPhrase.remove(currentPhrase.size() - 1);
        }
     }
    }
   /*
    The main method reads the input file.
    The first value in the file determines how many words each palindrome
    phrase must contain. The remaining values are stored as words.
    The words are converted to a String array and the recursive search
    is started. Once all possible phrases have been checked the
    palindromes are sorted alphabetically and printed one per line.
    Parameters: args - command-line arguments containing the input file name
    */
    public static void main(String[] args)throws FileNotFoundException  {
        /*
        Create an object of the HW2 class and an ArrayList to temporarily
        store the words from the input file. A Scanner to read the file.
        The first value in the file is stored as palPhraseLength.
         */
        HW2 pal = new HW2();
        ArrayList<String> wordList = new ArrayList<>();
        int palPhraseLength = 0;
        Scanner sc = new Scanner(new File(args[0]));
        palPhraseLength = sc.nextInt();
        /*
        Read each remaining word from the input file and add it to
        wordList. The words are then converted from an ArrayList to
        a String array
         */
        while(sc.hasNext()){
            wordList.add(sc.next());   
            }
        String[] words = wordList.toArray(new String[0]);
        /*
        Create an empty currentPhrase list and create another ArrayList to store the successful
        palindromes found by the recursive search.
        */
        ArrayList<String> currentPhrase = new ArrayList<>();
        ArrayList<String> palindromes = new ArrayList<>();
        /*
        Sends the parameters colelcted from the file and main method into the method 
        */
        pal.createWord(words, palPhraseLength, currentPhrase, palindromes);
        /*
        Sort the completed palindromes in alphabetical order. 
        Then prints the palindrome
        */
        palindromes.sort(null);
        for (int i = 0; i < palindromes.size(); i++) {
          System.out.println(palindromes.get(i));
        }
   }
 }
