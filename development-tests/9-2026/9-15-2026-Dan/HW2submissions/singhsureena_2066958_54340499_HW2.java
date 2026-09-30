/*
Author: Sureena Singh
Email: sureena2025@my.fit.edu
Course: CSE2010
Section: 
Description of this file: HW2: recursive algorithm to check for a palindrome
*/
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
public class HW2 {
    /*this method will check if the word is a palindrome or not.
    First we create a cleaned variable which will store the word without spaces.
    We then use recursion for actually checking the word. 
    The base cases checks for the length which if it is 0 or 1 it will return true.
    Then for decomposition we reduce the recusion by one character from each end and call the method again.
    */
    public static boolean isPalindrome(String checkWord){
      String cleaned = checkWord.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
      int len = cleaned.length();
      if (len==0) return true;
      else if (len ==1) return true;
      else{
        if (cleaned.charAt(0)==(cleaned.charAt(len-1))){
            return isPalindrome(cleaned.substring(1,len-1));
        }
        else return false;
      }
    }
    /*
    The findPalindrome method creates the unique combination of words and sends it to the isPalindrome() method.
    If the palPhraseLength is zero then the required number of words are there in the string, so the isPalindrome method is called.
    If it is not zero then a word is added from the wordArray to the used Arraylist and that element is removed from nextWords Arraylist which keeps track of the words not used.
    Then the findPalindrome() method is called again with the palPhraseLength reduced by 1 and then the used array is reduced by an element.
     */
    public static void  findPalindrome(ArrayList<String> wordArray, int palPhraseLength,ArrayList<String> used, ArrayList<String> results){
        
        if (palPhraseLength ==0){
            String checkWord = String.join(" ",used);
            if(isPalindrome(checkWord)){
                if (!results.contains(checkWord)){
                    results.add(checkWord);
                }
            }
            return;
        }    
        for (int i=0; i< wordArray.size();i++){
            String chosenWord = wordArray.get(i);
            used.add(chosenWord);
            ArrayList<String> nextWords = new ArrayList<>(wordArray);
            nextWords.remove(i);
            findPalindrome(nextWords, palPhraseLength-1, used, results);
            used.remove(used.size()-1);
        }
    }
    /*
    The main method deals with storing in the input in the designated elements and then calls the findPalindrome() method.
    After calling the method is prints the output i.e the contents of the results arraylist. */
    public static void main(String[]args)throws java.io.FileNotFoundException {
     if (args.length == 0) {
            System.out.println("Please provide a valid input file name.");
            return;
        }
        
     Scanner scanner = new Scanner(new File(args[0]));
     if (!scanner.hasNextInt()){
        scanner.close();
        return;
     }
     //the used arraylist will be used to store assigned words for each combination we create
     //the results arraylist will store all the values that satisfy the condition for being a palindrome.
     ArrayList<String> wordArray = new ArrayList<>(); 
     ArrayList<String> used = new ArrayList<>(); 
     ArrayList<String> results = new ArrayList<>(); 
     int palPhraseLength = scanner.nextInt();
     //we use the wordArray ArrayList to store all the words given in the input.
     while (scanner.hasNext()){
        wordArray.add(scanner.next());
     } scanner.close();

     findPalindrome(wordArray, palPhraseLength,used,results);
     //the sort() from collections makes the sorting process of the array easier.
     //the loop prints out the each phrase in a separate line. 
     //Each element of the array is stored in a variable phrase.
     Collections.sort(results);
     for (int i=0; i<results.size();i++){
        String phrase = results.get(i);
        System.out.println(phrase);
     }
    }

    
}
