import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner; 
import java.util.ArrayList; 
import java.util.Collections;


/*
Author: Keidgiler Claimon 
Email: kclaimo2025@fit.edu 
Course: CSE 2010 
Section: 01 
Description: This program checks for palindromes using recursion.

*/ 
public class HW2 { 
    public static boolean checkPalindrome(String palindrome) {
        //take the word and reverse it if it matches 
        //take the first letters of each end, then check the second letters of each end, then check the third letters
        //and so on 
        palindrome = palindrome.trim(); //trims spaces in palindrome 
         
        if (palindrome.length() <= 1) { //checks if there is only 1 character left
            return true;
        }
        
        //this statement checks if the two characters at each index are not the same.
        if (palindrome.charAt(0) != palindrome.charAt(palindrome.length()-1)) {
            return false;
        } else { 
            //or else, it will keep moving closer to the middle.
            return checkPalindrome(palindrome.substring(1, palindrome.length()-1));
        } 

        

        
    } 
    //this a public ArrayList<String> that will be used by the main method and puzzleString
    public ArrayList<String> setOfWords = new ArrayList<String>();



    public static ArrayList<String> puzzleString(ArrayList<String> setOfWords, int wordsToAdd, ArrayList<String> currentPhrase) {
        //This ArrayList will hold the phraseString made in the for loop
        ArrayList<String> combineList = new ArrayList<String>();
        
        for (int i = 0; i < setOfWords.size(); i++) { 
            //Since java is pushy with parameters, this program will make a copy of setOfWords
            ArrayList<String> copy = new ArrayList<String>(setOfWords); 
            String currentWord = copy.remove(i);
            currentPhrase.add(currentWord);
            if(wordsToAdd == 1) {
                String phraseString = String.join(" ", currentPhrase); 
                //the phraseString must also be checked if it is a palindrome
                if (checkPalindrome(phraseString) == true) {
                    //add phrase to output
                    combineList.add(phraseString);
                }
                

            } else { 
                //If wordsToAdd does not equal 1, then an ArrayList<String> result
                //will subtract a word from wordsToAdd, and everything will be stored in
                //combineList
                ArrayList<String> result = puzzleString(copy, wordsToAdd - 1, currentPhrase);
                combineList.addAll(result);
            } 
            //This will remove the last phrase in curentPhrase
            currentPhrase.removeLast(); 


        } 
            
        return combineList;
    }

    public static void main(String[] args) {
        Scanner scanner; 

        try { // a try catch exception for file use
            scanner = new Scanner(new File(args[0])); 
        } 
        catch(FileNotFoundException e) {
            System.out.println("File was not found"); 
            return;
        } 
        
        int k = scanner.nextInt();
        ArrayList<String> words = new ArrayList<String>(); 
        
       
       
        
        while (scanner.hasNext()) {
            words.add(scanner.next());

        } 

        //totalPhrases will be made from puzzleString and sorted alphabetically
        ArrayList<String> totalPhrases = puzzleString(words, k, new ArrayList<String>()); 
        Collections.sort(totalPhrases); 
        
        //this for loop prints out all the phrases from the ArrayList
        for (int i = 0; i < totalPhrases.size(); i++) {
            System.out.println(totalPhrases.get(i)); 
        }


        scanner.close(); 


        

    }
}
