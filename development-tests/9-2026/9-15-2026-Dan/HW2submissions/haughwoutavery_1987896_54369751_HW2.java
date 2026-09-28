/*

  Author: Avery Haughwout
  Email: ahaughwout2024@my.fit.edu
  Course: CSE 2012
  Section:
  Description of this file: Identifies Palindromes!


  GOAL #1:
  Design a recursive algorithm that checks if a string is a
  palindrome or not. - DONE!!!
  GOAL #2:
  Design an algorithm which checks if any COMBINATIONS of words-
  provided a given list - are palindromes and returns them if they are :)
  
 */
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;
public class HW2{
    static Scanner scnr = new Scanner(System.in);
    public static void main(String[] args){
      System.out.println("THE CODE IS RUNNING I PROMISE");
      String input = scnr.nextLine();
      String[] inputs = input.split(" ");//Divides the input into workable chunks
      int num = Integer.parseInt(inputs[0]); //gets the number
      LinkedList<String> empty = new java.util.LinkedList<>(); // to make 'Used' parameter of wordPuzzleSolve happy
      LinkedList<String> toUse = new java.util.LinkedList<>(); // Collection of unused words... that being all of them
      for(int i = 1; i < inputs.length; i ++){ // Populates toUse
        toUse.add(inputs[i]);
      }
      HashSet<LinkedList<String>> goodCombos = wordPuzzleSolve(num, empty, toUse);
      /*
      HashSet<LinkedList<String>> -> Iterator<LinkedList<String>>
      So.. I currently have an Iterator<LinkedList<String>>, and I need to figure out how to sort them by 1st word...*/
      
      //THE PRINT SECTION
      List<String> sortedPhrases =  new java.util.ArrayList<>();
        //iterates the phrases to prep them for sorting
        //puts them in an arraylist
        for (LinkedList<String> phrase : goodCombos) {
            //populates sortedPhrases w the concatinated phrases
            String fullPhrase = "";
            for(int i = 0; i < phrase.size(); i++){
                fullPhrase += phrase.get(i) + " ";
            }
            sortedPhrases.add(fullPhrase);//sorts :)
        }
      //System.out.println("Before: "+sortedPhrases);
      java.util.Collections.sort(sortedPhrases);
      //System.out.println("After: "+sortedPhrases);
      for(int i = 0; i < sortedPhrases.size(); i++){
        System.out.println(sortedPhrases.get(i));
      }
    }
    //END OF PRINT SECTION

    public static boolean isPalindrome(String s){
    /*
    isPalindrome!
    INPUT: string S, any word
    METHOD: Checks if the first and last letter are the same. Afterewards, does the same for the next inner two letters
    OUTPUT: TRUE if all checks were successful
    */
      if(s.length() > 2){
        return  s.substring(0,1).equals(s.substring(s.length()-1)) && isPalindrome(s.substring(1, s.length() -1));
      }
      else if(s.length()  == 2){
        return s.substring(0,1).equals(s.substring(1));//checks only the FIRST and LAST
      }
      else{//only true if length = 2 or 0
        return true;
      }
    }

    public static HashSet<java.util.LinkedList<String>> wordPuzzleSolve(int n, LinkedList<String> used, LinkedList<String> notUsed){
      /* for(){
        -append
        -check if done
          if yes, check if solution
          if not, recur
      } */
      HashSet<java.util.LinkedList<String>> validCombos = new HashSet<>();
      if(n == 0){ //Checks if the one added was the last one
        String checkPal = "";
        for(int j = 0; j < used.size(); j++){// Makes the concatinated string
          checkPal += used.get(j);
          //System.out.println(used);
        }
        if(isPalindrome(checkPal)){ // Checks if the new concatination is a palindrome
          validCombos.add(new LinkedList<>(used));// If it is, adds the bitch
        }
      }
      for(int i = 0; i < notUsed.size(); i++){//makes a branch for each currently unused word
        LinkedList<String> newUsed = new LinkedList<>(used);
        LinkedList<String> newNotUsed = new LinkedList<>(notUsed);
        String toAdd = newNotUsed.remove(i);
        newUsed.add(toAdd);
        validCombos.addAll(wordPuzzleSolve(n-1, newUsed, newNotUsed));// recurs to add the rest of the words
      }
      return validCombos;
    }
}

