/*
Author:Israel Caballero
Email:icaballeroes2025@my.fit.edu
Course:CSE2010-Algorithms and Data Structures
Section:01
Description of this file:Program that makes a combination list of all the words in a file of phrase length l
and then inserted into another list if the phrases are palindromes. The palindrome list is printed in alphabetical order.
*/
import java.util.Scanner;
import java.io.File;
import java.util.ArrayList;
public class HW2
{
//Checks if a string is a palindrome while disregarding if there are spaces. 
//Takes in the String s, (the string of words to check) an int leftIndex and int rightIndex that corresponds to a letter in the string
public static String checkIfPalindrome(String s, int leftIndex, int rightIndex){
    //Once all pairs have been matched return the string
    if((leftIndex == rightIndex) || (leftIndex > rightIndex)){
        return s;
    }
    //checks if there are spaces, if there is the leftindex is increased by one and the rightindex is decreased by one
    else{
        if(s.charAt(leftIndex) == ' '){
            leftIndex += 1;
        }
        if(s.charAt(rightIndex) == ' '){
            rightIndex -= 1;
        }
        //checks if the letters are the same at the indexes if they are call checkIfPalindrome again with leftIndex+1 and rightIndex-1
        if(s.charAt(leftIndex) == (s.charAt(rightIndex))){
            String result = checkIfPalindrome(s, leftIndex+1, rightIndex-1);
            return result;
        }
     //not palindrome   
     return "";
    }
}
//Method that computes the different combination of phrases and then when a combination is found it is sent to checkIfPalindrome and added to a list if its a palindrome
//Takes in remainingLength of words left in the phrase, chosenCombination(combination currently being made), and unusedWords(list of all words from the file)
public static ArrayList<String> stringCombination(int remainingLength, String chosenCombination, ArrayList<String> unusedWords){
    ArrayList<String> list= new ArrayList<>();
    String newCombination = "";
    //when the combination is found send through checkIfPalindrome.
    if(remainingLength == 0){
        String current = checkIfPalindrome(chosenCombination, 0, chosenCombination.length()-1);
        //add to list if current phrase is a palindrome
        if(!current.isEmpty()){
            list.add(current);
        }
        return list;
    }
    //goes through unusedWords to create a combination
    for(int i = 0; i < unusedWords.size();i++){
        String word = unusedWords.remove(i);
        //if the first word is being chosen simply add the word
        if(chosenCombination.equals("")){
            newCombination = chosenCombination.concat(word);
        }
        //if there are words in the list add a space
        else{
            newCombination = chosenCombination.concat( " " + word);
        }
        list.addAll(stringCombination(remainingLength-1, newCombination, unusedWords));
        unusedWords.add(i, word);
    }
    return list;
}
public static void main(String[] args)throws Exception{
    //Scanner to read the file, unusedWords ArrayList to keep all the words from the file, palindromeList ArrayList to keep all palindromes,
    //int wordsPerPhrase for how many words there should be in a phrase
    Scanner input = new Scanner(new File(args[0]));
    ArrayList<String> unusedWords = new ArrayList<>();
    ArrayList<String> palindromeList = new ArrayList<>();
    int wordsPerPhrase = 0;

    wordsPerPhrase = Integer.parseInt(input.nextLine());
    //While there is a line to read in the file
    while(input.hasNextLine()){
        unusedWords.add(input.nextLine());
    }
    //Call stringCombination to find all combinations of words and palindromes from that list
    palindromeList.addAll(stringCombination(wordsPerPhrase, "", unusedWords));
    //Sorts List
    palindromeList.sort(null);
    //Prints list
    for(int i = 0; i < palindromeList.size(); i++){
        System.out.println(palindromeList.get(i));
    }
}
}
