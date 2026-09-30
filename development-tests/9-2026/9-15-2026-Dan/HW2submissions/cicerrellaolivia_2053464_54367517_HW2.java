/*
Author:Olivia Cicerrella
Email:ocicerrella2025@fit.edu
Course:ALgor and Data Struc
Section:1
Description of this file: HW2 using recution to find if the string is a palindrome.
*/
import java.util.Scanner;
import java.io.FileNotFoundException;
import java.io.File;
import java.util.Arrays;
public class HW2{
/*
Description of each method, including parameters
*/
        //cleans the string that is in put to ensure that we can check if it 
        // is a palinsrome.
    public static String clean(String phrase){
        // the replace all [....] sectoin checks the string symbols or spaces 
        // and replaces it with "" witch is nothing just deleting it.
        return phrase.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
    }
    // method for the palindrome that checks if it is recursively by calling 
    // on itself till it gets an answer the returning it.
    public static boolean palindrome(String str) {
      if (str.length() <= 1) {
          return true;
      } if (str.charAt(0) == str.charAt(str.length()-1)) {
          return palindrome(str.substring(1, str.length()-1));
      } else return false;
    }
    // uses recurtion to go throught the algorithum 
    // this is apart of the second task or step for the homework
    //test for alphabetical order 
    public static void backwards(String[] wordList, boolean[] visited, int wordsUsed, int target, String current){
        // base case adds the number of words.
        if (wordsUsed == target) {
            if(palindrome(clean(current))){
                System.out.println(current);
            }
            return; 
        }
        // loop through every word in the sorted section. 
        for (int i = 0; i < wordList.length; i++){
            // if true that word is already in the currrent phrase
            if (!visited[i]) {
                visited[i] = true;
                
                String next;
                // if it is blank then that it is goinf to be the first word.
                if (current.isEmpty()) {
                    next = wordList[i];
                } else {
                    next = current + " " + wordList[i];
                }
                // this is the recurse step to get the next word.
                backwards(wordList, visited, wordsUsed +1, target, next);
                
                visited[i]=false;
                
            }
        }
    }
    public static void main(String[] args){
    /* 
    /* description of each block (around 5-10 lines) of instructions */
    // scanner for th in put
    // verables and the scanner in in the code to that i can better understadn teh process.
        int palPhraseLength = 0; 
        int totalCount = 0;
        String [] tempArray = new String[1000];
        // ensure that the file is safely used and read
        try {
            File file = new File(args[0]);
            // this is the scanner
            Scanner fileScan = new Scanner(file);
            
            if (fileScan.hasNextInt()) {
                palPhraseLength = fileScan.nextInt();
                fileScan.nextLine();
            }
            // loops through the rest of the file till the end. 
            while (fileScan.hasNextLine()) {
                String line = fileScan.nextLine();
                // ensures that the line is not empty
                if(!line.isEmpty()) {
                    tempArray[totalCount] = line;
                    totalCount += 1;
                }
            }
            // catches the errors if it cant get to the file.
        } catch (FileNotFoundException e) {
            System.out.println("file not found");
            return;
        }
        
        String[] words = new String[totalCount];
        for (int i = 0; i < totalCount; i++) {
            words[i] = tempArray[i];
        }
        
        Arrays.sort(words);
        boolean[] visited = new boolean[words.length];
        
        backwards(words, visited, 0, palPhraseLength, "");
        
    }
    
}