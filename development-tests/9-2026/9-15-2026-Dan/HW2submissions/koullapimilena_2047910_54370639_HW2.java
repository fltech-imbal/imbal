/*

  Author: Milena Koullapi
  Email: mkoullapi2025@my.fit.edu
  Course: CSE 2010
  Section: 01
  Description of this file: Reads a list of words and a target phrase length from a file, 
    then recursively finds and prints every possible multi-word palindrome phrase, in alphabetical order.

 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Collections;

public class HW2{

    /*
     * This method reads a filename from the command line, opens that file, reads the
     * palindrome phrase length and the word list from it. Then it finds every palindrome
     * phrase of that length, sorts them alphabetically, and prints each one.
     * @param args - command-line arguments; args[0] is expected to be the path to the input file to read.
    */
    public static void main(String[] args){
        /* 
        Filename store the name or path of the file that is going to be read.
        Thats information is received from args[0], which represents the first command-line argument 
        */
        String filename = args[0];

        ArrayList<String> wordList = new ArrayList<>();


        try ( Scanner scanner = new Scanner(new File(filename))){
            // Reads the first line of the file and converts it into an integer,
            // This variable represents how many words each palindrome phrase should contain.
            int palPhraseLength = Integer.parseInt(scanner.nextLine());
            
            // Checks if there is another line of text left to read in the file. The loop will continue running as long as there is more lines.
            while(scanner.hasNextLine()){
                // Reads the next line of textand stores it in the variable called line.
                String line = scanner.nextLine();
                // The words are stored into wordList
                wordList.add(line);
            }

            // Runs the recursive search for all palindrome phrases of the required length, using the word list, and then storing the results.
            ArrayList<String> phrasesFound = findPalindromes(wordList, palPhraseLength);

            //Puts the found phrases into alphabetical order 
            Collections.sort(phrasesFound);

            // Prints each palindrome phrase found in a seperate line
            for(int i = 0; i < phrasesFound.size(); i++ ){
             System.out.println(phrasesFound.get(i));
            }
            
        }

        //This block only runs if the attempt to open the file specified by filename doesnt exists, cannot be opens or found in the specific path.
        catch (FileNotFoundException e) { 
        //It print "File not found" to the user
        System.out.println("File not found.");
        //it print the detailed technical details of the errors to help the developers understand where and why the error happened
        e.printStackTrace();
        }
    }

    /*
    * This method recursively checks whether a string is a palindrome by comparing characters from the inputed file
    * @param s - the (already cleaned, no spaces) string to check
    * @param left - the current left-side index being compared
    * @param right - the current right-side index being compared
    * @return true if the string is a palindrome, false otherwise
    */
    static boolean isPalidrome(String s, int left, int right){
        if(left >= right) {
            return true;
        }
        else if(s.charAt(left) != s.charAt(right)){
            return false;
        }
        else{
            return isPalidrome(s, left +1, right -1);
        }
    }

    /*
      This method is a simpler wrapper that cleans the input string first, removing spaces/punctuation,
      then starts the recursive character comparison from both ends of the string.
      @param s - the raw string to check (contain spaces/punctuation)
      @return true if the cleaned string is a palindrome, otherwise false
    */
    static boolean isPalidrome(String s){
        String cleaned = noSpace(s);
        return isPalidrome(cleaned, 0, cleaned.length()-1);
    }

    /*
    This method recursively builds a new string containing only the letters
    from the original string, removing spaces, punctuation, and any other non-letter characters.
    @param s - the original string to clean
    @return a new string with only the letter characters kept, in order
    */
    static String noSpace(String s){
        if(s.length() == 0){
            return "";
        }
        else if(Character.isLetter(s.charAt(0))){
            return s.charAt(0) + noSpace(s.substring(1));
        }
        else{
            return noSpace(s.substring(1));
        }
    }

    
    /*
     This method is a wrapper method that sets up the empty tracking lists
     needed for the search, words picked so far, and phrases found so far, then starts the recursive search for palindrome phrases.
     @param wordList - the full list of available words to build phrases from
     @param palPhraseLength - how many words each palindrome phrase should contain
     @return a list of all palindrome phrases found
    */
    static ArrayList<String> findPalindromes(ArrayList<String> wordList, int palPhraseLength){

    ArrayList<String> wordsPicked = new ArrayList<>();
    
    ArrayList<String> phrasesFound = new ArrayList<>();

    findLongerPalindromes(wordsPicked, wordList, phrasesFound, palPhraseLength);

    return phrasesFound;

    }

    /*
    This method recursively builds every possible ordered combination of palPhraseLength unique words from wordList,
    checking each complete combination to see if it forms a palindrome phrase.
    @param wordsPicked - the words chosen so far, in order
    @param wordList - the full list of available words
    @param phrasesFound - collects any palindrome phrases discovered
    @param palPhraseLength - the target number of words per phrase
    */ 
    static void findLongerPalindromes(ArrayList<String> wordsPicked, ArrayList<String> wordList,ArrayList<String> phrasesFound, int palPhraseLength ){
        // Base case: enough words have been picked. Combine them into one
        // phrase and check if that phrase is a palindrome; if so, save it.
        if(wordsPicked.size() == palPhraseLength){
            String combinedPhrase = "";
            for(int i = 0; i < wordsPicked.size(); i++){
                combinedPhrase += wordsPicked.get(i);
                if(i < wordsPicked.size() -1){
                    combinedPhrase += " ";
                }
            }
            
            if(isPalidrome(combinedPhrase)){
                phrasesFound.add(combinedPhrase);
                
            }
            
            return;
        }
        // Recursive case: more words are still needed. When each word in the wordList that hasn't already been picked, 
        // add it, recurse to fill the remaining slots, then remove it again to try the next word.
        else{
                for(int i = 0; i < wordList.size(); i++){
                    String currentWord = wordList.get(i);
                    if(wordsPicked.contains(currentWord)){
                        continue;
                    }
                    else{
                        wordsPicked.add(currentWord);
                        findLongerPalindromes(wordsPicked, wordList, phrasesFound, palPhraseLength);
                        wordsPicked.remove(currentWord);
                    }
                }
        }
    }
}
