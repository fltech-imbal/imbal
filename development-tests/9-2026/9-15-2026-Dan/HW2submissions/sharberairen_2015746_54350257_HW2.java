/*
Author: Airen Sharber
Email: asharber2025@code01.fit.edu
Course: CSE 2010 
Section: 1
Description:

*/
import java.util.Scanner;
import java.io.*;
public class HW2 {
  public static void main(String[] args) throws Exception {
    Scanner fileReader = new Scanner(new File(args[0]));
    int palPhraseLength = fileReader.nextInt();
    
    String[] words = new String[1000];
    int numWords = 0;
    while (fileReader.hasNext()) {
      words[numWords] = fileReader.next();
      numWords++;
    }
    String[] actualWords = new String[numWords];
    for (int i = 0; i < numWords; i++) {
      actualWords[i] = words[i];
    }
    Arrays.sort(actualWords);
    boolean[] used = new boolean[numWords];
    findPalindromes(actualWords, used, "", 0, palPhraseLength);
    fileReader.close();
  }
  static boolean palindrome(String word, int left, int right) {
    if (left >= right) return true;
    else if (word.charAt(left) == ' ') {
      left++;
      return palindrome(word, left, right);
    } else if (word.charAt(right) == ' ') {
      right--;
      return palindrome(word, left, right);
    } else if (word.charAt(left) == word.charAt(right))
      return palindrome(word, left+1, right-1);
    else return false;
  }
  static void findPalindromes (String[] words, boolean[] used, String phrase,
                               int count, int palPhraseLength) {
    if (count == palPhraseLength) {
      if (palindrome(phrase, 0, phrase.length()-1)) {
        System.out.println(phrase);
      }
      return;
    }
    for (int i = 0; i < words.length; i++) {
      if (!used[i]) {
        used[i] = true;
        String newPhrase;
        if (phrase.length() == 0) newPhrase = words[i];
        else newPhrase = phrase + " " + words[i];

        findPalindromes(words, used, newPhrase, count+1, palPhraseLength);
        used[i] = false;
      }
    }
  }
}
