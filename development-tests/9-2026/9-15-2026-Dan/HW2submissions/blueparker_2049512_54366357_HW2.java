/*
  Author:: Parker Blue
  Email: pblue2025@my.fit.edu
  Course: Algorithms and Data Structures
  Section: 01
  Description of this file: This code will take text inputs and will check whether the text, will be words, are palindromes or not and return telling you whether the word or words you have entered are indeed
  palindromes.
*/

import java.io.*;
import java.util.*;

public class HW2 {

    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.out.println("Usage: java HW2 <filename>");
            return;
        }

        BufferedReader br = new BufferedReader(new FileReader(args[0]));
        int palPhraseLength = Integer.parseInt(br.readLine().trim());

        List<String> words = new ArrayList<>();
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty()) {
                words.add(line);
            }
        }
        br.close();

        List<String> results = new ArrayList<>();
        boolean[] used = new boolean[words.size()];
        String[] current = new String[palPhraseLength];

        findPalindromes(words, used, current, 0, palPhraseLength, results);

        Collections.sort(results);
        StringBuilder sb = new StringBuilder();
        for (String phrase : results) {
            sb.append(phrase).append("\n");
        }
        System.out.print(sb);
    }

    // Recursively builds every ordered selection of 'palPhraseLength' to check if it matches the palindrome property.
    private static void findPalindromes(List<String> words, boolean[] used,
                                         String[] current, int depth,
                                         int palPhraseLength, List<String> results) {
        if (depth == palPhraseLength) {
            String phrase = String.join(" ", current);
            String concatenated = phrase.replace(" ", "");
            if (isPalindrome(concatenated, 0, concatenated.length() - 1)) {
                results.add(phrase);
            }
            return;
        }

        for (int i = 0; i < words.size(); i++) {
            if (!used[i]) {
                used[i] = true;
                current[depth] = words.get(i);
                findPalindromes(words, used, current, depth + 1, palPhraseLength, results);
                used[i] = false;
            }
        }
    }

    // Recursively checks whether word/phrase is a palindrome.
    private static boolean isPalindrome(String s, int start, int end) {
        if (start >= end) {
            return true;
        }
        if (s.charAt(start) != s.charAt(end)) {
            return false;
        }
        return isPalindrome(s, start + 1, end - 1);
    }
}
